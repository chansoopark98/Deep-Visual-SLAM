"""EXP-016 (E1 part 1): one inference interface for the large feed-forward geometry models on TUM-VI windows.

Every runner has the signature `run_<model>(frames, K) -> dict`:
  frames: list of N grayscale uint8 images, 512x512 (the rectified cam0 cache, dvslam.data.tumvi.K_OUT).
  K:      3x3 pinhole intrinsics of `frames` in pixels (K_OUT: f = 190, principal point 255.5).
Returned dict (numpy, CPU):
  T_wc     (N, 4, 4) float64: camera-to-world of each frame in the model's own world, re-based so that frame 0 is the
           identity (T_wc[i] = inv(T_wc_model[0]) @ T_wc_model[i]). Camera axes are OpenCV (x right, y down,
           z forward) for every model here, so T_wc[i] is the pose of frame i in the frame-0 camera. Translations are up
           to scale (the model's own unit; `metric` says whether the model claims metres).
  depth    (N, h, w) float32 z-depth at the model resolution (same unit as T_wc), or None.
  conf     (N, h, w) float32 model confidence (model-specific meaning, larger = more confident), or None.
  K_pred   (N, 3, 3) float64 intrinsics predicted by the model, converted to the 512x512 input pixels, or None.
  res      (h, w) model input resolution.
  metric   bool: the model's translations are meant to be metres.
  seconds  forward-pass wall time on the GPU (synchronised, excludes model loading and image I/O).
  peak_gb  torch.cuda.max_memory_allocated during the forward pass, weights included.

Pre-processing (identical in spirit for all, details per runner): gray replicated to 3 channels; the full 512x512 frame
is RESIZED (never cropped) to the model's square resolution with cv2.INTER_AREA (shrink) or cv2.INTER_CUBIC (enlarge);
K is scaled with the same factor (pixel-centre convention: c' = (c + 0.5) s - 0.5). Autocast bf16 (all READMEs:
bf16 on compute capability >= 8; the RTX 5090 is 12.0); outputs are cast to float32 before any pose algebra and the
re-basing is done in float64. Exception: Anchor3R uses fp16, the dtype of its official infer.yaml.

  model        res      K input                         weights (~/park/models/hf/...)
  vggt         518x518  no (cannot take K)              facebook__VGGT-1B/model.safetensors
  pi3x         504x504  yes, `intrinsics` (use_K=False   yyfz233__Pi3X/model.safetensors
                        drops the multimodal branch)
  mapanything  518x518  yes, `intrinsics`               facebook__map-anything-apache/model.safetensors
  da3_large    504x504  no (not a documented input)     depth-anything__DA3-LARGE-1.1/model.safetensors
  da3_nested   504x504  no                              depth-anything__DA3NESTED-GIANT-LARGE-1.1/model.safetensors
  worldmirror  504x504  yes, `camera_intrs` + cond flag  tencent__HY-World-2.0/HY-WorldMirror-2.0/model.safetensors
  abot_recon   504x504  no                              acvlab__ABot-Recon/abot_recon.safetensors
  anchor3r     518x518  no                              polar-explorer__Anchor3R/Anchor3R.pt (weights_only=True)

ABot-Recon and Anchor3R are streaming models; both are run on the 8 frames as one sequence (ABot: causal stream with
its fixed 12-frame local window, state reset per call, loop closure off; Anchor3R: window_size = 8, i.e. the whole
window is its first, jointly processed chunk, loop closure off). Their official pre-processing CROPS a square frame
(ABot: width-locked resize to 504 then centre crop to 504x280; Anchor3R: long edge 518 then centre crop to 4:3);
here the full frame is resized to a square instead (no crop), a shape they were not trained on.

Resolution choices follow each official pre-processing for a 512x512 input: VGGT `load_and_preprocess_images` and
MapAnything `load_images` give 518; Pi3 `load_multimodal_data` (PIXEL_LIMIT 255000) gives 504; DA3 `InputProcessor`
(process_res 504, upper_bound_resize) gives 504; WorldMirror `compute_adaptive_target_size` (min(longest edge, 952)
rounded down to 14) gives 504. DA3 resizes through its own InputProcessor (also a resize, no crop).

Third-party code is imported from ~/park/third_party/<repo> (sys.path, nothing pip-installed from the repos) and every
checkpoint is a .safetensors file, so no pickle is loaded. Two documented substitutions avoid network / binary
installs: (1) MapAnything builds its DINOv2 encoder through `torch.hub.load('facebookresearch/dinov2', ...)`, which
downloads and executes GitHub code; `_no_hub_dinov2` replaces that call by the copy of the same DINOv2 code vendored
in map-anything (`mapanything/models/external/dinov2`, pretrained=False). All encoder weights then come from the
MapAnything checkpoint (strict load). (2) WorldMirror hard-imports `flash_attn`; `_install_flash_attn_shim` registers a
module whose `flash_attn_func(q, k, v)` (layout B, N, H, D, non-causal, default scale) is
torch.nn.functional.scaled_dot_product_attention, i.e. the same attention computed by PyTorch's own kernels.
WorldMirror is built with enable_gs=False (the 3D-Gaussian head is not used for poses); its module-level
`import gsplat` is satisfied by `_install_gsplat_stub` (raises if rendering were called). MapAnything's checkpoint
stores shared tensors once, so every checkpoint is loaded with safetensors `load_model` (see `_load`).

Smoke test (server, venv ~/park/envs/geo): `python scripts/exp016_models.py --models all --scene room5 --gaps 1 5`.
It picks, per gap, the 8-frame window of room5 with the largest mocap rotation (below 90 deg) between its first and
last frame (diagnostic use of GT only) and compares each model's frame-0 -> frame-7 relative rotation with mocap.
This file imports nothing from the legacy `vo/` or `model/` packages.
"""
import argparse
import json
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch

HOME = Path.home()
WEIGHTS = HOME / 'park/models/hf'
THIRD = HOME / 'park/third_party'
AMP = torch.bfloat16
_CACHE: dict = {}


# ---------------------------------------------------------------- shared helpers
def _add_path(p: Path) -> None:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _resize(frames: list[np.ndarray], size: int) -> np.ndarray:
    """(N, size, size, 3) uint8: gray replicated to 3 channels, whole frame resized (no crop)."""
    out = []
    for f in frames:
        if f.ndim != 2:
            raise ValueError(f'expected a grayscale frame, got shape {f.shape}')
        interp = cv2.INTER_AREA if size < f.shape[0] else cv2.INTER_CUBIC
        g = cv2.resize(f, (size, size), interpolation=interp) if size != f.shape[0] else f
        out.append(np.repeat(g[..., None], 3, axis=2))
    return np.stack(out)


def _scale_K(K: np.ndarray, s: float) -> np.ndarray:
    Ks = np.asarray(K, dtype=np.float64).copy()
    Ks[0, 0] *= s
    Ks[1, 1] *= s
    Ks[0, 2] = (Ks[0, 2] + 0.5) * s - 0.5
    Ks[1, 2] = (Ks[1, 2] + 0.5) * s - 0.5
    return Ks


def _to_unit(rgb: np.ndarray, device: str = 'cuda') -> torch.Tensor:
    """(N, H, W, 3) uint8 -> (1, N, 3, H, W) float32 in [0, 1]."""
    return torch.from_numpy(rgb).to(device).permute(0, 3, 1, 2).float().div(255.0)[None]


def _rebase(T_wc: np.ndarray) -> np.ndarray:
    T_wc = np.asarray(T_wc, dtype=np.float64)
    if T_wc.shape[-2:] == (3, 4):
        T_wc = np.concatenate([T_wc, np.broadcast_to([0.0, 0.0, 0.0, 1.0], T_wc.shape[:-2] + (1, 4))], axis=-2)
    return np.linalg.inv(T_wc[0]) @ T_wc


def _w2c_to_T_wc(E: np.ndarray) -> np.ndarray:
    E = np.asarray(E, dtype=np.float64)
    if E.shape[-2:] == (3, 4):
        E = np.concatenate([E, np.broadcast_to([0.0, 0.0, 0.0, 1.0], E.shape[:-2] + (1, 4))], axis=-2)
    return np.linalg.inv(E)


def _timed(fn):
    """Run fn() on the GPU; return (result, seconds, peak GB)."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0, torch.cuda.max_memory_allocated() / 1e9


def _load(name: str, module: torch.nn.Module, path: Path, allow_unexpected: str | None = None) -> None:
    """safetensors `load_model`: restores tensors that the file stores once for several aliases (MapAnything, DA3,
    WorldMirror checkpoints list them in the file metadata). No missing key is allowed; unexpected keys only with the
    given prefix (heads deliberately not built)."""
    from safetensors.torch import load_model
    missing, unexpected = load_model(module, str(path), strict=False)
    print(f'[{name}] load_model: {len(missing)} missing, {len(unexpected)} unexpected'
          + (f' (all {allow_unexpected}*)' if unexpected and allow_unexpected else ''), flush=True)
    bad = [k for k in unexpected if not (allow_unexpected and k.startswith(allow_unexpected))]
    if missing or bad:
        raise RuntimeError(f'{name}: checkpoint mismatch, missing {missing[:5]}, unexpected {bad[:5]}')


def unload() -> None:
    _CACHE.clear()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- VGGT-1B
def _vggt():
    if 'vggt' not in _CACHE:
        _add_path(THIRD / 'vggt')
        from vggt.models.vggt import VGGT
        m = VGGT(enable_track=False)
        _load('vggt', m, WEIGHTS / 'facebook__VGGT-1B/model.safetensors', allow_unexpected='track_head.')
        _CACHE['vggt'] = m.eval().cuda()
    return _CACHE['vggt']


def run_vggt(frames: list[np.ndarray], K: np.ndarray) -> dict:
    """VGGT-1B: pose_enc -> extrinsic (camera-from-world, OpenCV) -> T_wc = inv. K is not an input (VGGT #421)."""
    m = _vggt()
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    size = 518
    x = _to_unit(_resize(frames, size))

    def fwd():
        with torch.autocast('cuda', dtype=AMP):
            return m(x)
    p, sec, peak = _timed(fwd)
    extr, intr = pose_encoding_to_extri_intri(p['pose_enc'].float(), (size, size))
    s = frames[0].shape[0] / size
    K_pred = intr[0].double().cpu().numpy()
    K_pred = np.stack([_scale_K(k, s) for k in K_pred])
    return {'T_wc': _rebase(_w2c_to_T_wc(extr[0].cpu().numpy())), 'depth': p['depth'][0, ..., 0].float().cpu().numpy(),
            'conf': p['depth_conf'][0].float().cpu().numpy(), 'K_pred': K_pred, 'res': (size, size), 'metric': False,
            'seconds': sec, 'peak_gb': peak}


# ---------------------------------------------------------------- Pi3X
def _pi3x(use_K: bool):
    key = f'pi3x_{use_K}'
    if key not in _CACHE:
        _add_path(THIRD / 'Pi3')
        from pi3.models.pi3x import Pi3X
        m = Pi3X(use_multimodal=True)
        _load('pi3x', m, WEIGHTS / 'yyfz233__Pi3X/model.safetensors')
        if not use_K:
            m.disable_multimodal()        # what Pi3X.from_pretrained users do without conditions (example_mm.py)
        _CACHE[key] = m.eval().cuda()
    return _CACHE[key]


def run_pi3x(frames: list[np.ndarray], K: np.ndarray, use_K: bool = True) -> dict:
    """Pi3X: camera_poses are camera-to-world, OpenCV (pi3x.py docstring). `metric` head gives approximate metres."""
    size = 504
    m = _pi3x(use_K)
    x = _to_unit(_resize(frames, size))
    kw = {}
    if use_K:
        Ks = _scale_K(K, size / frames[0].shape[0])
        kw['intrinsics'] = torch.from_numpy(np.broadcast_to(Ks, (len(frames), 3, 3)).copy()).float().cuda()[None]

    def fwd():
        with torch.autocast('cuda', dtype=AMP):
            return m(imgs=x, **kw)
    p, sec, peak = _timed(fwd)
    # forward_head already multiplies camera translations and local points by the `metric` head (approximate metres)
    return {'T_wc': _rebase(p['camera_poses'][0].float().cpu().numpy()),
            'depth': p['local_points'][0, ..., 2].float().cpu().numpy(),
            'conf': torch.sigmoid(p['conf'][0, ..., 0].float()).cpu().numpy(), 'K_pred': None, 'res': (size, size),
            'metric': True, 'seconds': sec, 'peak_gb': peak}


# ---------------------------------------------------------------- MapAnything (apache weights)
class _no_hub_dinov2:
    """Context manager: torch.hub.load('facebookresearch/dinov2', name, ...) -> vendored DINOv2, pretrained=False."""

    def __enter__(self):
        from mapanything.models.external.dinov2.hub import backbones
        self._orig = torch.hub.load

        def load(repo, name, *a, **kw):
            if repo != 'facebookresearch/dinov2':
                raise RuntimeError(f'blocked torch.hub.load({repo!r})')
            return getattr(backbones, name)(pretrained=False)
        torch.hub.load = load
        return self

    def __exit__(self, *exc):
        torch.hub.load = self._orig


def _mapanything():
    if 'mapanything' not in _CACHE:
        _add_path(THIRD / 'map-anything')
        from mapanything.models.mapanything.model import MapAnything
        d = WEIGHTS / 'facebook__map-anything-apache'
        cfg = json.loads((d / 'config.json').read_text())
        cfg['encoder_config'] = dict(cfg['encoder_config'], torch_hub_pretrained=False)
        with _no_hub_dinov2():
            m = MapAnything(**cfg)
        _load('mapanything', m, d / 'model.safetensors')
        _CACHE['mapanything'] = m.eval().cuda()
    return _CACHE['mapanything']


def run_mapanything(frames: list[np.ndarray], K: np.ndarray, use_K: bool = True) -> dict:
    """MapAnything: per-view `camera_poses` are camera-to-world, OpenCV; outputs are metric (scale token)."""
    from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
    size = 518
    m = _mapanything()
    x = _to_unit(_resize(frames, size))[0]
    norm = IMAGE_NORMALIZATION_DICT['dinov2']
    x = (x - norm.mean.view(1, 3, 1, 1).cuda()) / norm.std.view(1, 3, 1, 1).cuda()
    Ks = torch.from_numpy(_scale_K(K, size / frames[0].shape[0])).float().cuda()[None]
    views = []
    for i in range(len(frames)):
        v = {'img': x[i:i + 1], 'data_norm_type': ['dinov2']}
        if use_K:
            v['intrinsics'] = Ks
        views.append(v)

    def fwd():
        return m.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype='bf16', apply_mask=True,
                       mask_edges=True)
    out, sec, peak = _timed(fwd)
    T = np.stack([o['camera_poses'][0].float().cpu().numpy() for o in out])
    depth = np.stack([o['depth_z'][0, ..., 0].float().cpu().numpy() for o in out])
    conf = np.stack([o['conf'][0].float().cpu().numpy() for o in out])
    s = frames[0].shape[0] / size
    K_pred = np.stack([_scale_K(o['intrinsics'][0].double().cpu().numpy(), s) for o in out])
    return {'T_wc': _rebase(T), 'depth': depth, 'conf': conf, 'K_pred': K_pred, 'res': (size, size), 'metric': True,
            'seconds': sec, 'peak_gb': peak}


# ---------------------------------------------------------------- Depth Anything 3 (any-view and nested)
def _da3(name: str):
    if name not in _CACHE:
        _add_path(THIRD / 'Depth-Anything-3/src')
        from depth_anything_3.cfg import create_object, load_config
        from depth_anything_3.registry import MODEL_REGISTRY
        repo = {'da3-large': 'depth-anything__DA3-LARGE-1.1',
                'da3nested-giant-large': 'depth-anything__DA3NESTED-GIANT-LARGE-1.1'}[name]
        holder = torch.nn.Module()           # checkpoint keys are 'model.<...>' (api.DepthAnything3.model = the net)
        holder.model = create_object(load_config(MODEL_REGISTRY[name]))
        _load(name, holder, WEIGHTS / repo / 'model.safetensors')
        _CACHE[name] = holder.model.eval().cuda()
    return _CACHE[name]


def _run_da3(name: str, frames: list[np.ndarray], metric: bool) -> dict:
    """DA3: extrinsics are camera-from-world (OpenCV, api.py `_normalize_extrinsics`) -> T_wc = inv."""
    net = _da3(name)
    from depth_anything_3.utils.io.input_processor import InputProcessor
    from depth_anything_3.utils.io.output_processor import OutputProcessor
    rgb = list(_resize(frames, frames[0].shape[0]))          # gray -> 3 channels only; InputProcessor resizes to 504
    imgs, _, _ = InputProcessor()(rgb, None, None, 504, 'upper_bound_resize', sequential=True, desc=None)
    x = (imgs if imgs.dim() == 5 else imgs[None]).float().cuda()

    def fwd():
        with torch.autocast('cuda', dtype=AMP):
            return net(x, None, None, [], False, False, 'saddle_balanced')
    raw, sec, peak = _timed(fwd)
    pred = OutputProcessor()(raw)
    h, w = x.shape[-2:]
    s = frames[0].shape[0] / w
    K_pred = None if pred.intrinsics is None else np.stack([_scale_K(k, s) for k in pred.intrinsics])
    return {'T_wc': _rebase(_w2c_to_T_wc(pred.extrinsics)), 'depth': np.asarray(pred.depth, np.float32),
            'conf': None if pred.conf is None else np.asarray(pred.conf, np.float32), 'K_pred': K_pred, 'res': (h, w),
            'metric': metric, 'seconds': sec, 'peak_gb': peak}


def run_da3_large(frames: list[np.ndarray], K: np.ndarray) -> dict:
    """DA3-LARGE-1.1 (any-view, relative scale). K unused (not a documented input)."""
    return _run_da3('da3-large', frames, metric=False)


def run_da3_nested(frames: list[np.ndarray], K: np.ndarray) -> dict:
    """DA3NESTED-GIANT-LARGE-1.1 (giant any-view + metric branch: metres). K unused."""
    return _run_da3('da3nested-giant-large', frames, metric=True)


# ---------------------------------------------------------------- WorldMirror-2.0 (HY-World-2.0)
def _install_flash_attn_shim() -> None:
    if 'flash_attn.flash_attn_interface' in sys.modules:
        return
    import torch.nn.functional as F

    def flash_attn_func(q, k, v, dropout_p=0.0, **_):
        # flash-attn layout (B, N, H, D) -> SDPA layout (B, H, N, D) and back; non-causal, softmax scale 1/sqrt(D)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p)
        return o.transpose(1, 2)
    pkg = types.ModuleType('flash_attn')
    sub = types.ModuleType('flash_attn.flash_attn_interface')
    sub.flash_attn_func = flash_attn_func
    pkg.flash_attn_interface = sub
    sys.modules['flash_attn'] = pkg
    sys.modules['flash_attn.flash_attn_interface'] = sub


def _install_gsplat_stub() -> None:
    """worldmirror.py imports its Gaussian-splat renderer (gsplat) at module level; with enable_gs=False it is never
    built. The stub only satisfies the import and raises if rendering were ever called."""
    if 'gsplat' in sys.modules:
        return

    def rasterization(*_a, **_k):
        raise RuntimeError('gsplat stub: Gaussian-splat rendering is disabled in exp016_models')

    class DefaultStrategy:
        pass
    pkg, rendering, strategy = (types.ModuleType(n) for n in ('gsplat', 'gsplat.rendering', 'gsplat.strategy'))
    rendering.rasterization = rasterization
    strategy.DefaultStrategy = DefaultStrategy
    pkg.rendering, pkg.strategy = rendering, strategy
    sys.modules.update({'gsplat': pkg, 'gsplat.rendering': rendering, 'gsplat.strategy': strategy})


def _worldmirror():
    if 'worldmirror' not in _CACHE:
        _add_path(THIRD / 'HY-World-2.0')
        _install_flash_attn_shim()
        _install_gsplat_stub()
        from hyworld2.worldrecon.hyworldmirror.models.models.worldmirror import (
            WorldMirror,
        )
        d = WEIGHTS / 'tencent__HY-World-2.0/HY-WorldMirror-2.0'
        cfg = dict(json.loads((d / 'config.json').read_text()), enable_gs=False)
        m = WorldMirror(**cfg)
        _load('worldmirror', m, d / 'model.safetensors', allow_unexpected='gs_')
        _CACHE['worldmirror'] = m.eval().cuda()
    return _CACHE['worldmirror']


def run_worldmirror(frames: list[np.ndarray], K: np.ndarray, use_K: bool = True) -> dict:
    """WorldMirror-2.0: camera_poses are camera-to-world (inverse of the predicted w2c), OpenCV."""
    size = 504
    m = _worldmirror()
    views = {'img': _to_unit(_resize(frames, size))}
    cond = [0, 0, 0]
    if use_K:
        Ks = _scale_K(K, size / frames[0].shape[0])
        views['camera_intrs'] = torch.from_numpy(np.broadcast_to(Ks, (len(frames), 3, 3)).copy()).float().cuda()[None]
        cond[2] = 1

    def fwd():
        with torch.autocast('cuda', dtype=AMP):
            return m(views=views, cond_flags=cond, is_inference=True)
    p, sec, peak = _timed(fwd)
    s = frames[0].shape[0] / size
    K_pred = np.stack([_scale_K(k, s) for k in p['camera_intrs'][0].double().cpu().numpy()])
    depth = p['depth'][0].float()
    depth = depth[..., 0] if depth.shape[-1] == 1 else depth
    conf = p.get('depth_conf')
    return {'T_wc': _rebase(p['camera_poses'][0].float().cpu().numpy()), 'depth': depth.cpu().numpy(),
            'conf': None if conf is None else conf[0].float().cpu().numpy(), 'K_pred': K_pred, 'res': (size, size),
            'metric': False, 'seconds': sec, 'peak_gb': peak}


# ---------------------------------------------------------------- ABot-Recon (streaming, adjacent relative pose)
def _abot():
    if 'abot' not in _CACHE:
        _add_path(THIRD / 'ABot-Recon')
        from abot_recon.config import InferenceConfig
        from abot_recon.model import ReleasedABotReconModel
        cfg = InferenceConfig(checkpoint=WEIGHTS / 'acvlab__ABot-Recon/abot_recon.safetensors', height=504, width=504,
                              attention_backend='sdpa', loop_closure=False, max_frames=64)
        _CACHE['abot'] = ReleasedABotReconModel(cfg)     # loads the safetensors checkpoint itself (strict mapping)
    return _CACHE['abot']


def run_abot_recon(frames: list[np.ndarray], K: np.ndarray) -> dict:
    """ABot-Recon: `camera_poses` are camera-to-world (abot_recon.geometry.relative_from_c2w), OpenCV."""
    m = _abot()
    from abot_recon.preprocessing import preprocess_image
    size = 504
    xs = [preprocess_image(rgb, height=size, width=size)[0] for rgb in _resize(frames, frames[0].shape[0])]

    def fwd():
        m.reset()
        stream = (x[None, None].to('cuda', dtype=AMP) for x in xs)
        with torch.autocast('cuda', dtype=AMP):
            return m.network.inference_stream_iter(stream, num_frames=len(xs),
                                                   output_keys=['camera_poses', 'local_points', 'conf'],
                                                   dense_output_indices=None)
    out, sec, peak = _timed(fwd)
    conf = out.get('conf')
    return {'T_wc': _rebase(out['camera_poses'][0].float().cpu().numpy()),
            'depth': out['local_points'][0, ..., 2].float().cpu().numpy(),
            'conf': None if conf is None else torch.sigmoid(conf[0].float().squeeze(-1)).cpu().numpy(),
            'K_pred': None, 'res': (size, size), 'metric': False, 'seconds': sec, 'peak_gb': peak}


# ---------------------------------------------------------------- Anchor3R (streaming, VGGT-based)
def _install_sksparse_stub() -> None:
    """anchor3r.geometry.camera imports scikit-sparse (CHOLMOD, a compiled SuiteSparse binding) at module level for
    multi-window rotation / position averaging. A single chunk never reaches it (stitch_camera_graph returns the
    window's own poses); the stub only satisfies the import and raises if it were called."""
    if 'sksparse' in sys.modules:
        return

    def _blocked(*_a, **_k):
        raise RuntimeError('sksparse stub: multi-window pose averaging is not available in exp016_models')
    pkg, chol = types.ModuleType('sksparse'), types.ModuleType('sksparse.cholmod')
    chol.analyze = chol.cholesky = _blocked
    pkg.cholmod = chol
    sys.modules.update({'sksparse': pkg, 'sksparse.cholmod': chol})


def _anchor3r():
    if 'anchor3r' not in _CACHE:
        _add_path(THIRD / 'Anchor3R')
        _install_sksparse_stub()
        import yaml
        from anchor3r.runtime.model import Anchor3RModel
        cfg = yaml.safe_load((THIRD / 'Anchor3R/anchor3r/configs/infer.yaml').read_text())['model']
        # Anchor3RModel.load_checkpoint: torch.load(..., weights_only=True), strict load_state_dict
        m = Anchor3RModel(cfg['sampler_cfg'], checkpoint=str(WEIGHTS / 'polar-explorer__Anchor3R/Anchor3R.pt'),
                          strict_load=True)
        _CACHE['anchor3r'] = m.eval().cuda()
    return _CACHE['anchor3r']


def run_anchor3r(frames: list[np.ndarray], K: np.ndarray) -> dict:
    """Anchor3R: decode_cam_map gives camera-from-world (w2c, OpenCV) -> T_wc = inv. fp16 autocast (infer.yaml)."""
    m = _anchor3r()
    from anchor3r.geometry.camera import build_camera_trajectories, decode_cam_map
    size, n = 518, len(frames)
    x = _to_unit(_resize(frames, size))

    def fwd():
        state = m.new_state()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            out = m.forward_chunk(x, window_size=n, chunk_idx=0, state=state)
        m.advance_state(state, is_last=True)
        return out
    out, sec, peak = _timed(fwd)
    fov = [out['fov'].float()] if 'fov' in out else []
    traj = build_camera_trajectories([out['chunk_cam_map'].float()], fov, num_frames=n, window_size=n)
    w2c, intri = decode_cam_map(traj['online'], (size, size))
    s = frames[0].shape[0] / size
    dpt = out.get('dpt_map')
    depth = None if dpt is None else dpt[0, :, :, 0].float().reshape(n, size, size).cpu().numpy()
    cnf = out.get('dpt_cnf')
    conf = None if cnf is None else cnf[0].float().reshape(n, size, size).cpu().numpy()
    return {'T_wc': _rebase(_w2c_to_T_wc(w2c[0].cpu().numpy())), 'depth': depth, 'conf': conf,
            'K_pred': np.stack([_scale_K(k, s) for k in intri[0].double().cpu().numpy()]), 'res': (size, size),
            'metric': False, 'seconds': sec, 'peak_gb': peak}


MODELS = {
    'vggt': run_vggt,
    'pi3x': run_pi3x,
    'mapanything': run_mapanything,
    'da3_large': run_da3_large,
    'da3_nested': run_da3_nested,
    'worldmirror': run_worldmirror,
    'abot_recon': run_abot_recon,
    'anchor3r': run_anchor3r,
}


# ---------------------------------------------------------------- smoke test (diagnostic use of mocap only)
def _rot_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def _rotvec(R: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(R, np.float64))[0].ravel()


def _sanity(T_pred: np.ndarray, T_gt: np.ndarray) -> dict:
    """Compare frame-0 -> frame-(N-1) relative rotation and translation direction (both in the frame-0 camera)."""
    Rp, Rg = T_pred[-1, :3, :3], T_gt[-1, :3, :3]
    vp, vg = _rotvec(Rp), _rotvec(Rg)
    tp, tg = T_pred[-1, :3, 3], T_gt[-1, :3, 3]
    pair_err = [_rot_deg((np.linalg.inv(T_gt[i]) @ T_gt[i + 1])[:3, :3].T
                         @ (np.linalg.inv(T_pred[i]) @ T_pred[i + 1])[:3, :3]) for i in range(len(T_gt) - 1)]
    out = {'rot_gt_deg': _rot_deg(Rg), 'rot_pred_deg': _rot_deg(Rp), 'rot_err_deg': _rot_deg(Rg.T @ Rp),
               'axis_cos': float(vp @ vg / (np.linalg.norm(vp) * np.linalg.norm(vg) + 1e-12)),
               'trans_dir_cos': float(tp @ tg / (np.linalg.norm(tp) * np.linalg.norm(tg) + 1e-12)),
               'pair_rot_err_med_deg': float(np.median(pair_err))}
    out['pass'] = bool(out['axis_cos'] > 0.9 and abs(out['rot_pred_deg'] - out['rot_gt_deg']) < 0.2 * out['rot_gt_deg'])
    return out


def _pick_window(T_wc: np.ndarray, valid: np.ndarray, gap: int, n: int, max_deg: float = 90.0) -> int:
    """Start of the window with the largest first-to-last rotation below max_deg (near 180 deg the axis sign is
    ill-defined, so the axis check would be meaningless)."""
    span = gap * (n - 1)
    best, best_i = -1.0, 0
    for i in range(len(T_wc) - span):
        if not valid[i:i + span + 1:gap].all():
            continue
        a = _rot_deg((np.linalg.inv(T_wc[i]) @ T_wc[i + span])[:3, :3])
        if best < a < max_deg:
            best, best_i = a, i
    return best_i


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--models', nargs='+', default=['all'])
    ap.add_argument('--scene', default='room5')
    ap.add_argument('--gaps', nargs='+', type=int, default=[1, 5])
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--repeats', type=int, default=3, help='timed repeats after one warm-up run')
    ap.add_argument('--out', default='runs/eval/exp016_setup/smoke.json')
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dvslam.data import tumvi
    seq = tumvi.load_sequence(args.scene, load_images=False)
    valid = np.asarray(seq.gt_valid, bool)
    windows = {}
    for gap in args.gaps:
        i0 = _pick_window(seq.T_wc, valid, gap, args.n)
        idx = list(range(i0, i0 + gap * (args.n - 1) + 1, gap))
        frames = [cv2.imread(str(seq.image_paths[i]), cv2.IMREAD_GRAYSCALE) for i in idx]
        windows[gap] = (idx, frames, _rebase(seq.T_wc[idx]))

    variants = []
    for name in (list(MODELS) if args.models == ['all'] else args.models):
        variants.append((name, {}))
        if name in ('pi3x', 'worldmirror', 'mapanything'):
            variants.append((name, {'use_K': False}))
    results = []
    for name, kw in variants:
        tag = name + ('' if not kw else '_noK')
        for gap, (idx, frames, T_gt) in windows.items():
            try:
                MODELS[name](frames, tumvi.K_OUT, **kw)                 # warm-up (includes model loading)
                runs = [MODELS[name](frames, tumvi.K_OUT, **kw) for _ in range(args.repeats)]
            except Exception as e:                       # noqa: BLE001  (recorded in the output, next model runs)
                print(f'[{tag}] gap {gap}: FAILED {type(e).__name__}: {e}', flush=True)
                results.append({'model': tag, 'gap': gap, 'error': f'{type(e).__name__}: {e}'})
                continue
            r = runs[-1]
            row = dict(model=tag, gap=gap, frames=idx, res=list(r['res']), metric=r['metric'],
                       T_shape=list(r['T_wc'].shape), T0_is_identity=bool(np.allclose(r['T_wc'][0], np.eye(4))),
                       depth_shape=None if r['depth'] is None else list(r['depth'].shape),
                       conf_shape=None if r['conf'] is None else list(r['conf'].shape),
                       f_pred_px=None if r['K_pred'] is None else float(np.mean(r['K_pred'][:, 0, 0])),
                       seconds_median=float(np.median([x['seconds'] for x in runs])),
                       peak_gb=float(max(x['peak_gb'] for x in runs)),
                       pose_repeat_max_diff=float(max(np.abs(x['T_wc'] - r['T_wc']).max() for x in runs)),
                       **_sanity(r['T_wc'], T_gt))
            print(json.dumps(row), flush=True)
            results.append(row)
        unload()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'scene': args.scene, 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
                                   'results': results}, indent=1))
    print(f'wrote {out}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
