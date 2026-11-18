"""DAv2-S metric depth engine: fixed-shape ONNX export, TensorRT build, runners and depth parity (plan v3 item 5.2).

Export recipe (research 16 §7, plan 5.2):
- `torch.onnx.export(..., dynamo=False, opset_version=17)`, fixed 1x3xHxW input, no dynamic axes.
- Classic attention: DINOv2 `Attention.forward` (explicit q·kᵀ softmax). xFormers is forced off, no SDPA node.
- Position embeddings: the bicubic interpolation for the fixed patch grid is evaluated once in PyTorch and stored as
  a constant, so the graph has no Resize on the position table.
- 1/4 output: the DPT head's last upsample targets the 1/4-resolution camera grid (120x160 for 640x480, DPVO's
  RES=4 grid, plan §2.2) instead of the network input size, so `output_conv2` runs at 1/4 resolution and the engine
  writes 19k floats per frame. `out_hw=None` keeps the original full network-resolution head.

Engine input: ImageNet-normalised RGB float32 (1, 3, H, W) from `preprocess` (cv2 INTER_LINEAR resize, identical to
`dvslam.train.depth_data.prepare_image` at 476x630). Engine output: depth in metres (1, OUT_H, OUT_W) for the
canonical camera; runners divide by `factor` (canon_factor(K); 1.0 on Redwood).
"""
import hashlib
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dvslam.depth import WEIGHTS_ROOT, add_third_party, read_bgr
from dvslam.odometry.base import Frame

SIZES = {'378x504': (378, 504), '476x630': (476, 630)}
OUT_HW = (120, 160)
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
_CONFIGS = {'vits': dict(features=64, out_channels=[48, 96, 192, 384])}


def preprocess(bgr: np.ndarray, net_hw: tuple[int, int]) -> np.ndarray:
    """uint8 BGR -> float32 (1, 3, H, W) ImageNet-normalised RGB."""
    h, w = net_hw
    rgb = cv2.cvtColor(cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    return ((rgb.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)[None].copy()


def load_dav2_metric(ckpt: str | Path | None = None, encoder: str = 'vits', max_depth: float = 20.0) -> nn.Module:
    """Hypersim metric weights (ckpt=None) or a Phase-3 checkpoint (raw state dict or {'model': state dict})."""
    add_third_party('depth_anything_v2')
    from dav2_metric.dpt import DepthAnythingV2
    net = DepthAnythingV2(encoder=encoder, max_depth=max_depth, **_CONFIGS[encoder])
    path = ckpt or WEIGHTS_ROOT / 'dav2' / f'depth_anything_v2_metric_hypersim_{encoder}.pth'
    sd = torch.load(path, map_location='cpu', weights_only=True)
    net.load_state_dict(sd['model'] if 'model' in sd else sd, strict=True)
    return net.eval()


class DepthExportNet(nn.Module):
    """DepthAnythingV2 forward for one fixed input size, with the export changes listed in the module docstring."""

    def __init__(self, net: nn.Module, net_hw: tuple[int, int], out_hw: tuple[int, int] | None = OUT_HW):
        super().__init__()
        add_third_party('depth_anything_v2')
        from dav2_metric.dinov2_layers import attention
        attention.XFORMERS_AVAILABLE = False          # classic attention (no memory_efficient_attention / SDPA)
        h, w = net_hw
        if h % 14 or w % 14:
            raise ValueError(f'input {net_hw} must be a multiple of 14')
        self.net, self.net_hw = net, net_hw
        self.out_hw = tuple(out_hw) if out_hw else (h, w)
        enc = net.pretrained
        with torch.no_grad():
            tokens = torch.zeros(1, 1 + (h // 14) * (w // 14), enc.embed_dim)
            pos = enc.interpolate_pos_encoding(tokens, h, w)   # prepare_tokens_with_masks passes (x, H, W)
        self.register_buffer('pos_embed', pos.detach().clone())
        enc.interpolate_pos_encoding = self._const_pos_embed  # instance override, original class untouched

    def _const_pos_embed(self, x, h, w):
        if (h, w) != self.net_hw or x.shape[1] != self.pos_embed.shape[1]:
            raise ValueError(f'engine is fixed to {self.net_hw}, got {(h, w)}')
        return self.pos_embed.to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net, head = self.net, self.net.depth_head
        ph, pw = x.shape[-2] // 14, x.shape[-1] // 14
        feats = net.pretrained.get_intermediate_layers(x, net.intermediate_layer_idx[net.encoder],
                                                       return_class_token=True)
        # DPTHead.forward (third_party/depth_anything_v2/dav2_metric/dpt.py:116-149) with the last resize target
        # replaced by self.out_hw.
        out = []
        for i, f in enumerate(feats):
            if head.use_clstoken:
                f, cls = f[0], f[1]
                f = head.readout_projects[i](torch.cat((f, cls.unsqueeze(1).expand_as(f)), -1))
            else:
                f = f[0]
            f = f.permute(0, 2, 1).reshape((f.shape[0], f.shape[-1], ph, pw))
            out.append(head.resize_layers[i](head.projects[i](f)))
        s = head.scratch
        l1, l2, l3, l4 = s.layer1_rn(out[0]), s.layer2_rn(out[1]), s.layer3_rn(out[2]), s.layer4_rn(out[3])
        p4 = s.refinenet4(l4, size=l3.shape[2:])
        p3 = s.refinenet3(p4, l3, size=l2.shape[2:])
        p2 = s.refinenet2(p3, l2, size=l1.shape[2:])
        p1 = s.refinenet1(p2, l1)
        y = F.interpolate(s.output_conv1(p1), self.out_hw, mode='bilinear', align_corners=True)
        return s.output_conv2(y)[:, 0] * net.max_depth                       # (1, OUT_H, OUT_W) metres


# ------------------------------------------------------------------------------------------------ export / build

def import_tensorrt():
    """`tensorrt` (meta package) or the `tensorrt_bindings` wheel (pip tensorrt-cu12-bindings + -libs)."""
    try:
        import tensorrt as trt
    except ImportError:
        import tensorrt_libs  # noqa: F401  (loads libnvinfer before the bindings)
        import tensorrt_bindings as trt
    return trt


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def export_onnx(onnx_path: str | Path, net_hw: tuple[int, int], ckpt: str | Path | None = None,
                out_hw: tuple[int, int] | None = OUT_HW) -> dict:
    import onnx
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    m = DepthExportNet(load_dav2_metric(ckpt), net_hw, out_hw).eval()
    x = torch.zeros(1, 3, *net_hw)
    with torch.no_grad():
        torch.onnx.export(m, (x,), str(onnx_path), dynamo=False, opset_version=17, do_constant_folding=True,
                          input_names=['image'], output_names=['depth'])
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    ops = sorted({n.op_type for n in model.graph.node})
    meta = {'onnx': str(onnx_path), 'ckpt': str(ckpt) if ckpt else 'hypersim-zero-shot', 'net_hw': list(net_hw),
            'out_hw': list(m.out_hw), 'opset': 17, 'dynamo': False, 'nodes': len(model.graph.node), 'ops': ops,
            'sdpa_free': 'Attention' not in ops and 'ScaledDotProductAttention' not in ops,
            'sha256': _sha256(onnx_path)}
    onnx_path.with_suffix('.json').write_text(json.dumps(meta, indent=1))
    return meta


# ONNX node names of the DPT head body that may run in FP16 (encoder: /patch_embed, /blocks.N, /norm*, unnamed
# constants; the output convs, final resize and sigmoid stay FP32 with the encoder).
_HEAD_LAYER = re.compile(r'^/(projects|resize_layers|layer\d_rn|refinenet\d)')


def build_engine(onnx_path: str | Path, engine_path: str | Path, precision: str = 'fp16',
                 workspace_gb: float = 4.0) -> dict:
    """precision (weakly typed builder flags, TensorRT 10.x):
    fp32 (strict, TF32 off) | fp16 | bf16 |
    fp16_head: FP32 DINOv2 encoder and output convs, FP16 DPT head body (projects/resize/refinenets). The encoder dominates the FP16 per-frame scale shift; pinning
    only LayerNorm/Softmax to FP32 had no effect (TensorRT fuses each encoder block into one Myelin node)."""
    trt = import_tensorrt()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        raise RuntimeError('ONNX parse failed: ' + '; '.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    pinned = 0
    if precision in ('fp16', 'fp16_head'):
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == 'bf16':
        config.set_flag(trt.BuilderFlag.BF16)
    elif precision == 'fp32':
        config.clear_flag(trt.BuilderFlag.TF32)       # strict FP32 (TF32 is on by default)
    else:
        raise ValueError(precision)
    if precision == 'fp16_head':
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        skip = (trt.LayerType.CONSTANT, trt.LayerType.SHAPE, trt.LayerType.SHUFFLE, trt.LayerType.GATHER,
                trt.LayerType.SLICE, trt.LayerType.CONCATENATION)
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.type in skip or any(layer.get_output(k).dtype not in (trt.float32, trt.float16)
                                         for k in range(layer.num_outputs)):
                continue
            if not _HEAD_LAYER.match(layer.name):
                layer.precision = trt.float32
                for k in range(layer.num_outputs):
                    layer.set_output_type(k, trt.float32)
                pinned += 1
    t0 = time.perf_counter()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError('TensorRT build failed')
    engine_path = Path(engine_path)
    engine_path.write_bytes(bytes(blob))
    meta = {'engine': str(engine_path), 'onnx': str(onnx_path), 'precision': precision, 'trt': trt.__version__,
            'fp32_pinned_layers': pinned, 'device': torch.cuda.get_device_name(), 'build_s': round(time.perf_counter() - t0, 1),
            'size_mb': round(engine_path.stat().st_size / 2**20, 1)}
    engine_path.with_suffix('.json').write_text(json.dumps(meta, indent=1))
    return meta


# ------------------------------------------------------------------------------------------------ runners

class _RunnerBase:
    """DepthPrior protocol: predict(frame) -> (depth_m at the frame resolution, None)."""
    net_hw: tuple[int, int]
    factor: float = 1.0
    full_res: bool = True

    def infer(self, x: np.ndarray) -> np.ndarray:            # (1,3,H,W) float32 -> (OUT_H, OUT_W) float32
        raise NotImplementedError

    def predict_image(self, bgr: np.ndarray) -> tuple[np.ndarray, None]:
        d = self.infer(preprocess(bgr, self.net_hw)) / self.factor
        if self.full_res:
            d = cv2.resize(d, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        return d.astype(np.float32), None

    def predict(self, frame: Frame) -> tuple[np.ndarray, None]:
        return self.predict_image(read_bgr(frame.image_path))


class TorchDepthRunner(_RunnerBase):
    """PyTorch reference of the exact exported graph (FP32, or FP16 autocast)."""
    name = 'dav2s_torch'

    def __init__(self, net_hw, ckpt=None, out_hw=OUT_HW, fp16=False, factor=1.0, full_res=True, device='cuda'):
        if not fp16:   # strict FP32 reference: no TF32 in cuDNN convs or matmuls (torch enables TF32 convs by default)
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        self.model = DepthExportNet(load_dav2_metric(ckpt), net_hw, out_hw).to(device).eval()
        self.net_hw, self.fp16, self.factor, self.full_res, self.device = net_hw, fp16, factor, full_res, device

    @torch.inference_mode()
    def forward_t(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.fp16):
            return self.model(x)[0].float()

    def infer(self, x):
        return self.forward_t(torch.from_numpy(x).to(self.device)).cpu().numpy()


class TRTDepthEngine(_RunnerBase):
    """TensorRT engine runner. Device buffers are torch tensors (Python harness only; the C++ runner in
    cpp/src/trt_depth.cpp is the PyTorch-free path)."""
    name = 'dav2s_trt'

    def __init__(self, engine_path: str | Path, factor: float = 1.0, full_res: bool = True):
        trt = import_tensorrt()
        self._logger = trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(self._logger).deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f'cannot deserialize {engine_path}')
        self.context = self.engine.create_execution_context()
        self.buffers = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            self.buffers[name] = torch.empty(shape, dtype=torch.float32, device='cuda')
            self.context.set_tensor_address(name, self.buffers[name].data_ptr())
        self.net_hw = tuple(self.buffers['image'].shape[-2:])
        self.out_hw = tuple(self.buffers['depth'].shape[-2:])
        self.stream = torch.cuda.Stream()
        self.factor, self.full_res = factor, full_res

    def enqueue(self) -> None:
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError('execute_async_v3 failed')

    def forward_t(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.stream(self.stream):
            self.buffers['image'].copy_(x, non_blocking=True)
            self.enqueue()
            out = self.buffers['depth'][0].clone()
        self.stream.synchronize()
        return out

    def infer(self, x):
        return self.forward_t(torch.from_numpy(x).cuda()).cpu().numpy()


class ORTDepthRunner(_RunnerBase):
    """onnxruntime runner, FP32. provider 'cuda' (CUDA EP; the probe run fails if the onnxruntime-gpu build has no
    kernels for this GPU, e.g. 1.23 on sm_120) or 'cpu'."""
    name = 'dav2s_ort'

    def __init__(self, onnx_path: str | Path, provider: str = 'cuda', factor: float = 1.0, full_res: bool = True):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        ep = {'cuda': 'CUDAExecutionProvider', 'cpu': 'CPUExecutionProvider'}[provider]
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=[ep])
        if ep not in self.sess.get_providers():
            raise RuntimeError(f'onnxruntime {ep} unavailable')
        self.net_hw = tuple(self.sess.get_inputs()[0].shape[-2:])
        self.factor, self.full_res = factor, full_res
        self.infer(np.zeros((1, 3, *self.net_hw), np.float32))      # probe: raises if kernels are missing

    def infer(self, x):
        return self.sess.run(['depth'], {'image': x})[0][0]


# ------------------------------------------------------------------------------------------------ parity / timing

def depth_parity(pred: np.ndarray, ref: np.ndarray, gt: np.ndarray | None = None,
                 gt_range: tuple[float, float] = (0.1, 10.0)) -> dict:
    """One frame. pred, ref: (H, W) depth of the candidate and the FP32 reference at the same resolution.
    rel = |pred-ref|/ref per pixel; scale = median(pred/ref) - 1; with gt (same H, W; 0 = invalid):
    b_diff = median(pred/gt) / median(ref/gt) - 1, the change of the per-frame bias b_t (plan 5.2)."""
    finite = np.isfinite(pred)
    out = {'nan_frac': float(1 - finite.mean())}
    ok = finite & (ref > 0)
    rel = np.abs(pred[ok] - ref[ok]) / ref[ok]
    out.update(rel=rel, rel_median=float(np.median(rel)), rel_p99=float(np.quantile(rel, 0.99)),
               scale=float(np.median(pred[ok] / ref[ok]) - 1))
    if gt is not None:
        v = ok & (gt > gt_range[0]) & (gt < gt_range[1])
        out['b_diff'] = float(np.median(pred[v] / gt[v]) / np.median(ref[v] / gt[v]) - 1) if v.any() else float('nan')
    return out


def summarize_parity(frames: list[dict]) -> dict:
    rel = np.concatenate([f['rel'] for f in frames])
    scale = np.array([f['scale'] for f in frames])
    b = np.array([f.get('b_diff', np.nan) for f in frames])
    s = {'frames': len(frames), 'rel_median_pct': float(np.median(rel) * 100),
         'scale_p95_abs_pct': float(np.quantile(np.abs(scale), 0.95) * 100),
         'rel_p99_pct': float(np.quantile(rel, 0.99) * 100),
         'rel_p99_worst_frame_pct': float(max(f['rel_p99'] for f in frames) * 100),
         'scale_max_abs_pct': float(np.abs(scale).max() * 100), 'scale_mean_pct': float(scale.mean() * 100),
         'b_diff_max_abs_pct': float(np.nanmax(np.abs(b)) * 100) if np.isfinite(b).any() else float('nan'),
         'nan_frac': float(np.mean([f['nan_frac'] for f in frames]))}
    s['pass'] = bool(s['nan_frac'] == 0 and s['rel_median_pct'] < 1.0 and s['rel_p99_pct'] < 2.0
                     and s['scale_max_abs_pct'] <= 0.2
                     and (np.isnan(s['b_diff_max_abs_pct']) or s['b_diff_max_abs_pct'] <= 0.2))
    return s


def cuda_event_latency(fn, stream: torch.cuda.Stream, warmup: int = 50, iters: int = 300) -> dict:
    """GPU time of fn() enqueued on `stream`, measured with CUDA events per call (ms)."""
    for _ in range(warmup):
        with torch.cuda.stream(stream):
            fn()
    stream.synchronize()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            a.record(stream)
            fn()
            b.record(stream)
        b.synchronize()
        times.append(a.elapsed_time(b))
    t = np.array(times)
    return {'p50_ms': float(np.median(t)), 'p95_ms': float(np.quantile(t, 0.95)), 'mean_ms': float(t.mean()),
            'iters': iters, 'warmup': warmup}


def wall_latency(fn, warmup: int = 50, iters: int = 300) -> dict:
    """Host wall time of a synchronous fn() (ms), for runtimes that own their stream (onnxruntime)."""
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1e3)
    t = np.array(t)
    return {'p50_ms': float(np.median(t)), 'p95_ms': float(np.quantile(t, 0.95)), 'mean_ms': float(t.mean()),
            'iters': iters, 'warmup': warmup}
