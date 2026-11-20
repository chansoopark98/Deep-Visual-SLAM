"""Plan v3.4 S3: low-rank (LoRA) adapter on a frozen SSL depth net and its one-pass test-sequence calibration loss.

Adapter: every selected nn.Conv2d of the frozen depth net gets a parallel low-rank branch
    y = conv(x) + (alpha / r) * up(down(x)),   down: Conv2d(c_in, r, same kernel / stride / padding / dilation),
                                               up: Conv2d(r, c_out, 1), zero-initialised,
so a fresh adapter reproduces the base output bit for bit. Default targets are the decoder convs (`depth_net.decoder`:
upconvs and dispconvs). The base stays frozen (requires_grad False) and in eval mode, so its BatchNorm statistics never
change; only the low-rank weights train. merge_lora folds the trained branches into the base convs (one plain conv
each, same output up to float rounding) for deployment.

Frozen scale anchor (per frame): with d_a the adapted and d_f the frozen depth of the same frame,
    d = d_a * exp(-c),   c = median over all pixels of log(d_a / d_f),
so the anchored depth has exactly the frozen net's per-frame median log-depth ratio 0: the adapter may change the depth
SHAPE (and how it changes from frame to frame) but not the per-frame scale, which stays the frozen net's. The same
function runs in calibration and in inference (re-caching). scale_anchor_loss = mean c^2 is the soft form (0 for a pure
shape change); it is reported as a diagnostic.

Calibration loss (calib_loss) on target frame t with FIXED pseudo-label poses T_{src<-t} (from the tracker run with the
frozen depth), cam0 only: per scale (0..3, depth upsampled to full resolution as in training), the Monodepth2 min over
the sources of 0.85 SSIM + 0.15 L1 with auto-mask on the photometric loss images (dvslam.ssl.losses.temporal_term,
same masking as training), edge-aware smoothness / 2^scale, and optionally the SC-Depth geometric consistency
(scale 0) against the anchored, detached adapted depth of each source. Nothing else (no cam1, IMU, mocap) is used.
"""
import math

import torch
from torch import nn

from dvslam.ssl import geometry, losses
from dvslam.ssl.models import NUM_SCALES, SSLModel


class LoRAConv2d(nn.Module):
    """Frozen conv + zero-initialised low-rank residual branch (see module docstring)."""

    def __init__(self, base: nn.Conv2d, rank: int, alpha: float):
        super().__init__()
        if base.groups != 1:
            raise ValueError('grouped convolutions are not supported')
        self.base = base
        self.down = nn.Conv2d(base.in_channels, rank, base.kernel_size, base.stride, base.padding, base.dilation,
                              bias=False, padding_mode=base.padding_mode)
        self.up = nn.Conv2d(rank, base.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        return y + self.scale * self.up(self.down(x)) if self.enabled else y


def add_lora(module: nn.Module, rank: int, alpha: float, prefixes: tuple[str, ...] = ('depth_net.decoder',)) -> int:
    """Wrap every nn.Conv2d whose qualified name starts with one of `prefixes` in a LoRAConv2d (in place), freeze all
    other parameters. Returns the number of wrapped convs. A conv shared under several names gets one wrapper, set
    under every name. ValueError (module untouched) if nothing matches."""
    targets = [(name, m) for name, m in module.named_modules(remove_duplicate=False)
               if isinstance(m, nn.Conv2d) and not isinstance(m, LoRAConv2d) and name.startswith(prefixes)]
    if not targets:
        raise ValueError(f'no nn.Conv2d under {prefixes}')
    module.requires_grad_(False)
    wrap = {}
    for name, conv in targets:
        if id(conv) not in wrap:
            wrap[id(conv)] = LoRAConv2d(conv, rank, alpha)
        _replace(module, name, wrap[id(conv)])
    for m in module.modules():
        if isinstance(m, LoRAConv2d):
            m.base.requires_grad_(False)
            m.down.requires_grad_(True)
            m.up.requires_grad_(True)
    return len(wrap)


def _replace(module: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, attr = name.rpartition('.')
    setattr(module.get_submodule(parent_name) if parent_name else module, attr, new)


@torch.no_grad()
def merge_lora(module: nn.Module) -> int:
    """Fold every LoRA branch into its base conv in place and put the base conv back under every name of its wrapper:
    W' = W + scale · up ∘ down, i.e. W'[o] = W[o] + scale · Σ_r up[o, r] down[r] (up is 1 × 1 and down has the base
    conv's kernel / stride / padding / dilation, so the sum of the two convs is one conv). Disabled wrappers are folded
    too. Returns the number of merged convs; the module then has no adapter (set_adapters / frozen_depth are no-ops)."""
    names = [(name, m) for name, m in module.named_modules(remove_duplicate=False) if isinstance(m, LoRAConv2d)]
    done = {}
    for name, m in names:
        if id(m) not in done:
            dw = torch.einsum('or,rikl->oikl', m.up.weight[:, :, 0, 0], m.down.weight)
            m.base.weight += m.scale * dw.to(m.base.weight.dtype)
            done[id(m)] = m.base
        _replace(module, name, done[id(m)])
    return len(done)


def adapter_state(module: nn.Module) -> dict[str, torch.Tensor]:
    """The trainable (low-rank) tensors only, keyed as in module.state_dict(): <wrapper name>.down / .up.weight
    (a wrapper shared under several names is stored under its first)."""
    return {f'{name}.{part}.weight': getattr(m, part).weight.detach().cpu().clone()
            for name, m in module.named_modules() if isinstance(m, LoRAConv2d) for part in ('down', 'up')}


def set_adapters(module: nn.Module, enabled: bool) -> None:
    for m in module.modules():
        if isinstance(m, LoRAConv2d):
            m.enabled = enabled


def count_params(module: nn.Module) -> tuple[int, int]:
    """(adapter parameters, base parameters); base = everything that is not a low-rank branch."""
    ada = sum(p.numel() for m in module.modules() if isinstance(m, LoRAConv2d)
              for p in (*m.down.parameters(), *m.up.parameters()))
    return ada, sum(p.numel() for p in module.parameters()) - ada


# ---------------------------------------------------------------- frozen scale anchor
def median_log_ratio(d_a: torch.Tensor, d_f: torch.Tensor) -> torch.Tensor:
    """(B,) per-frame median over all pixels of log(d_a / d_f); d_* (B, 1, H, W) metres."""
    return (torch.log(d_a) - torch.log(d_f)).flatten(1).median(1).values


def anchor_scale(d_a: torch.Tensor, d_f: torch.Tensor) -> torch.Tensor:
    """exp(-c) (B, 1, 1, 1): the per-frame factor that gives d_a the frozen net's median log-depth. Not detached, so
    the anchored depth d_a * exp(-c) is exactly invariant to a global rescaling of d_a and the loss has no gradient
    along the scale direction (a detached factor lets the raw output scale drift without bound)."""
    return torch.exp(-median_log_ratio(d_a, d_f.detach()))[:, None, None, None]


def scale_anchor_loss(d_a: torch.Tensor, d_f: torch.Tensor) -> torch.Tensor:
    """Mean over frames of c^2, c = median log(d_a / d_f): 0 for a pure shape change, > 0 for a scale change."""
    return (median_log_ratio(d_a, d_f) ** 2).mean()


class AdaptedDepth(nn.Module):
    """Frozen SSLModel depth net + LoRA adapter + frozen scale anchor. Always in eval mode (BatchNorm frozen)."""

    def __init__(self, model: SSLModel, rank: int, alpha: float, prefixes: tuple[str, ...] = ('depth_net.decoder',)):
        super().__init__()
        self.model = model
        self.n_wrapped = add_lora(model, rank, alpha, prefixes)
        self.model.eval()

    def train(self, mode: bool = True):
        return super().train(False)

    @torch.no_grad()
    def frozen_depth(self, img: torch.Tensor) -> torch.Tensor:
        set_adapters(self.model, False)
        try:
            return self.model.depth_m(img)
        finally:
            set_adapters(self.model, True)

    def forward(self, img: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        """(disparities per scale, anchored metric depths per scale, frozen depth (scale 0)); img (B, 3, H, W)."""
        d_f = self.frozen_depth(img)
        disps = self.model.depth(img)[0]
        depths = [self.model.disp_to_depth(d) for d in disps]
        k = anchor_scale(depths[0], d_f)
        return disps, [d * k for d in depths], d_f

    @torch.no_grad()
    def depth(self, img: torch.Tensor) -> torch.Tensor:
        """Anchored full-resolution metric depth (B, 1, H, W)."""
        return self(img)[1][0]


# ---------------------------------------------------------------- calibration loss
def pose_chain(inc: torch.Tensor, ok: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(C (N, 4, 4) = T_{0<-k}, bad (N,) = number of untracked pairs in (0, k]) of the pseudo-label increments.

    inc (N, 4, 4) float64: inc[k] = T_{k-1<-k} (the tracker's rel_increment of pair (k-1, k); inc[0] unused);
    ok (N,) bool: pair (k-1, k) was tracked (ok[0] ignored).
    """
    chain = [torch.eye(4, dtype=inc.dtype, device=inc.device)]
    for k in range(1, len(inc)):
        chain.append(chain[-1] @ inc[k])
    bad = torch.cumsum(torch.cat([ok.new_zeros(1, dtype=torch.int64), (~ok[1:]).to(torch.int64)]), 0)
    return torch.stack(chain), bad


def source_poses(C: torch.Tensor, bad: torch.Tensor, targets: torch.Tensor,
                 offsets: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed pseudo-label poses T_{t+o<-t} = C_{t+o}^-1 C_t (B, S, 4, 4) and usable (B, S) for targets t (B,).
    A pose is usable when the source is inside [0, N) and every pair between source and target was tracked."""
    n = len(C)
    out, use = [], []
    for o in offsets:
        s = targets + o
        sc = s.clamp(0, n - 1)
        lo, hi = torch.minimum(sc, targets), torch.maximum(sc, targets)
        use.append((s >= 0) & (s < n) & (bad[hi] - bad[lo] == 0))
        out.append(torch.linalg.inv(C[sc]) @ C[targets])
    return torch.stack(out, 1), torch.stack(use, 1)


def loss_images(gray: torch.Tensor, V: torch.Tensor, exposure: torch.Tensor, exposure_t: torch.Tensor,
                v_min: float = 0.25, sat_level: int = 250) -> tuple[torch.Tensor, torch.Tensor]:
    """dvslam.ssl.tumvi_data.photometric on the GPU: gray uint8 (B, H, W), vignette V (H, W), exposures (B,) of the
    frames and of their targets -> (loss image (B, 1, H, W), valid (B, 1, H, W)) float32."""
    ok_v = V >= v_min
    ratio = (exposure_t / exposure).float()[:, None, None]
    pho = torch.where(ok_v, gray.float() / 255 / torch.where(ok_v, V, 1) * ratio, 0)
    return pho[:, None], (ok_v & (gray < sat_level)).float()[:, None]


def calib_loss(net: AdaptedDepth, img_t: torch.Tensor, pho_t: torch.Tensor, valid_t: torch.Tensor,
               src_pho: list[torch.Tensor], src_valid: list[torch.Tensor], src_img: list[torch.Tensor],
               T_src: torch.Tensor, use: torch.Tensor, K: torch.Tensor, w_smooth: float = 1e-3,
               w_geo: float = 0.0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Calibration loss of a batch of targets (module docstring). img_t (B, 3, H, W) network image; pho_t / valid_t
    (B, 1, H, W); per source i: src_pho[i], src_valid[i] (B, 1, H, W) (exposure-normalised to the target), src_img[i]
    network image (only needed with w_geo > 0); T_src (B, S, 4, 4) = T_{src<-t}; use (B, S) bool."""
    disps, depths, d_f = net(img_t)
    T_src = T_src.float()
    useS = [use[:, i].float()[:, None, None, None] for i in range(len(src_pho))]
    terms = {'temporal': 0.0, 'smooth': 0.0}
    src_masks = [losses.erode(v) * u for v, u in zip(src_valid, useS)]

    def warp_masked(i: int, depth: torch.Tensor):
        out, vis, z, pix = geometry.warp(torch.cat([src_pho[i], src_valid[i]], 1), depth, T_src[:, i], K)
        return out[:, :1], losses.erode(vis * valid_t * (out[:, 1:] > 0.999).float()) * useS[i], z, pix

    for s in range(NUM_SCALES):
        warped = [warp_masked(i, depths[s]) for i in range(len(src_pho))]
        terms['temporal'] = terms['temporal'] + losses.temporal_term(
            pho_t, [x[0] for x in warped], [x[1] for x in warped], src_pho, src_masks)
        terms['smooth'] = terms['smooth'] + losses.smooth_term(disps[s], pho_t) / 2 ** s
    terms = {k: v / NUM_SCALES for k, v in terms.items()}
    if w_geo > 0:
        geo = []
        for i in range(len(src_pho)):
            with torch.no_grad():
                d_src = net.depth(src_img[i])
            _, mask, z, pix = warp_masked(i, depths[0])
            geo.append(losses.geometric_term(z, geometry.sample(d_src, pix), mask)[0])
        terms['geometric'] = sum(geo) / len(geo)
    total = terms['temporal'] + w_smooth * terms['smooth'] + w_geo * terms.get('geometric', 0.0)
    with torch.no_grad():
        terms['anchor_c2'] = scale_anchor_loss(net.model.disp_to_depth(disps[0]), d_f)   # before the anchor
        terms['shape_dlog'] = (torch.log(depths[0]) - torch.log(d_f)).abs().mean()
    return total, terms

