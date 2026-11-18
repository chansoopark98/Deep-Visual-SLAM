"""Phase 3.2: fine-tune DAv2-S metric at the fixed deployment K / resolution (plan v3 §4.5).

Loss on log depth, residual R = log d_pred - log d_gt over valid GT pixels (0.1-10 m, canonical metres):
  SILog (per image, lambda 0.15: 85 % of the mean-residual term is kept, so global scale is penalised)
  + w_grad * multi-scale gradient matching of R (4 scales, MiDaS-style)
  + w_tgm  * temporal gradient matching (Video-Depth-Anything TGM idea, reimplemented on log depth): for consecutive
    frames, |(log p_{t+1} - log p_t) - (log g_{t+1} - log g_t)| on pixels valid in both frames whose GT changes by
    less than `tgm_th` in log (static, non-occluding pixels), trimmed mean (VDA loss.py:198-256 uses 5 % of the
    depth range on affine-aligned output; here metric log depth, so a log threshold).

  python scripts/train_depth.py --fold single --steps 6000 --out runs/phase3/single
  python scripts/train_depth.py --fold loso-apartment --steps 4000 --out runs/phase3/loso-apartment
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dvslam.depth.dav2_ft import load_dav2_metric  # noqa: E402
from dvslam.train.depth_data import D_MAX, D_MIN, FOLDS, ClipDataset, clip_paths, fold_lists  # noqa: E402


def silog(R, m, lam=0.15):
    n = m.flatten(1).sum(1).clamp(min=1)
    mu = (R * m).flatten(1).sum(1) / n
    msq = (R * R * m).flatten(1).sum(1) / n
    return torch.sqrt((msq - lam * mu * mu).clamp(min=1e-8)).mean()


def grad_loss(R, m, scales=4):
    tot = 0.0
    for s in range(scales):
        r, v = R[:, ::2 ** s, ::2 ** s], m[:, ::2 ** s, ::2 ** s]
        vx, vy = v[:, :, 1:] * v[:, :, :-1], v[:, 1:] * v[:, :-1]
        gx, gy = (r[:, :, 1:] - r[:, :, :-1]).abs() * vx, (r[:, 1:] - r[:, :-1]).abs() * vy
        tot = tot + (gx.sum() + gy.sum()) / (vx.sum() + vy.sum()).clamp(min=1)
    return tot / scales


def tgm_loss(logp, logg, m, th=0.05, trim=0.2):
    """logp, logg, m: (B, T, H, W). Temporal-gradient matching on log depth with a trimmed mean."""
    dp, dg = logp[:, 1:] - logp[:, :-1], logg[:, 1:] - logg[:, :-1]
    tm = m[:, 1:] * m[:, :-1] * (dg.abs() < th)
    err = (dp - dg).abs()[tm > 0]
    if err.numel() == 0:
        return logp.sum() * 0
    if trim > 0:
        k = int(err.numel() * (1 - trim))
        err = torch.topk(err, k, largest=False, sorted=False).values if k > 0 else err
    return err.mean()


def use_sdpa():
    """Training-only memory fix: the DINOv2 copy falls back to explicit softmax(QK^T) without xFormers. Swap in
    torch SDPA (same maths, default scale head_dim^-0.5) so 1,530-token batches fit on the shared GPU."""
    import torch.nn.functional as F
    from dav2_metric.dinov2_layers import attention as A

    def forward(self, x, attn_bias=None):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))
    A.Attention.forward = forward
    A.MemEffAttention.forward = forward


def lr_at(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fold', required=True, choices=FOLDS)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=4, help='clips per step (images = batch * T)')
    ap.add_argument('--T', type=int, default=2)
    ap.add_argument('--lr', type=float, default=5e-6, help='encoder lr; decoder uses 10x (DAv2 metric recipe)')
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--w-grad', type=float, default=0.5)
    ap.add_argument('--w-tgm', type=float, default=2.0)
    ap.add_argument('--tgm-th', type=float, default=0.05)
    ap.add_argument('--redwood-frac', type=float, default=0.5, help='sampling share of Redwood clips (LOSO folds)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    lists = fold_lists(args.fold, T=args.T)
    ext, red = lists['external'], lists['redwood']
    clips = ext + red
    held = set(lists['heldout_scenes'])
    leaked = [p for p in clip_paths(red) if any(f'/{s}/' in p for s in held)]
    assert not leaked, f'leakage: {leaked[:3]}'
    if args.fold == 'single':
        assert not red
    w = np.concatenate([np.full(len(ext), (1 - args.redwood_frac) / len(ext) if red else 1 / len(ext)),
                        np.full(len(red), args.redwood_frac / max(1, len(red)))])
    sampler = WeightedRandomSampler(torch.from_numpy(w), args.steps * args.batch, replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    dl = DataLoader(ClipDataset(clips, augment=True), batch_size=args.batch, sampler=sampler,
                    num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=True,
                    prefetch_factor=4)
    print(f'[{args.fold}] external clips {len(ext)}, redwood clips {len(red)}, held-out {sorted(held)}', flush=True)

    net = load_dav2_metric().cuda().train()
    use_sdpa()
    enc = [p for n, p in net.named_parameters() if n.startswith('pretrained.')]
    dec = [p for n, p in net.named_parameters() if not n.startswith('pretrained.')]
    opt = torch.optim.AdamW([dict(params=enc, lr=args.lr), dict(params=dec, lr=10 * args.lr)], weight_decay=0.01)
    base = [g['lr'] for g in opt.param_groups]

    log, t0, step = [], time.time(), 0
    for img, dep in dl:
        B, T = img.shape[:2]
        img = img.cuda(non_blocking=True).flatten(0, 1)
        dep = dep.cuda(non_blocking=True)
        for g, b in zip(opt.param_groups, base):
            g['lr'] = b * lr_at(step, args.steps, args.warmup)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = net(img)
        pred = pred.float().view(B, T, *pred.shape[-2:])
        m = ((dep >= D_MIN) & (dep <= D_MAX)).float()
        logp = torch.log(pred.clamp(min=1e-3))
        logg = torch.log(dep.clamp(min=1e-3))
        R = (logp - logg) * m
        l_si = silog(R.flatten(0, 1), m.flatten(0, 1))
        l_gr = grad_loss(R.flatten(0, 1), m.flatten(0, 1))
        l_tg = tgm_loss(logp, logg, m, th=args.tgm_th) if T > 1 else torch.zeros((), device='cuda')
        loss = l_si + args.w_grad * l_gr + args.w_tgm * l_tg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            n = m.flatten(2).sum(-1).clamp(min=1)
            bias = ((R.flatten(2).sum(-1) / n).exp() - 1).mean().item()
        rec = dict(step=step, loss=loss.item(), silog=l_si.item(), grad=l_gr.item(), tgm=l_tg.item(),
                   bias_mean=bias, t=time.time() - t0)
        log.append(rec)
        if step % 50 == 0 or step == args.steps - 1:
            print(' '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in rec.items()),
                  flush=True)
        step += 1
        if step >= args.steps:
            break

    ck = args.out / 'model.pth'
    torch.save({'model': net.state_dict()}, ck)
    meta = dict(args={k: str(v) for k, v in vars(args).items()}, n_external_clips=len(ext), n_redwood_clips=len(red),
                heldout_scenes=sorted(held), train_seqs=sorted({f.seq for c in clips for f in c}),
                ckpt=str(ck), sha256=sha256(ck), gpu_seconds=time.time() - t0, torch=torch.__version__)
    (args.out / 'train.json').write_text(json.dumps(meta, indent=1))
    np.save(args.out / 'log.npy', np.array([[r[k] for k in ('step', 'loss', 'silog', 'grad', 'tgm', 'bias_mean')]
                                            for r in log]))
    print(json.dumps({k: meta[k] for k in ('ckpt', 'sha256', 'gpu_seconds')}), flush=True)


if __name__ == '__main__':
    main()
