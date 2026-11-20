"""Plan v3.4 S3: one-pass test-sequence calibration of the SSL depth net with a LoRA adapter (dvslam/ssl/adapter.py).

  train  python apps/calib_s3.py train --ckpt runs/ssl/<run>/last.pt --scene room5 --labels <run>/room5.inc.npz \
             --out runs/v34/s3/<variant>/room5 [--rank 8 --alpha 8 --steps 2000 --batch 8 --lr 2e-4 --offsets -1 1
             --w-geo 0 --w-smooth 1e-3 --seed 0]
         -> <out>/adapter.pt (low-rank weights + settings + provenance), train_log.jsonl
  cache  python apps/calib_s3.py cache --ckpt ... --adapter <out>/adapter.pt --scene room5 --out <depth dir>
         -> tracker depth cache <depth dir>/<scene>.npy / .paths.txt / .meta.json (klt_depth_cache.write_cache),
            one image per forward pass like scripts/exp003_cache_depth.py; without --adapter: the frozen net.
  score  python apps/calib_s3.py score --cache before=<dir> after=<dir> --scenes room4 room5 [--json f]
         -> depth vs the stereo hints (scripts/s5_teacher_depth_audit.py metrics: tiled 8-frame windows, gaps 1 / 5,
            frame / window / no alignment, within-window scale sd). Evaluation only (the hints use cam1).

Inputs of calibration (cam0 only, plan §14.6 Q2): the tracker sequence's rectified cam0 frames (dvslam.data.tumvi
load_sequence: the frame range the tracker runs on), the cam0 vignette and exposure times (photometric loss images as
in training, dvslam.ssl.tumvi_data.photometric), and the FIXED pseudo-label increments of ONE tracker run with the
frozen depth (<labels> room<N>.inc.npz, start frame 0, all pairs 1..N-1; rel_increment = T_{t-1<-t}, metres). No
cam1, IMU or mocap. The frozen net is the checkpoint's EMA weights (as the tracker's depth caches).
Targets are drawn uniformly (seeded) from frames with at least one usable source; sources are t + offsets, posed by
composing the pseudo-label increments; a pair the tracker did not track ('motion_model') makes its source unusable.
"""
import os

for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_var, '4')

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dvslam.data import tumvi
from dvslam.odometry.base import load_increments
from dvslam.odometry.klt_depth_cache import write_cache
from dvslam.ssl import adapter as ada
from dvslam.ssl.predict import Predictor
from dvslam.ssl.tumvi_data import rectified_vignette


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(ckpt: Path, rank: int, alpha: float, prefixes: tuple[str, ...],
          device: str) -> tuple[ada.AdaptedDepth, np.ndarray]:
    pred = Predictor.from_checkpoint(ckpt, device)          # EMA weights, exactly the tracker caches' depth net
    return ada.AdaptedDepth(pred.model, rank, alpha, prefixes).to(device), pred.K


def net_image(gray: torch.Tensor) -> torch.Tensor:
    """uint8 (B, H, W) -> (B, 3, H, W) [0, 1], as dvslam.ssl.predict.Predictor."""
    return (gray.float() / 255)[:, None].expand(-1, 3, -1, -1)


def load_frames(scene: str, device: str) -> dict:
    seq = tumvi.load_sequence(scene)
    d = tumvi.sequence_dir(scene)
    _, intrinsics, dist = tumvi.load_cam_calib(d, 'cam0')
    t_exp = np.loadtxt(d / 'dso/cam0/times.txt', comments='#', usecols=0, dtype=np.int64)
    exposure = np.loadtxt(d / 'dso/cam0/times.txt', comments='#', usecols=2, dtype=np.float64)
    j = np.clip(np.searchsorted(t_exp, seq.timestamps_ns), 0, len(t_exp) - 1)
    if not np.array_equal(t_exp[j], seq.timestamps_ns):
        raise ValueError(f'{scene}: dso/cam0/times.txt lacks exposure times for some frames')
    gray = np.stack([cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in seq.image_paths])
    return {'seq': seq, 'gray': torch.from_numpy(gray).to(device),
            'V': torch.from_numpy(rectified_vignette(d, 'cam0', intrinsics, dist)).to(device),
            'exposure': torch.from_numpy(exposure[j]).to(device)}


def load_labels(path: Path, scene: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(inc (n, 4, 4) with inc[k] = T_{k-1<-k}, tracked (n,) bool) from a start-0 tracker run of `scene`."""
    z = load_increments(path)
    if z['meta']['scene'] != scene:
        raise ValueError(f'{path}: scene {z["meta"]["scene"]} != {scene}')
    if not np.array_equal(z['index'], np.arange(1, n)):
        raise ValueError(f'{path}: increments must cover pairs 1..{n - 1} (a start-0 run of the whole sequence)')
    inc = np.tile(np.eye(4), (n, 1, 1))
    inc[1:] = z['rel_increment']
    ok = np.r_[True, z['source'] == 'tracked']
    return inc, ok


def train(args) -> int:
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    dev = args.device
    net, K = build(args.ckpt, args.rank, args.alpha, tuple(args.prefixes), dev)
    n_ada, n_base = ada.count_params(net)
    n_depth = sum(p.numel() for p in net.model.depth_net.parameters()) - n_ada
    fr = load_frames(args.scene, dev)
    n = len(fr['gray'])
    inc, ok = load_labels(args.labels, args.scene, n)
    C, bad = ada.pose_chain(torch.from_numpy(inc), torch.from_numpy(ok))
    offsets = tuple(args.offsets)
    _, use_all = ada.source_poses(C, bad, torch.arange(n), offsets)
    targets = torch.nonzero(use_all.any(1)).flatten()
    Kt = torch.as_tensor(K, dtype=torch.float32, device=dev)[None].expand(args.batch, 3, 3)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    warm = min(100, args.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(args.steps - warm, 1))))
    args.out.mkdir(parents=True, exist_ok=True)
    print(f'{args.scene}: {n} frames, {len(targets)} targets, adapter {n_ada} params = '
          f'{100 * n_ada / n_depth:.2f} % of the frozen depth net ({n_depth}; whole checkpoint {n_base}), '
          f'{net.n_wrapped} convs', flush=True)
    t0 = time.perf_counter()
    with open(args.out / 'train_log.jsonl', 'w', buffering=1) as log:
        for step in range(args.steps):
            t = targets[torch.randint(len(targets), (args.batch,), generator=gen)]
            T_src, use = ada.source_poses(C, bad, t, offsets)
            s_idx = (t[:, None] + torch.tensor(offsets)).clamp(0, n - 1)
            t_d, s_d = t.to(dev), s_idx.to(dev)
            e_t = fr['exposure'][t_d]
            pho_t, valid_t = ada.loss_images(fr['gray'][t_d], fr['V'], e_t, e_t)
            src = [ada.loss_images(fr['gray'][s_d[:, i]], fr['V'], fr['exposure'][s_d[:, i]], e_t)
                   for i in range(len(offsets))]
            src_img = [net_image(fr['gray'][s_d[:, i]]) for i in range(len(offsets))] if args.w_geo > 0 else []
            total, terms = ada.calib_loss(net, net_image(fr['gray'][t_d]), pho_t, valid_t, [x[0] for x in src],
                                          [x[1] for x in src], src_img, T_src.to(dev), use.to(dev), Kt,
                                          args.w_smooth, args.w_geo)
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            if step % 25 == 0 or step == args.steps - 1:
                rec = {'step': step, 'lr': sched.get_last_lr()[0], 'loss': float(total.detach()),
                       **{k: float(v.detach()) for k, v in terms.items()}, 's': time.perf_counter() - t0}
                if dev.startswith('cuda'):
                    rec['peak_gb'] = torch.cuda.max_memory_allocated() / 2 ** 30
                log.write(json.dumps(rec) + '\n')
                if step % 250 == 0 or step == args.steps - 1:
                    print(json.dumps(rec), flush=True)
    torch.save({'adapter': ada.adapter_state(net), 'rank': args.rank, 'alpha': args.alpha,
                'prefixes': list(args.prefixes), 'ckpt': str(args.ckpt), 'ckpt_sha256': sha256(args.ckpt),
                'labels': str(args.labels), 'labels_sha256': sha256(args.labels), 'scene': args.scene,
                'n_params': n_ada, 'n_base': n_base, 'n_depth_net': n_depth,
                'args': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != 'func'}},
               args.out / 'adapter.pt')
    print(f'saved {args.out / "adapter.pt"} after {time.perf_counter() - t0:.0f} s', flush=True)
    return 0


def cache(args) -> int:
    dev = args.device
    if args.adapter is None:
        net, _ = build(args.ckpt, 1, 1.0, ('depth_net.decoder',), dev)      # zero-initialised = the frozen net
        provenance = args.ckpt
    else:
        a = torch.load(args.adapter, map_location='cpu', weights_only=True)
        if a['ckpt_sha256'] != sha256(args.ckpt):
            raise ValueError(f'{args.adapter} was trained on another checkpoint ({a["ckpt"]})')
        net, _ = build(args.ckpt, a['rank'], a['alpha'], tuple(a['prefixes']), dev)
        missing = set(a['adapter']) ^ set(ada.adapter_state(net))
        if missing:
            raise ValueError(f'adapter keys differ: {sorted(missing)[:4]}')
        net.load_state_dict(a['adapter'], strict=False)
        provenance = args.adapter
    seq = tumvi.load_sequence(args.scene)
    d = np.empty((len(seq), seq.height, seq.width), np.float32)
    with torch.no_grad():
        for i, p in enumerate(seq.image_paths):
            g = torch.from_numpy(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)).to(dev)[None]
            d[i] = net.depth(net_image(g))[0, 0].float().cpu().numpy()
    write_cache(args.out, args.scene, [str(p) for p in seq.image_paths], d, model='SSLDepth+LoRA', ckpt=provenance)
    print(f'{args.scene}: {len(seq)} maps, median depth {float(np.median(d[::50])):.3f} m -> {args.out}', flush=True)
    return 0


def score(args) -> int:
    sys.path.insert(0, str(REPO_ROOT / 'scripts'))
    import s5_teacher_depth_audit as s5
    caches = dict(c.split('=', 1) for c in args.cache)
    rows = {}
    for scene in args.scenes:
        paths, off = s5.scene_index(scene)
        hints = s5._hints(scene)
        maps = {name: s5._cache(d, scene, paths) for name, d in caches.items()}
        for gap in args.gaps:
            win = s5.tiled_windows(len(paths), gap)
            refs = [s5._ref(hints, w, off) for w in win]
            for name, m in maps.items():
                preds = [np.stack([np.asarray(m[i][s5.SUB], np.float32) for i in w]) for w in win]
                rows[(name, scene, gap)] = s5.summarise([s5.evaluate_window(r, p) for r, p in zip(refs, preds)])
    print(f'{"cache":<14} {"scene":<6} {"g":>2} {"frames":>6} | {"frame AbsRel":>12} {"window AbsRel":>13} '
          f'{"logRMSE(w)":>10} | {"scale sd med/mean %":>19} | {"none AbsRel":>11} {"bias %":>7}')
    for (name, scene, gap), v in rows.items():
        print(f'{name:<14} {scene:<6} {gap:>2} {v["frames_ok"]:>6} | {100 * v["frame_absrel"]:12.3f} '
              f'{100 * v["window_absrel"]:13.3f} {100 * v["window_logrmse"]:10.3f} | {100 * v["scale_sd_median"]:9.3f} '
              f'{100 * v["scale_sd_mean"]:9.3f} | {100 * v["none_absrel"]:11.3f} {100 * v["none_bias"]:+7.3f}')
    if args.json:
        args.json.write_text(json.dumps([dict(cache=k[0], scene=k[1], gap=k[2], **v) for k, v in rows.items()],
                                        indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('--labels', type=Path, required=True, help='start-0 tracker run room<N>.inc.npz of --scene')
    t.add_argument('--rank', type=int, default=8)
    t.add_argument('--alpha', type=float, default=8.0)
    t.add_argument('--prefixes', nargs='+', default=['depth_net.decoder'])
    t.add_argument('--steps', type=int, default=2000)
    t.add_argument('--batch', type=int, default=8)
    t.add_argument('--lr', type=float, default=2e-4)
    t.add_argument('--offsets', nargs='+', type=int, default=[-1, 1])
    t.add_argument('--w-smooth', type=float, default=1e-3)
    t.add_argument('--w-geo', type=float, default=0.0)
    t.add_argument('--seed', type=int, default=0)
    c = sub.add_parser('cache')
    c.add_argument('--adapter', type=Path, default=None)
    for p in (t, c):
        p.add_argument('--ckpt', type=Path, required=True)
        p.add_argument('--scene', required=True)
        p.add_argument('--out', type=Path, required=True)
        p.add_argument('--device', default='cuda')
    s = sub.add_parser('score')
    s.add_argument('--cache', nargs='+', required=True, metavar='NAME=DIR')
    s.add_argument('--scenes', nargs='+', default=['room4', 'room5'])
    s.add_argument('--gaps', nargs='+', type=int, default=[1, 5])
    s.add_argument('--json', type=Path, default=None)
    args = ap.parse_args()
    return {'train': train, 'cache': cache, 'score': score}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
