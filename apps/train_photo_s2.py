"""Plan v3.4 §14.4 S2: train PhotoNet (dvslam/odometry/photo_learned.py) through the unrolled photometric LM of
klt_photo_torch.refine_torch. No ground truth in any loss (mocap is used only by the separate room5 tracker scoring).

Data: per room the dump of scripts/s2_dump_pairs.py (<dump>/<room>.s2pairs.npz, .tracks.npz, .inc.npz) from a frozen
tracker run (final-candidate options, photo_backend=torch), the rectified images and the frozen depth cache (--depth,
the tracker's own prior). A sample is a window of L + 1 consecutive frames (L = largest gap) whose L pairs came from a
tracked increment and ran both refinement directions. Each pair is refined with the tracker's inputs (its points,
depths × the pair's map ratio s, exposure ratios, start increment), with the net's weights / feature, both directions.
The trainer replays only one tracker configuration: LM on pyramid level 1 only (LEVELS), no point cap (MAX_PTS),
the tracker's default depth limits (DEPTH_LIM), photo_bidir, --iters / --huber / --tol / --cell / --grad. The dump
records the run's options (opt_* keys) and they must match (dump_options; a dump without the record only warns).
The directions are combined as the tracker did in the dumped run (its ok_f / ok_b): the midpoint when both were
accepted, the accepted one otherwise; a pair with neither accepted contributes no loss term (R_k composites through
it, B and P are skipped). Checkpoints trained before this rule used the midpoint of both directions for every pair.

Losses (weights on the command line):
  R_k  gap-k reprojection (k = 1, plus --gaps): Cauchy (--c-px) error of the tracker's sparse tracks between frames
       i and i + k under the composed refined pair poses, depths = frozen depth of frame i × s. Wide gaps enter only
       here (EXP-008 §3.5: wide photometric refinement hurt).
  B    bidirectional consistency: Cauchy (--c-bi mm) of |t_f − t_b| (metric) on pairs whose tracker step (its own
       output, e^{-g} · |t|) is >= --bi-min-mm (S2a: no signal below 5 mm).
  P    photometric anchor: raw-intensity, unweighted cost of the forward pose relative to the LM's start cost
       (photo_learned.photo_cost); keeps learned weights / features on the image evidence.
  L = Σ_k λ_k R_k + λ_bi B + λ_ph P, means over the window's pairs / composites.

  python apps/train_photo_s2.py --dump runs/v34/s2/dump --depth runs/eval/exp014/depth/hints_core_last \
      --out runs/v34/s2/train/pilot --mode w --steps 2000
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dvslam.data import registry
from dvslam.odometry import photo_learned as pl
from dvslam.odometry.base import Frame
from dvslam.odometry.klt_depth_cache import CachedDepth
from dvslam.odometry.klt_photo import PhotoNorm, prepare, select_points_g
from dvslam.odometry.klt_photo_torch import prepare_torch, refine_torch
from dvslam.odometry.klt_window import sample_depth

DEPTH_LIM = (0.2, 6.0, 1.05)                      # tracker defaults depth_min / depth_max / depth_edge
LEVELS, MAX_PTS = 1, 0                            # photo_levels (PhotoNet runs on level 1 only), photo_max_pts (off)


def dump_options(a) -> dict:
    """The tracker options this trainer replays; the dump's opt_* record (scripts/s2_dump_pairs.py) must match."""
    return {'photo_levels': LEVELS, 'photo_max_pts': MAX_PTS, 'depth_min': DEPTH_LIM[0], 'depth_max': DEPTH_LIM[1],
            'depth_edge': DEPTH_LIM[2], 'photo_iters': a.iters, 'photo_huber': a.huber, 'photo_tol': a.tol,
            'photo_bidir': True, 'photo_cell': a.cell, 'photo_grad': a.grad}


def check_dump_options(d: dict, expect: dict, name: str) -> None:
    """ValueError if the dump's recorded tracker options differ from `expect`; a warning if it has no record."""
    rec = {k[len('opt_'):]: d[k].item() for k in d if k.startswith('opt_')}
    if not rec:
        warnings.warn(f'{name}: no tracker-option record (dump older than opt_*); assuming it was made with {expect}')
        return
    bad = {k: (rec.get(k, 'missing'), v) for k, v in expect.items() if k not in rec or rec[k] != v}
    if bad:
        raise ValueError(f'{name}: dump made with other tracker options than the trainer replays '
                         f'(option: (dump, trainer)): {bad}')


def correspondences(fr: np.ndarray, gid: np.ndarray, px: np.ndarray, a: int, b: int):
    """Tracks observed at both frames a and b: (p_a (n, 2), p_b (n, 2)), from tracks.npz arrays sorted by frame."""
    ia, ib = slice(*np.searchsorted(fr, [a, a + 1])), slice(*np.searchsorted(fr, [b, b + 1]))
    _, ka, kb = np.intersect1d(gid[ia], gid[ib], assume_unique=True, return_indices=True)
    return px[ia][ka], px[ib][kb]


class Windows(Dataset):
    """Windows of L + 1 frames (start every `stride` frames) whose L pairs are all usable."""

    def __init__(self, rooms, dump, depth, L, gaps, cell=1, grad=0.002, stride=1, limit=0, seed=0, expect=None):
        self.L, self.gaps, self.cell, self.grad, self.depth_root = L, gaps, cell, grad, depth
        self.rooms, self.items = {}, []
        for room in rooms:
            seq = registry.load_sequence('tumvi', room, None, registry.default_root('tumvi'), 0.0)
            d = dict(np.load(Path(dump) / f'{room}.s2pairs.npz'))
            if expect is not None:
                check_dump_options(d, expect, f'{room}.s2pairs.npz')
            inc = np.load(Path(dump) / f'{room}.inc.npz')
            tracked = dict(zip(inc['index'].tolist(), (inc['source'] == 'tracked').tolist()))
            row = {int(t): k for k, (t, p) in enumerate(zip(d['index'], d['prev']))
                   if p == t - 1 and tracked.get(int(t), False) and np.isfinite(d['T_b'][k]).all()}
            tr = np.load(Path(dump) / f'{room}.tracks.npz')
            o = np.argsort(tr['obs_frame'], kind='stable')
            self.rooms[room] = {'seq': seq, 'd': d, 'row': row, 'fr': tr['obs_frame'][o], 'gid': tr['obs_gid'][o],
                                'px': tr['obs_px'][o]}
            self.items += [(room, t0) for t0 in range(0, len(seq) - L, stride)
                           if all(t0 + i + 1 in row for i in range(L))]
        if limit and len(self.items) > limit:
            keep = np.sort(np.random.default_rng(seed).choice(len(self.items), limit, replace=False))
            self.items = [self.items[k] for k in keep]
        self.nm = self.depth = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        if self.nm is None:                            # per worker
            cv2.setNumThreads(1)
            self.nm, self.depth = PhotoNorm(), CachedDepth(self.depth_root)
        room, t0 = self.items[k]
        r = self.rooms[room]
        seq, d = r['seq'], r['d']
        fr = list(range(t0, t0 + self.L + 1))
        P, pts, D, e = [], [], [], []
        for f in fr:
            path = seq.image_paths[f]
            I, M = self.nm(path, cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), 1.0)
            St = prepare(I, M, LEVELS)
            P.append(St[0])
            pts.append(select_points_g(St[0][..., 1], St[0][..., 2], M, self.cell, self.grad, MAX_PTS)[0])
            D.append(self.depth.predict(Frame(f, float(seq.timestamps[f]), path))[0])
            e.append(self.nm.exposure(path))
        pairs = []
        for i in range(self.L):
            j = r['row'][fr[i + 1]]
            s = float(d['s'][j])
            zf = sample_depth(D[i], pts[i], *DEPTH_LIM) * s
            zb = sample_depth(D[i + 1], pts[i + 1], *DEPTH_LIM) * s
            vf, vb = np.isfinite(zf), np.isfinite(zb)
            pairs.append({'index': fr[i + 1], 's': s, 'g': float(d['g'][j]), 'inc': d['inc'][j], 'out': d['out'][j],
                          'T_f': d['T_f'][j], 'T_b': d['T_b'][j], 'ok_f': bool(d['ok_f'][j]),
                          'ok_b': bool(d['ok_b'][j]), 'ratio_f': e[i] / e[i + 1],
                          'ratio_b': e[i + 1] / e[i], 'vf': vf, 'vb': vb, 'zf': zf[vf], 'zb': zb[vb]})
        corr = []
        for gap in self.gaps:
            for i in range(self.L - gap + 1):
                pa, pb = correspondences(r['fr'], r['gid'], r['px'], fr[i], fr[i + gap])
                z = sample_depth(D[i], pa, *DEPTH_LIM) * pairs[i]['s']
                v = np.isfinite(z)
                corr.append((gap, i, pa[v], z[v], pb[v]))
        return {'room': room, 'frames': fr, 'P': P, 'pts': pts, 'pairs': pairs, 'corr': corr, 'K': seq.K}


def _first(batch):
    return batch[0]


def run_window(net, smp, a, dev, dtype):
    """Refine every pair of the window with the net; returns (loss terms dict of tensors, stats dict)."""
    K = smp['K']
    P = [torch.from_numpy(p).to(dev) for p in smp['P']]
    img = torch.stack([p[..., 0] for p in P])[:, None].to(next(net.parameters()).dtype)
    val = torch.stack([p[..., 3] > 0.5 for p in P])[:, None]
    feat, logw = net(img, val)
    S = [[p.to(dtype)] if feat is None else prepare_torch(feat[j].to(dtype), val[j, 0], LEVELS)
         for j, p in enumerate(P)]
    raw = [[p.to(dtype)] for p in P]
    pts = [torch.from_numpy(x).to(dev) for x in smp['pts']]
    T_mid, bi, ph, st = [], [], [], {'dt_f_mm': [], 'dt_b_mm': [], 'ess': [], 'n_acc': 0}
    for i, pr in enumerate(smp['pairs']):
        pf, pb = pts[i][torch.from_numpy(pr['vf']).to(dev)], pts[i + 1][torch.from_numpy(pr['vb']).to(dev)]
        zf, zb = torch.from_numpy(pr['zf']).to(dev), torch.from_numpy(pr['zb']).to(dev)
        wf = None if logw is None else pl.point_weights(logw[i, 0], pf)
        wb = None if logw is None else pl.point_weights(logw[i + 1, 0], pb)
        A0 = np.linalg.inv(pr['inc'])
        Rf, tf, inf = refine_torch(S[i], S[i + 1], pr['ratio_f'], pf, zf, K, A0[:3, :3], A0[:3, 3], LEVELS,
                                   a.iters, a.huber, a.tol, wf)
        Rb, tb, inb = refine_torch(S[i + 1], S[i], pr['ratio_b'], pb, zb, K, pr['inc'][:3, :3], pr['inc'][:3, 3],
                                   LEVELS, a.iters, a.huber, a.tol, wb)
        Tf, Tb = pl.inv_T(pl.to_T(Rf, tf)), pl.to_T(Rb, tb)
        if pr['ok_f'] and pr['ok_b']:                  # the tracker's combination of the accepted directions
            T_mid.append(pl.pose_mid(Tf, Tb))
        else:
            T_mid.append(Tf if pr['ok_f'] else Tb if pr['ok_b'] else None)
        sc = np.exp(-pr['g']) * 1e3
        st['dt_f_mm'].append(float(torch.linalg.norm(Tf[:3, 3].detach() - torch.from_numpy(pr['T_f'][:3, 3]))) * sc)
        st['dt_b_mm'].append(float(torch.linalg.norm(Tb[:3, 3].detach() - torch.from_numpy(pr['T_b'][:3, 3]))) * sc)
        st['n_acc'] += int(inf['cost'] < inf['cost0']) + int(inb['cost'] < inb['cost0'])
        if wf is not None:
            st['ess'].append(float(wf.detach().sum() ** 2 / (len(wf) * (wf.detach() ** 2).sum())))
        if T_mid[-1] is None:                          # neither direction accepted: no loss term from this pair
            continue
        step = np.linalg.norm(pr['out'][:3, 3]) * sc if np.isfinite(pr['out']).all() else 0.0
        if a.w_bi > 0 and step >= a.bi_min_mm:
            bi.append(pl.cauchy((((Tf[:3, 3] - Tb[:3, 3]) * sc) ** 2).sum(), a.c_bi))
        if a.w_ph > 0:
            c0 = pl.photo_cost(raw[i], raw[i + 1], pr['ratio_f'], pf, zf, K, torch.from_numpy(A0[:3, :3]),
                               torch.from_numpy(A0[:3, 3]), a.huber).detach()
            ph.append(pl.photo_cost(raw[i], raw[i + 1], pr['ratio_f'], pf, zf, K, Rf, tf, a.huber) / c0)
    terms = {}
    for gap, i, pa, za, pb in smp['corr']:
        if len(pa) < a.min_tracks or any(T_mid[j] is None for j in range(i, i + gap)):
            continue
        T = T_mid[i]
        for j in range(i + 1, i + gap):
            T = T @ T_mid[j]                           # X_i = T X_{i+gap}
        loss = pl.reproj_loss(pl.inv_T(T), torch.from_numpy(pa), torch.from_numpy(za), torch.from_numpy(pb), K,
                              a.c_px)
        terms.setdefault(f'R{gap}', []).append(loss)
    if bi:
        terms['B'] = bi
    if ph:
        terms['P'] = ph
    terms = {k: torch.stack(v).mean() for k, v in terms.items()}
    if logw is not None:
        st['logw_sd'] = float(logw.detach()[val].std())
    if feat is not None:
        st['dfeat_sd'] = float((feat - img).detach()[val].std())
    return terms, st


def total(terms, a):
    w = {'B': a.w_bi, 'P': a.w_ph, **{f'R{g}': (a.w_r1 if g == 1 else a.w_rk) for g in a.gaps}}
    return sum(w[k] * v for k, v in terms.items())


def evaluate(net, ds, a, dev, dtype):
    net.eval()
    acc, n = {}, 0
    with torch.no_grad():
        for smp in DataLoader(ds, batch_size=1, num_workers=a.workers, collate_fn=_first):
            terms, st = run_window(net, smp, a, dev, dtype)
            for k, v in terms.items():
                acc.setdefault(k, []).append(float(v))
            for k in ('dt_f_mm', 'dt_b_mm', 'ess'):
                acc.setdefault(k, []).extend(st[k])
            n += 1
    net.train()
    return {k: float(np.mean(v)) for k, v in acc.items() if v} | {'windows': n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', type=Path, required=True)
    ap.add_argument('--depth', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--rooms', default='room1,room2,room3,room4')
    ap.add_argument('--val-rooms', default='room5')
    ap.add_argument('--val-windows', type=int, default=200)
    ap.add_argument('--mode', default='w', choices=pl.MODES)
    ap.add_argument('--width', type=int, default=16)
    ap.add_argument('--gaps', default='1', help='comma list of reprojection gaps; 1 is always included')
    ap.add_argument('--w-r1', type=float, default=1.0)
    ap.add_argument('--w-rk', type=float, default=1.0, help='weight of each gap > 1 term')
    ap.add_argument('--w-bi', type=float, default=0.0)
    ap.add_argument('--w-ph', type=float, default=0.0)
    ap.add_argument('--c-px', type=float, default=0.5)
    ap.add_argument('--c-bi', type=float, default=0.5)
    ap.add_argument('--bi-min-mm', type=float, default=5.0)
    ap.add_argument('--min-tracks', type=int, default=10)
    ap.add_argument('--iters', type=int, default=10)
    ap.add_argument('--huber', type=float, default=0.03)
    ap.add_argument('--tol', type=float, default=0.01)
    ap.add_argument('--cell', type=int, default=1)
    ap.add_argument('--grad', type=float, default=0.002)
    ap.add_argument('--dtype', default='float32', choices=('float32', 'float64'))
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--accum', type=int, default=4)
    ap.add_argument('--steps', type=int, default=2000, help='optimizer steps')
    ap.add_argument('--eval-every', type=int, default=500)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    a.gaps = sorted({1, *map(int, a.gaps.split(','))})
    torch.manual_seed(a.seed)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    dev, dtype = torch.device('cuda'), getattr(torch, a.dtype)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'args.json').write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                                                indent=1))
    L = max(a.gaps)
    tr = Windows(a.rooms.split(','), a.dump, a.depth, L, a.gaps, a.cell, a.grad, expect=dump_options(a))
    va = Windows(a.val_rooms.split(','), a.dump, a.depth, L, a.gaps, a.cell, a.grad, stride=L, limit=a.val_windows,
                 expect=dump_options(a))
    print(f'train windows {len(tr)}, val windows {len(va)}, L {L}, gaps {a.gaps}', flush=True)
    net = pl.PhotoNet(a.mode, a.width).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.0)
    def write(rec):
        with open(a.out / 'log.jsonl', 'a') as f:
            f.write(json.dumps(rec) + '\n')
        print(json.dumps(rec), flush=True)

    write({'step': 0, 'val': evaluate(net, va, a, dev, dtype)})
    pl.save_net(a.out / 'step0.pt', net, {'step': 0})
    step, k, t0, run = 0, 0, time.time(), {}
    while step < a.steps:
        for smp in DataLoader(tr, batch_size=1, shuffle=True, num_workers=a.workers, collate_fn=_first,
                              prefetch_factor=4, persistent_workers=False):
            terms, st = run_window(net, smp, a, dev, dtype)
            if not terms:                              # no pair of the window was accepted by the tracker
                continue
            loss = total(terms, a)
            (loss / a.accum).backward()
            for kk, v in terms.items():
                run.setdefault(kk, []).append(float(v.detach()))
            run.setdefault('loss', []).append(float(loss.detach()))
            for kk in ('dt_f_mm', 'dt_b_mm', 'ess'):
                run.setdefault(kk, []).extend(st[kk])
            k += 1
            if k % a.accum:
                continue
            gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0))
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            run.setdefault('grad_norm', []).append(gn)
            if step % 50 == 0:
                rec = {'step': step, 's': round(time.time() - t0, 1),
                       'train': {kk: float(np.mean(v)) for kk, v in run.items() if v}}
                rec['train'].update({kk: st[kk] for kk in ('logw_sd', 'dfeat_sd') if kk in st})
                write(rec)
                run = {}
            if step % a.eval_every == 0 or step == a.steps:
                write({'step': step, 'val': evaluate(net, va, a, dev, dtype)})
                pl.save_net(a.out / f'step{step}.pt', net, {'step': step})
                pl.save_net(a.out / 'last.pt', net, {'step': step})
            if step >= a.steps:
                break
    (a.out / 'DONE').write_text('ok\n')


if __name__ == '__main__':
    main()
