"""Phase 3.3 / depth-only 3.4: E1 metrics (b, jitter, f2f jitter, AbsRel) for fine-tuned DAv2-S checkpoints.

Same sampling and statistics as scripts/e1_depth_bias.py (anchors t = 0, s, 2s, ... plus t+1; per-frame
r = median pred/gt at GT resolution; dvslam.depth.bias), FP32 and FP16 (autocast) on the same images.

Models: a checkpoint path (fixed 476x630 pipeline, dvslam.depth.dav2_ft), `zs476` (Hypersim weights through the same
fixed pipeline) or `zs518` (official zero-shot DAv2-S as in E1, input_size 518).
Targets: Redwood `<scene>:<split>` pairs, or `tum-holdout` (item 3.3; GT in true metres, the fine-tuned output is
converted from the canonical camera with each sequence's calibrated K).

  python scripts/eval_depth_ckpt.py --name single --model runs/phase3/single/model.pth --redwood all \
      --out runs/phase3/eval
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'scripts'))
from dvslam.data.redwood import SCENES, load_depth  # noqa: E402
from dvslam.depth import read_bgr  # noqa: E402
from dvslam.depth.bias import frame_stats  # noqa: E402
from dvslam.train.depth_data import holdout_clips, redwood_frames  # noqa: E402
from e1_depth_bias import PRECS, sample_indices, summarize_model  # noqa: E402


def build_model(spec: str):
    if spec == 'zs518':
        from dvslam.depth.dav2 import DAv2Metric
        return DAv2Metric(input_size=518)
    from dvslam.depth.dav2_ft import DAv2FT
    return DAv2FT(None if spec == 'zs476' else spec)


def frame_lists(args):
    """[(group, [(bgr_path, depth_path, depth_scale, canon_factor)], pairs)] per evaluated sequence."""
    out = []
    if args.redwood:
        tg = [(sc, sp) for sc in SCENES for sp in ('validation', 'test')] if args.redwood == ['all'] else \
            [tuple(t.split(':')) for t in args.redwood]
        for sc, sp in tg:
            fr = redwood_frames(sc, sp)
            idx, pairs = sample_indices(len(fr), args.stride)
            out.append((f'{sc}/{sp}', [(fr[i].rgb, fr[i].depth, 1000.0, 1.0) for i in idx], pairs))
    if args.tum_holdout:
        by_seq = {}
        for c in holdout_clips(T=2, stride=args.tum_stride):
            by_seq.setdefault(c[0].seq, []).append(c)
        for seq, clips in by_seq.items():
            items = [(f.rgb, f.depth, f.depth_scale, f.canon_factor) for c in clips for f in c]
            pairs = np.arange(len(items)).reshape(-1, 2)
            out.append((f'tum/{seq}', items, pairs))
    return out


def run(spec, lists):
    m = build_model(spec)
    rec = {p: {k: [] for k in ('scene', 'r', 'absrel', 'absrel_med', 'n_valid', 'fwd_ms', 'total_ms')}
           for p in PRECS}
    pairs_all, dlog, off = [], [], 0
    for group, items, pairs in lists:
        pairs_all.append(pairs + off)
        off += len(items)
        t0 = time.perf_counter()
        for rgb, dpath, scale, factor in items:
            bgr = read_bgr(Path(rgb))
            gt = load_depth(Path(dpath)) if scale == 1000.0 else \
                cv2.imread(dpath, cv2.IMREAD_UNCHANGED).astype(np.float32) / scale
            if hasattr(m, 'factor'):
                m.factor = factor
            preds = {}
            for p in PRECS:
                m.fp16 = p == 'fp16'
                t1 = time.perf_counter()
                preds[p], _ = m.predict_image(bgr)
                total = (time.perf_counter() - t1) * 1e3
                st = frame_stats(preds[p], gt)
                for k, v in (('scene', group), ('fwd_ms', m.last_forward_ms), ('total_ms', total), *st.items()):
                    rec[p][k].append(v)
            ok = (preds['fp32'] > 0) & (preds['fp16'] > 0)
            dlog.append(float(np.median(np.abs(np.log(preds['fp16'][ok] / preds['fp32'][ok])))))
        print(f'  {spec} {group}: {len(items)} frames {time.perf_counter() - t0:.0f} s', flush=True)
    del m
    torch.cuda.empty_cache()
    return {p: {k: np.array(v) for k, v in r.items()} for p, r in rec.items()}, np.concatenate(pairs_all), \
        np.array(dlog)


def subset(rec, pairs, sel):
    pos = np.flatnonzero(sel)
    remap = -np.ones(len(sel), np.int64)
    remap[pos] = np.arange(len(pos))
    return {k: v[sel] for k, v in rec.items()}, remap[pairs[np.all(sel[pairs], axis=1)]]


def summarize_all(rec, pairs):
    """Per group (scene/split), pooled per split ('pooled/<split>') and pooled over everything ('pooled')."""
    out = summarize_model(rec, pairs)            # per group + 'pooled'
    splits = sorted({g.split('/')[0] if g.startswith('tum') else g.split('/')[1] for g in rec['scene']})
    for sp in splits:
        sel = np.array([(g.split('/')[0] if g.startswith('tum') else g.split('/')[1]) == sp for g in rec['scene']])
        r, pr = subset(rec, pairs, sel)
        out[f'pooled/{sp}'] = summarize_model(r, pr)['pooled']
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', required=True)
    ap.add_argument('--model', required=True, help='checkpoint path | zs476 | zs518')
    ap.add_argument('--redwood', nargs='*', default=[], help="'all' or scene:split ...")
    ap.add_argument('--tum-holdout', action='store_true')
    ap.add_argument('--stride', type=int, default=12)
    ap.add_argument('--tum-stride', type=int, default=30)
    ap.add_argument('--out', type=Path, default=Path('runs/phase3/eval'))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lists = frame_lists(args)
    rec, pairs, dlog = run(args.model, lists)
    for p in PRECS:
        np.savez(args.out / f'frames_{args.name}_{p}.npz', pairs=pairs, fp16_dlog=dlog, **rec[p])
    summary = {p: summarize_all(rec[p], pairs) for p in PRECS}
    summary['fp16_vs_fp32_median_abs_dlog'] = float(np.median(dlog))
    summary['meta'] = dict(model=args.model, redwood=args.redwood, tum_holdout=args.tum_holdout, stride=args.stride,
                           tum_stride=args.tum_stride, gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
    (args.out / f'{args.name}.json').write_text(json.dumps(summary, indent=1, default=float))
    for p in PRECS:
        for g, s in summary[p].items():
            if g.startswith('pooled'):
                print(f"{args.name} {p} {g}: b {100 * s['b_median']:+.2f} % jitter {100 * s['jitter']:.2f} % "
                      f"f2f {100 * s['f2f_jitter']:.2f} % AbsRel {100 * s['absrel']:.1f} % (n={s['n_frames']})")


if __name__ == '__main__':
    main()
