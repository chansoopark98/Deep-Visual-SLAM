"""E1 (plan v3 item 0.10): zero-shot metric-depth bias b and jitter on Redwood `validation`, FP32 vs FP16.

Frames: anchors t = 0, s, 2s, ... and their successors t+1 (consecutive pairs for frame-to-frame jitter).
Each model is built once per precision; both precisions see the same loaded image.

  python scripts/e1_depth_bias.py                                  # all 3 models, 5 scenes, stride 6
  python scripts/e1_depth_bias.py --models dav2s --stride 50 --diag   # + DAv2 focal / resolution diagnostics

Outputs in --out: frames_<model>_<prec>.npz (per-frame r, AbsRel, timing), summary.json, tables.md.
--resume reuses finished models' npz (e.g. after a CUDA OOM on the shared GPU).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dvslam.data.redwood import SCENES, load_depth, load_sequence  # noqa: E402
from dvslam.depth import MODELS, build, read_bgr  # noqa: E402
from dvslam.depth.bias import frame_stats, summarize  # noqa: E402
from dvslam.depth.dav2 import HYPERSIM_F_OVER_W  # noqa: E402

PRECS = ('fp32', 'fp16')


def sample_indices(n: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Sorted frame indices and (P, 2) positions of consecutive pairs (t, t+1) within them."""
    anchors = np.arange(0, n - 1, stride)
    idx = np.unique(np.concatenate([anchors, anchors + 1]))
    pos = {int(f): i for i, f in enumerate(idx)}
    pairs = np.array([[pos[int(t)], pos[int(t) + 1]] for t in anchors], np.int64)
    return idx, pairs


def run_model(name, seqs, stride, kwargs):
    # One network serves both precisions (fp16 = autocast toggle) to keep GPU memory low on the shared card.
    m = build(name, seqs[0].K, fp16=False, **kwargs)
    warm = read_bgr(seqs[0].image_paths[0])
    for p in PRECS:
        m.fp16 = p == 'fp16'
        for _ in range(3):
            m.predict_image(warm)
    rec = {p: {k: [] for k in ('scene', 'frame', 'r', 'absrel', 'absrel_med', 'n_valid', 'fwd_ms', 'total_ms')}
           for p in PRECS}
    pairs_all, fp16_dlog, offset = [], [], 0
    for seq in seqs:
        idx, pairs = sample_indices(len(seq), stride)
        pairs_all.append(pairs + offset)
        offset += len(idx)
        t_scene = time.perf_counter()
        for i in idx:
            bgr, gt = read_bgr(seq.image_paths[i]), load_depth(seq.depth_paths[i])
            preds = {}
            for p in PRECS:
                m.fp16 = p == 'fp16'
                t0 = time.perf_counter()
                preds[p], _ = m.predict_image(bgr)
                total = (time.perf_counter() - t0) * 1e3
                st = frame_stats(preds[p], gt)
                for k, v in (('scene', seq.scene), ('frame', int(seq.frame_ids[i])), ('fwd_ms', m.last_forward_ms),
                             ('total_ms', total), *st.items()):
                    rec[p][k].append(v)
            ok = np.isfinite(preds['fp32']) & np.isfinite(preds['fp16']) & (preds['fp32'] > 0) & (preds['fp16'] > 0)
            fp16_dlog.append(float(np.median(np.abs(np.log(preds['fp16'][ok] / preds['fp32'][ok])))))
        print(f'  {name} {seq.scene}: {len(idx)} frames in {time.perf_counter() - t_scene:.0f} s', flush=True)
    del m
    torch.cuda.empty_cache()
    return {p: {k: np.array(v) for k, v in r.items()} for p, r in rec.items()}, np.concatenate(pairs_all), \
        np.array(fp16_dlog)


def summarize_model(rec, pairs):
    out = {}
    scenes = rec['scene']
    for sc in list(dict.fromkeys(scenes)) + ['pooled']:
        if sc == 'pooled':
            sel, pr = np.ones(len(scenes), bool), pairs
        else:
            sel = scenes == sc
            pos = np.flatnonzero(sel)
            remap = -np.ones(len(scenes), np.int64)
            remap[pos] = np.arange(len(pos))
            pr = remap[pairs[np.all(sel[pairs], axis=1)]]
        s = summarize(rec['r'][sel], pr, rec['absrel'][sel])
        s['absrel_med'] = float(np.nanmean(rec['absrel_med'][sel]))
        s['fwd_ms_median'] = float(np.median(rec['fwd_ms'][sel]))
        s['total_ms_median'] = float(np.median(rec['total_ms'][sel]))
        out[sc] = s
    out['pooled']['jitter_within_scene_median'] = float(np.median(
        [out[sc]['jitter'] for sc in out if sc != 'pooled']))
    return out


def diag_dav2(seqs, stride):
    """DAv2-S FP32: (a) input-size sweep; (b) centre crops that change f/W (crop c -> f/W' = f/(cW)).
    A network that has absorbed the Hypersim focal predicts depth ∝ (f/W)^-1, so r_crop / r_full = c.
    A focal-blind network gives r_crop / r_full = 1."""
    out = {'input_size': {}, 'crop': {}}
    frames = [(seq, i) for seq in seqs for i in range(0, len(seq), stride)]
    data = [(read_bgr(seq.image_paths[i]), load_depth(seq.depth_paths[i])) for seq, i in frames]
    for size in (392, 518, 644):
        m = build('dav2s', seqs[0].K, fp16=False, input_size=size)
        r = np.array([frame_stats(m.predict_image(bgr)[0], gt)['r'] for bgr, gt in data])
        out['input_size'][size] = dict(b_median=float(np.nanmedian(r) - 1), n=int(np.isfinite(r).sum()))
        del m
    m = build('dav2s', seqs[0].K, fp16=False)
    r_full = None
    for c in (1.0, 0.9, 0.8, 0.7):
        rs = []
        for bgr, gt in data:
            h, w = gt.shape
            ch, cw = int(round(h * c)), int(round(w * c))
            y0, x0 = (h - ch) // 2, (w - cw) // 2
            pred, _ = m.predict_image(np.ascontiguousarray(bgr[y0:y0 + ch, x0:x0 + cw]))
            rs.append(frame_stats(pred, gt[y0:y0 + ch, x0:x0 + cw])['r'])
        rs = np.array(rs)
        if r_full is None:
            r_full = rs
        ratio = rs / r_full
        out['crop'][c] = dict(f_over_w=float(seqs[0].K[0, 0] / (c * w)), b_median=float(np.nanmedian(rs) - 1),
                              ratio_to_full_median=float(np.nanmedian(ratio)), expected_if_focal_absorbed=c)
    del m
    torch.cuda.empty_cache()
    return out


def fmt_tables(summary, fp16_dlog, f_over_w):
    canon = f_over_w / HYPERSIM_F_OVER_W
    lines = ['## Pooled (validation, 5 scenes)', '',
             '| model | prec | frames | pairs | b median | b mean | jitter σ(log r) | within-scene jitter (median) '
             '| f2f jitter σ(Δlog r) | AbsRel | AbsRel (median-scaled) | fwd ms | total ms |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for name in summary:
        for p in PRECS:
            s = summary[name][p]['pooled']
            lines.append(f"| {name} | {p} | {s['n_frames']} | {s['n_pairs']} | {100 * s['b_median']:+.2f} % | "
                         f"{100 * s['b_mean']:+.2f} % | {100 * s['jitter']:.2f} % | "
                         f"{100 * s['jitter_within_scene_median']:.2f} % | {100 * s['f2f_jitter']:.3f} % | "
                         f"{100 * s['absrel']:.1f} % | {100 * s['absrel_med']:.1f} % | {s['fwd_ms_median']:.1f} | "
                         f"{s['total_ms_median']:.1f} |")
    lines += ['', 'σ(log r) in % = 100·std of natural-log ratio. fwd ms = network + post-processing on GPU '
              '(MoGe includes the CPU shift solve); total ms = predict_image wall time incl. preprocessing. '
              'RTX 5090 shared with other jobs.', '']
    lines += ['## Per scene', '', '| model | prec | scene | frames | b median | b mean | jitter | f2f jitter | AbsRel |',
              '|---|---|---|---|---|---|---|---|---|']
    for name in summary:
        for p in PRECS:
            for sc, s in summary[name][p].items():
                if sc == 'pooled':
                    continue
                lines.append(f"| {name} | {p} | {sc} | {s['n_frames']} | {100 * s['b_median']:+.2f} % | "
                             f"{100 * s['b_mean']:+.2f} % | {100 * s['jitter']:.2f} % | {100 * s['f2f_jitter']:.3f} % | "
                             f"{100 * s['absrel']:.1f} % |")
    lines += ['', '## FP16 vs FP32 (same frames)', '', '| model | median_t median_px abs log(d16/d32) | Δb (fp16 − fp32) |',
              '|---|---|---|']
    for name in summary:
        db = summary[name]['fp16']['pooled']['b_median'] - summary[name]['fp32']['pooled']['b_median']
        lines.append(f"| {name} | {100 * np.median(fp16_dlog[name]):.3f} % | {100 * db:+.3f} %p |")
    if 'dav2s' in summary:
        lines += ['', f'DAv2 focal-canonicalised (×(f/W)_Redwood/(f/W)_Hypersim = ×{canon:.4f}, a constant, so jitter '
                  'is unchanged):', '']
        for p in PRECS:
            b = summary['dav2s'][p]['pooled']['b_median']
            lines.append(f'- {p}: b median {100 * b:+.2f} % -> {100 * ((1 + b) * canon - 1):+.2f} %')
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', default='validation')
    ap.add_argument('--scenes', nargs='+', default=list(SCENES))
    ap.add_argument('--models', nargs='+', default=list(MODELS), choices=MODELS)
    ap.add_argument('--stride', type=int, default=6, help='anchor stride; each anchor t adds frames t and t+1')
    ap.add_argument('--diag', action='store_true', help='DAv2 input-size and centre-crop (focal) diagnostics')
    ap.add_argument('--diag-stride', type=int, default=60)
    ap.add_argument('--out', type=Path, default=Path('runs/e1_depth_bias'))
    ap.add_argument('--resume', action='store_true', help='reuse frames_<model>_*.npz already in --out')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    seqs = [load_sequence(sc, args.split) for sc in args.scenes]
    K = seqs[0].K
    assert all(np.allclose(s.K, K) for s in seqs)
    summary, fp16_dlog = {}, {}
    for name in args.models:
        files = {p: args.out / f'frames_{name}_{p}.npz' for p in PRECS}
        if args.resume and all(f.exists() for f in files.values()):
            print(f'[{name}] reusing {args.out}', flush=True)
            z = {p: dict(np.load(f)) for p, f in files.items()}
            pairs, fp16_dlog[name] = z['fp32'].pop('pairs'), z['fp32'].pop('fp16_dlog')
            rec = {p: {k: v for k, v in z[p].items() if k not in ('pairs', 'fp16_dlog')} for p in PRECS}
        else:
            print(f'[{name}]', flush=True)
            rec, pairs, fp16_dlog[name] = run_model(name, seqs, args.stride, {})
            for p in PRECS:
                np.savez(files[p], pairs=pairs, fp16_dlog=fp16_dlog[name], **rec[p])
        summary[name] = {p: summarize_model(rec[p], pairs) for p in PRECS}
        summary[name]['fp16_vs_fp32_median_abs_dlog'] = float(np.median(fp16_dlog[name]))
    meta = dict(split=args.split, scenes=args.scenes, stride=args.stride, K=K.tolist(),
                torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
    result = dict(meta=meta, summary=summary)
    if args.diag:
        print('[diag dav2s]', flush=True)
        result['dav2_diag'] = diag_dav2(seqs, args.diag_stride)
    (args.out / 'summary.json').write_text(json.dumps(result, indent=1, default=float))
    md = fmt_tables(summary, fp16_dlog, K[0, 0] / seqs[0].width)
    if args.diag:
        d = result['dav2_diag']
        md += '\n## DAv2-S FP32 diagnostics\n\n| input_size | b median |\n|---|---|\n'
        md += ''.join(f"| {k} | {100 * v['b_median']:+.2f} % |\n" for k, v in d['input_size'].items())
        md += ('\n| centre crop c | f/W | b median | median r_crop/r_full | expected if focal absorbed |\n'
               '|---|---|---|---|---|\n')
        md += ''.join(f"| {c} | {v['f_over_w']:.3f} | {100 * v['b_median']:+.2f} % | {v['ratio_to_full_median']:.4f} | "
                      f"{v['expected_if_focal_absorbed']:.3f} |\n" for c, v in d['crop'].items())
    (args.out / 'tables.md').write_text(md)
    print(md)


if __name__ == '__main__':
    main()
