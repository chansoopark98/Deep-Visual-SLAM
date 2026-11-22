"""Score a run directory of <scene>.inc.npz files: pair-wise pose errors + diagnostics (plan v3 §4.8, item 0.4).

Translation step error: mean |‖t_gt‖ − ‖t_pred‖| / ‖t_gt‖ (%) over consecutive pairs, no scale alignment.
Rotation error: mean |roll|, |pitch|, |yaw| error of the relative rotation (deg). Both are compared with the target
thresholds of dvslam/eval/metrics.py.

  python apps/eval_pairs.py --runs runs/<id>     -> prints and writes runs/<id>/eval_pairs.txt, summary.csv

PASS/FAIL is given for the two official sets: Redwood test (5 scenes) and TUM-VI train + test (room1-4, room6;
tumvi.SPLITS). Any other set gets raw values only.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dvslam.data import redwood, registry  # noqa: E402
from dvslam.eval import diagnostics as D  # noqa: E402
from dvslam.eval.metrics import (POS_MAPE_TARGET_PCT, ROT_ABS_TARGET_DEG, pair_metrics,  # noqa: E402
                                 scene_summary, target_items)
from dvslam.geometry.se3 import relative_pose  # noqa: E402
from dvslam.odometry.base import causal_trajectory, load_increments  # noqa: E402

OFFICIAL_TUMVI_SPLITS = ('train', 'test')               # official TUM-VI set (user decision 2026-11-19)


def load_run_scene(inc_path: Path, data_root: Path | None):
    """(increments, meta, GT sequence) of one scene of a run, checked to cover frames 1..n-1 in order."""
    inc = load_increments(inc_path)
    meta = inc['meta']
    dataset = meta.get('dataset', 'redwood')            # runs written before V0 have no 'dataset' key
    # same GT time offset as the run (it decides the kept frame range); images are not needed to score
    seq = registry.load_sequence(dataset, meta['scene'], meta['split'], data_root or meta['data_root'],
                                 meta.get('time_offset_s', 0.0), load_images=False)
    n = len(seq)
    if not np.array_equal(inc['index'], np.arange(1, n)):
        raise ValueError(f'{inc_path}: increments must cover frames 1..{n - 1} in order')
    return inc, meta, seq


def evaluate_scene(inc_path: Path, data_root: Path | None) -> tuple[dict, np.ndarray, dict]:
    inc, meta, seq = load_run_scene(inc_path, data_root)
    pos, rpy, gt_t, pred_t = [], [], [], []
    for k, t in enumerate(inc['index']):
        m = pair_metrics(relative_pose(seq.T_wc[t - 1], seq.T_wc[t]), inc['rel_increment'][k])
        pos.append(m['pos_error_percent'])
        rpy.append(m['rpy_abs_err_deg'])
        gt_t.append(m['gt_t'])
        pred_t.append(m['pred_t'])
    pos, rpy, gt_t, pred_t = np.array(pos), np.array(rpy), np.array(gt_t), np.array(pred_t)
    tracked = inc['source'] == 'tracked'
    frame_ok = getattr(seq, 'gt_valid', None)           # TUM-VI / EuRoC: False where the GT bridges a dropout
    ok = slice(None) if frame_ok is None else frame_ok
    if frame_ok is not None:                            # score only pairs with measured GT at both ends
        keep = seq.pair_valid
        pos, rpy, gt_t, pred_t, tracked = pos[keep], rpy[keep], gt_t[keep], pred_t[keep], tracked[keep]
    fps = int(round(meta.get('frame_rate_hz', 30.0)))   # 30 for Redwood: windows stay 30 / 60 frames
    s = scene_summary(meta['scene'], list(pos), rpy, gt_t)

    T_est = causal_trajectory(inc['rel_increment'])
    s.update({
        'tracked_pairs': int(tracked.sum()),
        'pos_mape_tracked_pct': float(pos[tracked].mean()) if tracked.any() else float('nan'),
        'median_ratio': D.median_step_ratio(gt_t, pred_t),
        'rescaled_mape_pct': D.best_rescale(gt_t, pred_t)[1],
        'win1s_rescaled_mape_pct': D.windowed_rescale_mape(gt_t, pred_t, fps),
        'win2s_rescaled_mape_pct': D.windowed_rescale_mape(gt_t, pred_t, 2 * fps),
        **D.sigma_parallel(gt_t, pred_t),
        'lt2mm_error_share': D.bin_contribution(gt_t, pred_t, edges_mm=(0, 2, np.inf))[0]['error_share'],
        'ate_se3_m': D.ate_rmse(seq.T_wc[ok], T_est[ok], sim3=False)[0],
        'ate_sim3_m': D.ate_rmse(seq.T_wc[ok], T_est[ok], sim3=True)[0],
        'rpe30_trans_m': D.rpe(seq.T_wc, T_est, fps, valid=frame_ok)['trans_rmse_m'],   # 1 s window
        'g1p_stride3_mape_pct': D.stride_mape(seq.T_wc, inc['rel_increment'], 3, valid=frame_ok),
        'g1p_min2mm_mape_pct': D.min_motion_mape(gt_t, pred_t),
    })
    return s, pos, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=Path, required=True)
    ap.add_argument('--data-root', type=Path, default=None, help='override the root stored in each .inc.npz')
    args = ap.parse_args()

    files = sorted(args.runs.glob('*.inc.npz'))
    if not files:
        sys.exit(f'no *.inc.npz in {args.runs}')
    summaries, per_scene_pos, metas = [], [], []
    for f in files:
        s, pos, meta = evaluate_scene(f, args.data_root)
        summaries.append(s)
        per_scene_pos.append(pos)
        metas.append(meta)
    if len({m.get('dataset', 'redwood') for m in metas}) > 1:
        sys.exit(f'{args.runs}: mixes datasets; score one dataset per run directory')
    k = target_items(summaries)
    win = {w: float(np.mean([s[f'win{w}s_rescaled_mape_pct'] for s in summaries])) for w in (1, 2)}
    fps = int(round(metas[0].get('frame_rate_hz', 30.0)))
    _, lo, hi = D.block_bootstrap_ci(per_scene_pos, block=fps)     # 1 s blocks (30 pairs for Redwood)

    lines = [f'=== eval_pairs: {args.runs} ===', '',
             f'{"scene":<10} {"pairs":>6} {"trans %":>9} {"rot °":>9} {"ratio":>6} {"rescaled%":>9} '
             f'{"σ∥ mm":>7} {"<2mm share":>10} {"ATE se3 m":>9} {"ATE sim3 m":>10}']
    for s in summaries:
        lines.append(f'{s["scene"]:<10} {s["pairs"]:>6} {s["pos_mape_pct"]:>9.4f} {s["rot_rpy_mean_deg"]:>9.6f} '
                     f'{s["median_ratio"]:>6.3f} {s["rescaled_mape_pct"]:>9.2f} {s["sigma_par_mm"]:>7.3f} '
                     f'{s["lt2mm_error_share"]*100:>9.1f}% {s["ate_se3_m"]:>9.4f} {s["ate_sim3_m"]:>10.4f}')
    splits = {m['split'] for m in metas}
    datasets = {m.get('dataset', 'redwood') for m in metas}
    is_redwood = datasets == {'redwood'}
    scored = sorted(s['scene'] for s in summaries)
    complete = is_redwood and scored == sorted(redwood.SCENES) and splits == {'test'}
    official_tumvi = datasets == {'tumvi'} and splits == set(OFFICIAL_TUMVI_SPLITS) and scored == sorted(
        sc for sp in OFFICIAL_TUMVI_SPLITS for sc in registry.expected_scenes('tumvi', sp))
    verdict = is_redwood or official_tumvi
    lines.insert(1, f'split: {", ".join(sorted(splits))}')
    lines += ['',
              f'translation step error {k["item03_mape_pct"]:.4f} %  95 % block CI [{lo:.2f}, {hi:.2f}]  '
              f'(target ≤ {POS_MAPE_TARGET_PCT} %) -> '
              + (('PASS' if k['item03_pass'] else 'FAIL') if verdict else 'raw value, no verdict'),
              f'rotation error {k["item04_rpy_mean_deg"]:.6f} °  (target ≤ {ROT_ABS_TARGET_DEG} °) -> '
              + (('PASS' if k['item04_pass'] else 'FAIL') if verdict else 'raw value, no verdict'),
              f'G0 oracle (scale fitted per window, uses GT): 1 s {win[1]:.3f} %, 2 s {win[2]:.3f} %',
              'G1\' (NO-GO branch ranking): stride-3 translation {:.3f} %, min-2mm translation {:.3f} %, RPE 1 s {:.4f} m, ATE SE3 {:.4f} m'.format(
                  *(float(np.mean([s[c] for s in summaries])) for c in ('g1p_stride3_mape_pct', 'g1p_min2mm_mape_pct', 'rpe30_trans_m', 'ate_se3_m'))),
              f'tracked-only pairs: {sum(s["tracked_pairs"] for s in summaries)} / {sum(s["pairs"] for s in summaries)}']
    if not is_redwood:
        tau = sorted({m.get('time_offset_s', 0.0) for m in metas})
        if official_tumvi:
            lines.append('Official set: TUM-VI train + test (room1-4, room6), raw errors; pairs without measured '
                         'GT are excluded. The model may have been trained on these sequences: say so in the report')
        else:
            lines.append(f'dataset: {", ".join(sorted(datasets))} -> the errors above are raw values (plan §12.2 '
                         f'diagnostic c), pairs without measured GT are excluded; PASS/FAIL only on the official set '
                         f'(TUM-VI train + test)')
        lines.append(f'GT time offset used: {tau} s (not measured yet: plan T.1); bootstrap blocks and the 1 s / 2 s '
                     f'windows run over the kept pairs, so they span GT dropouts')
        if len(datasets) == 1:
            expected = {sc for sp in splits for sc in registry.expected_scenes(*datasets, sp)}
            missing = sorted(expected - set(scored))
            if missing:
                lines.append(f'INCOMPLETE: {missing} of the {"+".join(sorted(splits))} split not scored')
    elif not complete:
        lines.append(f'INCOMPLETE: scenes {sorted(s["scene"] for s in summaries)}, split {sorted(splits)} '
                     f'-> not an official result (needs the 5 test scenes once each)')
    report = '\n'.join(lines)
    print(report)
    (args.runs / 'eval_pairs.txt').write_text(report + '\n', encoding='utf-8')
    with open(args.runs / 'summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0]))
        w.writeheader()
        w.writerows(summaries)


if __name__ == '__main__':
    main()
