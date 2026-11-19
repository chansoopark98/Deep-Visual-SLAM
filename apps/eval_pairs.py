"""Score a run directory of <scene>.inc.npz files: target-metric items 03/04 + diagnostics (plan v3 §4.8, item 0.4).

  python apps/eval_pairs.py --runs runs/<id>     -> prints and writes runs/<id>/eval_pairs.txt, summary.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dvslam.data import redwood, registry  # noqa: E402
from dvslam.eval import diagnostics as D  # noqa: E402
from dvslam.eval.metrics import (POS_MAPE_TARGET_PCT, ROT_ABS_TARGET_DEG, target-metric_items,  # noqa: E402
                                 pair_metrics, scene_summary)
from dvslam.geometry.se3 import relative_pose  # noqa: E402
from dvslam.odometry.base import causal_trajectory, load_increments  # noqa: E402
from scripts.verify_baseline import DEFAULT_BASELINE, sha256_of  # noqa: E402

BASELINE_MANIFEST = DEFAULT_BASELINE / 'MANIFEST.json'


def evaluate_scene(inc_path: Path, data_root: Path | None) -> tuple[dict, np.ndarray, dict]:
    inc = load_increments(inc_path)
    meta = inc['meta']
    dataset = meta.get('dataset', 'redwood')            # runs written before V0 have no 'dataset' key
    # same GT time offset as the run (it decides the kept frame range); images are not needed to score
    seq = registry.load_sequence(dataset, meta['scene'], meta['split'], data_root or meta['data_root'],
                                 meta.get('time_offset_s', 0.0), load_images=False)
    n = len(seq)
    if not np.array_equal(inc['index'], np.arange(1, n)):
        raise ValueError(f'{inc_path}: increments must cover frames 1..{n - 1} in order')
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


def check_test_gt(metas: list[dict], data_root: Path | None) -> str:
    """Compare each scored test scene's GT json against the sha256 frozen in the baseline MANIFEST."""
    manifest = json.loads(BASELINE_MANIFEST.read_text())['data']
    status = 'OK'
    for meta in metas:
        if meta['split'] != 'test' or meta['scene'] not in manifest:
            return 'n/a'
        gt = Path(data_root or meta['data_root']) / manifest[meta['scene']]['gt_json']
        if sha256_of(gt) != manifest[meta['scene']]['gt_json_sha256']:
            status = 'MISMATCH'
    return status


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
    k = target-metric_items(summaries)
    win = {w: float(np.mean([s[f'win{w}s_rescaled_mape_pct'] for s in summaries])) for w in (1, 2)}
    gt_check = check_test_gt(metas, args.data_root)
    fps = int(round(metas[0].get('frame_rate_hz', 30.0)))
    _, lo, hi = D.block_bootstrap_ci(per_scene_pos, block=fps)     # 1 s blocks (30 pairs for Redwood)

    lines = [f'=== eval_pairs: {args.runs} ===', '',
             f'{"scene":<10} {"pairs":>6} {"03 MAPE%":>9} {"04 RPY°":>9} {"ratio":>6} {"rescaled%":>9} '
             f'{"σ∥ mm":>7} {"<2mm share":>10} {"ATE se3 m":>9} {"ATE sim3 m":>10}']
    for s in summaries:
        lines.append(f'{s["scene"]:<10} {s["pairs"]:>6} {s["pos_mape_pct"]:>9.4f} {s["rot_rpy_mean_deg"]:>9.6f} '
                     f'{s["median_ratio"]:>6.3f} {s["rescaled_mape_pct"]:>9.2f} {s["sigma_par_mm"]:>7.3f} '
                     f'{s["lt2mm_error_share"]*100:>9.1f}% {s["ate_se3_m"]:>9.4f} {s["ate_sim3_m"]:>10.4f}')
    splits = {m['split'] for m in metas}
    datasets = {m.get('dataset', 'redwood') for m in metas}
    is_redwood = datasets == {'redwood'}
    complete = is_redwood and sorted(s['scene'] for s in summaries) == sorted(redwood.SCENES) and splits == {'test'}
    lines.insert(1, f'split: {", ".join(sorted(splits))}   GT sha256 vs baseline MANIFEST: {gt_check}')
    lines += ['',
              f'[03] {k["item03_mape_pct"]:.4f} %  95 % block CI [{lo:.2f}, {hi:.2f}]  (≤ {POS_MAPE_TARGET_PCT} %) -> '
              + (('PASS' if k['item03_pass'] else 'FAIL') if is_redwood else 'raw value, no verdict'),
              f'[04] {k["item04_rpy_mean_deg"]:.6f} °  (≤ {ROT_ABS_TARGET_DEG} °) -> {"PASS" if k["item04_pass"] else "FAIL"}',
              f'G0 oracle (scale fitted per window, uses GT): 1 s {win[1]:.3f} %, 2 s {win[2]:.3f} %',
              'G1\' (NO-GO branch ranking): stride-3 item 03 {:.3f} %, min-2mm item 03 {:.3f} %, RPE 1 s {:.4f} m, ATE SE3 {:.4f} m'.format(
                  *(float(np.mean([s[c] for s in summaries])) for c in ('g1p_stride3_mape_pct', 'g1p_min2mm_mape_pct', 'rpe30_trans_m', 'ate_se3_m'))),
              f'tracked-only pairs: {sum(s["tracked_pairs"] for s in summaries)} / {sum(s["pairs"] for s in summaries)}']
    if not is_redwood:
        tau = sorted({m.get('time_offset_s', 0.0) for m in metas})
        lines.append(f'dataset: {", ".join(sorted(datasets))} -> items 03/04 above are raw values (plan §12.2 diagnostic c), '
                     f'pairs without measured GT are excluded, and the verdict needs metric (a), not this PASS/FAIL')
        lines.append(f'GT time offset used: {tau} s (not measured yet: plan T.1); bootstrap blocks and the 1 s / 2 s '
                     f'windows run over the kept pairs, so they span GT dropouts')
        if len(datasets) == 1 and len(splits) == 1:
            missing = sorted(set(registry.expected_scenes(*datasets, *splits)) - {s['scene'] for s in summaries})
            if missing:
                lines.append(f'INCOMPLETE: {missing} of the {", ".join(splits)} split not scored')
    elif not complete:
        lines.append(f'INCOMPLETE: scenes {sorted(s["scene"] for s in summaries)}, split {sorted(splits)} '
                     f'-> not an official result (needs the 5 test scenes once each)')
    if gt_check == 'MISMATCH':
        lines.append('GT MISMATCH: test GT differs from the frozen baseline manifest -> not comparable to 40.0591 %')
    report = '\n'.join(lines)
    print(report)
    (args.runs / 'eval_pairs.txt').write_text(report + '\n', encoding='utf-8')
    with open(args.runs / 'summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0]))
        w.writeheader()
        w.writerows(summaries)


if __name__ == '__main__':
    main()
