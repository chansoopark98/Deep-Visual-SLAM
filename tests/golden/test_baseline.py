"""Golden test: dvslam.eval.metrics must reproduce the frozen 2026 target-metric baseline (plan v3 item 0.2)."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dvslam.eval.metrics import target-metric_items, position_error_pct, rpy_abs_err_deg, scene_summary

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE = REPO_ROOT / 'baselines/pairs_eval/20261118-125421'
SCENES = ['apartment', 'bedroom', 'boardroom', 'lobby', 'loft']
ITEM03_PCT, ITEM04_DEG = 40.0591, 0.096318
TOL03_PCT, TOL04_DEG = 1e-4, 1e-6


def load_rows(scene: str) -> dict[str, np.ndarray]:
    with open(BASELINE / f'{scene}_pose_errors.csv') as f:
        rows = list(csv.DictReader(f))
    cols = ['gt_tx', 'gt_ty', 'gt_tz', 'gt_roll', 'gt_pitch', 'gt_yaw',
            'pred_tx', 'pred_ty', 'pred_tz', 'pred_roll', 'pred_pitch', 'pred_yaw', 'pos_error_percent',
            'rot_abs_err_roll_deg', 'rot_abs_err_pitch_deg', 'rot_abs_err_yaw_deg']
    return {c: np.array([float(r[c]) for r in rows]) for c in cols}


def recompute_scene(scene: str) -> tuple[dict, dict]:
    r = load_rows(scene)
    gt_t = np.stack([r['gt_tx'], r['gt_ty'], r['gt_tz']], 1)
    pred_t = np.stack([r['pred_tx'], r['pred_ty'], r['pred_tz']], 1)
    gt_e = np.stack([r['gt_roll'], r['gt_pitch'], r['gt_yaw']], 1)
    pred_e = np.stack([r['pred_roll'], r['pred_pitch'], r['pred_yaw']], 1)
    pos = [position_error_pct(g, p) for g, p in zip(gt_t, pred_t)]
    rpy = np.stack([rpy_abs_err_deg(g, p) for g, p in zip(gt_e, pred_e)])
    return scene_summary(scene, pos, rpy, gt_t), r | {'pos_recomputed': np.array(pos), 'rpy_recomputed': rpy}


def test_per_pair_values_match_frozen_csv():
    for scene in SCENES:
        _, r = recompute_scene(scene)
        np.testing.assert_allclose(r['pos_recomputed'], r['pos_error_percent'], rtol=1e-12, atol=1e-12)
        stored = np.stack([r['rot_abs_err_roll_deg'], r['rot_abs_err_pitch_deg'], r['rot_abs_err_yaw_deg']], 1)
        np.testing.assert_allclose(r['rpy_recomputed'], stored, rtol=1e-12, atol=1e-15)


def test_target-metric_items_match_frozen_baseline():
    k = target-metric_items([recompute_scene(s)[0] for s in SCENES])
    assert abs(k['item03_mape_pct'] - ITEM03_PCT) <= TOL03_PCT
    assert abs(k['item04_rpy_mean_deg'] - ITEM04_DEG) <= TOL04_DEG
    assert not k['item03_pass'] and k['item04_pass']


def test_scene_summaries_match_frozen_summary_csv():
    with open(BASELINE / 'summary.csv') as f:
        frozen = {row['scene']: row for row in csv.DictReader(f)}
    for scene in SCENES:
        s, _ = recompute_scene(scene)
        assert s['pairs'] == int(frozen[scene]['pairs'])
        for key in ('pos_mape_pct', 'rot_roll_deg', 'rot_pitch_deg', 'rot_yaw_deg', 'rot_rpy_mean_deg'):
            assert s[key] == pytest.approx(float(frozen[scene][key]), rel=1e-12)


def test_manifest_hashes():
    manifest = json.loads((BASELINE / 'MANIFEST.json').read_text())
    if not Path(manifest['data_root']).exists():
        pytest.skip('Redwood data root not mounted')
    out = subprocess.run([sys.executable, str(REPO_ROOT / 'scripts/verify_baseline.py')], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
