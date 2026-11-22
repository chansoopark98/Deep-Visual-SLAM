"""Validation stage V0: the apps' --dataset path and the exclusion of pairs without measured GT."""
import sys
from pathlib import Path

import numpy as np
import pytest

from dvslam.data import registry
from dvslam.eval import diagnostics as D
from dvslam.geometry.se3 import relative_pose
from dvslam.odometry.base import PoseEstimate, save_increments

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'apps'))
import eval_pairs  # noqa: E402
import run_sequence  # noqa: E402


def test_registry_rejects_unknown_dataset():
    with pytest.raises(ValueError):
        registry.scenes('kitti', 'test')


def test_registry_redwood_is_unchanged():
    from dvslam.data import redwood
    assert registry.scenes('redwood', 'test') == list(redwood.SCENES)
    assert registry.default_root('redwood') == redwood.DEFAULT_ROOT
    assert registry.frame_rate('redwood') == (redwood.FRAME_RATE_HZ, redwood.FRAME_RATE_VERIFIED)


def test_registry_knows_the_defined_split_even_when_sequences_are_missing():
    assert registry.expected_scenes('tumvi', 'train') == ['room1', 'room2', 'room3', 'room4']
    assert registry.expected_scenes('tumvi', 'test') == ['room6']
    assert registry.expected_scenes('euroc', 'validation') == ['MH_01_easy', 'MH_02_easy']
    assert set(registry.scenes('tumvi', 'train')) <= set(registry.expected_scenes('tumvi', 'train'))


def test_registry_rejects_a_split_the_dataset_does_not_have():
    with pytest.raises(ValueError, match='no split'):
        registry.scenes('euroc', 'train')
    with pytest.raises(ValueError, match='no split'):
        registry.expected_scenes('tumvi', 'all')


def test_redwood_rejects_a_gt_time_offset():
    with pytest.raises(ValueError, match='no timestamps'):
        registry.load_sequence('redwood', 'apartment', 'test', time_offset_s=0.01)


def test_rpe_and_stride_skip_frames_without_gt():
    n = 40
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, 0, 3] = 0.05 * np.arange(n)                     # 50 mm steps along x
    inc = np.stack([relative_pose(T[i - 1], T[i]) for i in range(1, n)])
    T_gt = T.copy()
    T_gt[10:15, 1, 3] += 0.3                             # GT is wrong where it was interpolated across a dropout
    valid = np.ones(n, bool)
    valid[10:15] = False
    assert D.rpe(T_gt, T, 5)['trans_rmse_m'] > 0.05
    assert D.rpe(T_gt, T, 5, valid=valid)['trans_rmse_m'] < 1e-12
    assert D.stride_mape(T_gt, inc, 3) > 1.0
    assert D.stride_mape(T_gt, inc, 3, valid=valid) < 1e-9
    # default (valid=None) is the old behaviour
    assert D.rpe(T, T, 5)['trans_rmse_m'] < 1e-12 and D.stride_mape(T, inc, 3) < 1e-9


@pytest.mark.parametrize('dataset,scene,split', [('tumvi', 'room1', 'train'), ('euroc', 'MH_01_easy', 'validation')])
def test_eval_scores_gt_increments_as_zero_and_drops_pairs_without_gt(dataset, scene, split, tmp_path):
    try:
        present = scene in registry.scenes(dataset, split)
    except FileNotFoundError:
        present = False
    if not present:
        pytest.skip(f'{dataset}/{scene} not on disk')
    seq = registry.load_sequence(dataset, scene, split)
    estimates = []
    for t in range(1, len(seq)):
        # a tracker cannot know where the GT has holes: give those pairs an identity increment, which would
        # score 100 % if the evaluator counted them
        dT = relative_pose(seq.T_wc[t - 1], seq.T_wc[t]) if seq.pair_valid[t - 1] else np.eye(4)
        estimates.append(PoseEstimate(t, float(seq.timestamps[t]), dT))
    meta = {'dataset': dataset, 'scene': scene, 'split': split, 'data_root': str(registry.default_root(dataset)),
            'n_frames': len(seq), 'frame_rate_hz': seq.frame_rate_hz, 'frame_rate_verified': True}
    path = tmp_path / f'{scene}.inc.npz'
    save_increments(path, estimates, meta)
    s, pos, _ = eval_pairs.evaluate_scene(path, None)
    assert s['pairs'] == int(seq.pair_valid.sum()) == len(pos)
    assert s['pos_mape_pct'] < 1e-6
    assert s['rot_rpy_mean_deg'] < 1e-6
    assert np.isfinite([s['ate_se3_m'], s['ate_sim3_m'], s['rpe30_trans_m'], s['g1p_stride3_mape_pct']]).all()


def _gt_run(tmp_path, dataset, scene, split=None):
    split = split or next(k for k, v in registry._module(dataset).SPLITS.items() if scene in v)
    seq = registry.load_sequence(dataset, scene, split, load_images=False)
    est = [PoseEstimate(t, float(seq.timestamps[t]),
                        relative_pose(seq.T_wc[t - 1], seq.T_wc[t]) if seq.pair_valid[t - 1] else np.eye(4))
           for t in range(1, len(seq))]
    meta = {'dataset': dataset, 'scene': scene, 'split': split, 'data_root': str(registry.default_root(dataset)),
            'n_frames': len(seq), 'frame_rate_hz': seq.frame_rate_hz, 'frame_rate_verified': True}
    save_increments(tmp_path / f'{scene}.inc.npz', est, meta)


def test_eval_report_flags_a_partial_split_and_gives_no_verdict(tmp_path, monkeypatch):
    if 'room3' not in registry.scenes('tumvi', 'train'):
        pytest.skip('tumvi/room3 not on disk')
    _gt_run(tmp_path, 'tumvi', 'room3')
    monkeypatch.setattr(sys, 'argv', ['eval_pairs.py', '--runs', str(tmp_path)])
    eval_pairs.main()
    report = (tmp_path / 'eval_pairs.txt').read_text()
    assert 'INCOMPLETE' in report and 'room4' in report      # the other train rooms were not scored
    assert 'raw value, no verdict' in report
    assert 'GT time offset used: [0.0] s' in report


def test_eval_refuses_a_run_directory_that_mixes_datasets(tmp_path, monkeypatch):
    if 'room1' not in registry.scenes('tumvi', 'train') or 'MH_01_easy' not in registry.scenes('euroc', 'validation'):
        pytest.skip('tumvi/room1 or euroc/MH_01_easy not on disk')
    _gt_run(tmp_path, 'tumvi', 'room1')
    _gt_run(tmp_path, 'euroc', 'MH_01_easy', 'validation')
    monkeypatch.setattr(sys, 'argv', ['eval_pairs.py', '--runs', str(tmp_path)])
    with pytest.raises(SystemExit, match='mixes datasets'):
        eval_pairs.main()


def _all_rooms_on_disk(rooms):
    have = {sc for sp in ('train', 'validation', 'test') for sc in registry.scenes('tumvi', sp)}
    if not set(rooms) <= have:
        pytest.skip(f'TUM-VI {sorted(set(rooms) - have)} not on disk')


def test_eval_gives_a_verdict_only_on_the_tumvi_official_set(tmp_path, monkeypatch):
    rooms = ['room1', 'room2', 'room3', 'room4', 'room6']            # train + test
    _all_rooms_on_disk(rooms + ['room5'])
    for r in rooms:
        _gt_run(tmp_path, 'tumvi', r)
    monkeypatch.setattr(sys, 'argv', ['eval_pairs.py', '--runs', str(tmp_path)])
    eval_pairs.main()
    report = (tmp_path / 'eval_pairs.txt').read_text()
    assert 'Official set: TUM-VI train + test' in report and 'INCOMPLETE' not in report
    assert 'translation step error' in report
    assert '-> PASS' in report.split('translation step error')[1].splitlines()[0]
    assert '-> PASS' in report.split('rotation error')[1].splitlines()[0]
    # adding the validation room makes it a different set: raw values only
    _gt_run(tmp_path, 'tumvi', 'room5')
    eval_pairs.main()
    report = (tmp_path / 'eval_pairs.txt').read_text()
    assert 'raw value, no verdict' in report and 'Official set: TUM-VI' not in report
    assert 'raw value, no verdict' in report.split('rotation error')[1].splitlines()[0]


def test_run_sequence_refuses_several_redwood_splits(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['run_sequence.py', '--odometry', 'klt_window', '--split', 'train', 'test',
                                      '--out', str(tmp_path / 'r')])
    with pytest.raises(SystemExit, match='same five scenes'):
        run_sequence.main()
