"""Leakage tests for the Phase 3.1 depth fine-tuning lists (dvslam.train.depth_data). Fast, no GPU.

Synthetic trees exercise the logic everywhere; the real-data tests run when Redwood / TUM are mounted.
"""
import numpy as np
import pytest

from dvslam.data.redwood import DEFAULT_ROOT as REDWOOD_ROOT, SCENES
from dvslam.train import depth_data as D


def _fake_redwood(root, n=12):
    for sp in ('train', 'validation', 'test'):
        for sc in SCENES:
            for sub in ('image', 'depth'):
                (root / sp / sc / sub).mkdir(parents=True)
            for i in range(n):
                (root / sp / sc / 'image' / f'{i:06d}.jpg').touch()
                (root / sp / sc / 'depth' / f'{i:06d}.png').touch()
    return root


def _fake_tum(root, names=('rgbd_dataset_freiburg1_desk', 'rgbd_dataset_freiburg3_cabinet') + D.TUM_HOLDOUT, n=20):
    for name in names:
        d = root / name
        d.mkdir(parents=True)
        t = 1000.0 + np.arange(n) / 30.0
        (d / 'rgb.txt').write_text('# rgb\n' + ''.join(f'{x:.6f} rgb/{x:.6f}.png\n' for x in t))
        (d / 'depth.txt').write_text('# depth\n' + ''.join(f'{x + 0.005:.6f} depth/{x + 0.005:.6f}.png\n' for x in t))
    return root


@pytest.fixture
def fake_roots(tmp_path):
    return _fake_redwood(tmp_path / 'redwood'), _fake_tum(tmp_path / 'tum')


def _redwood_paths(lists, root):
    return [p for p in D.clip_paths(lists['external'] + lists['redwood']) if p.startswith(str(root))]


@pytest.mark.parametrize('room', SCENES)
def test_loso_fold_has_no_heldout_room(fake_roots, room):
    rw, tum = fake_roots
    L = D.fold_lists(f'loso-{room}', redwood_root=rw, tum_root=tum, redwood_stride=1)
    paths = _redwood_paths(L, rw)
    assert not [p for p in paths if f'/{room}/' in p]
    assert {p.split('/')[-3] for p in paths} == set(SCENES) - {room}   # the other 4 rooms are used
    assert L['eval'] == [(room, 'validation'), (room, 'test')]


def test_single_has_no_redwood(fake_roots):
    rw, tum = fake_roots
    L = D.fold_lists('single', redwood_root=rw, tum_root=tum)
    assert L['redwood'] == []
    assert _redwood_paths(L, rw) == []
    assert all(f.source == 'tum' for c in L['external'] for f in c)
    assert sorted({sc for sc, _ in L['eval']}) == sorted(SCENES)


def test_tum_holdout_excluded_from_training(fake_roots):
    rw, tum = fake_roots
    for fold in D.FOLDS:
        seqs = {f.seq for c in D.fold_lists(fold, redwood_root=rw, tum_root=tum)['external'] for f in c}
        assert seqs and not seqs & set(D.TUM_HOLDOUT)
    assert {f.seq for c in D.holdout_clips(tum_root=tum, stride=1) for f in c} == set(D.TUM_HOLDOUT)


def test_clips_are_consecutive_and_associated(fake_roots):
    _, tum = fake_roots
    fr = D.tum_frames(tum / 'rgbd_dataset_freiburg1_desk')
    assert len(fr) == 20 and fr[0].fx == 517.3
    clips = D.make_clips(fr, T=2, stride=4)
    assert all(c[1].rgb > c[0].rgb for c in clips) and len(clips) == 5


def test_canon_factor():
    assert D.canon_factor(525, 525, 640) == pytest.approx(1.0)
    # a longer focal sees the same content as a closer object for the deployment camera -> shrink depth
    assert D.canon_factor(535.4, 539.2, 640) == pytest.approx(525 / 537.3)


@pytest.mark.skipif(not (REDWOOD_ROOT / 'train').exists(), reason='Redwood not mounted')
@pytest.mark.parametrize('fold', D.FOLDS)
def test_real_lists_leakage(fold):
    L = D.fold_lists(fold, redwood_stride=50, tum_stride=50)
    rw = _redwood_paths(L, REDWOOD_ROOT)
    if fold == 'single':
        assert rw == []
    else:
        room = fold.removeprefix('loso-')
        assert rw and not [p for p in rw if f'/{room}/' in p]
