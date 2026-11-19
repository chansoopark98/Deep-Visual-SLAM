"""Tracker caches on disk (klt_depth_cache.py, klt_flow.py): <scene>.meta.json provenance and its H x W check."""
import hashlib
import json

import numpy as np
import pytest

from dvslam.odometry.base import Frame
from dvslam.odometry.klt_depth_cache import CachedDepth, write_cache, write_meta
from dvslam.odometry.klt_flow import FlowCache


def _depth_cache(root, hw=(4, 6)):
    paths = ['/c/a/0.png', '/c/a/1.png']
    depths = np.arange(2 * hw[0] * hw[1], dtype=np.float32).reshape(2, *hw)
    ckpt = root / 'best.pt'
    ckpt.write_bytes(b'weights')
    write_cache(root, 'room9', paths, depths, model='SSLDepth', ckpt=ckpt)
    return paths, depths


def test_depth_cache_records_provenance_and_checks_size(tmp_path):
    paths, depths = _depth_cache(tmp_path)
    meta = json.loads((tmp_path / 'room9.meta.json').read_text())
    assert meta['hw'] == [4, 6] and meta['model'] == 'SSLDepth' and meta['ckpt'].endswith('best.pt')
    assert meta['ckpt_sha256'] == hashlib.sha256(b'weights').hexdigest() and meta['created']
    c = CachedDepth(tmp_path, hw=(4, 6))
    np.testing.assert_array_equal(c.predict(Frame(1, 0.0, paths[1]))[0], depths[1])
    assert c.meta['room9']['model'] == 'SSLDepth'
    with pytest.raises(ValueError, match='4 x 6'):
        CachedDepth(tmp_path, hw=(512, 512)).predict(Frame(0, 0.0, paths[0]))


def test_depth_cache_without_meta_loads_as_before(tmp_path):
    paths, depths = _depth_cache(tmp_path)
    (tmp_path / 'room9.meta.json').unlink()
    c = CachedDepth(tmp_path, hw=(512, 512))                # no meta.json: nothing to check against
    np.testing.assert_array_equal(c.predict(Frame(0, 0.0, paths[0]))[0], depths[0])
    assert c.meta == {}
    np.testing.assert_array_equal(CachedDepth(tmp_path).predict(Frame(1, 0.0, paths[1]))[0], depths[1])


def _flow_cache(root, hw=(4, 6), meta=True):
    paths = ['/c/a/0.png', '/c/a/1.png']
    fwd = np.random.default_rng(0).normal(size=(2, 2, *hw)).astype(np.float32)
    np.save(root / 'room9.fwd.npy', fwd)
    np.save(root / 'room9.bwd.npy', -fwd)
    (root / 'room9.paths.txt').write_text(''.join(f'{p}\n' for p in paths))
    if meta:
        write_meta(root, 'room9', hw, 'torchvision raft_large')
    return paths, fwd


def test_flow_cache_checks_recorded_size(tmp_path):
    paths, fwd = _flow_cache(tmp_path)
    c = FlowCache(tmp_path, hw=(4, 6))
    f, b = c.pair(paths[1])
    np.testing.assert_array_equal(f, fwd[1])
    np.testing.assert_array_equal(b, -fwd[1])
    assert c.meta['room9']['ckpt'] is None and c.meta['room9']['hw'] == [4, 6]
    with pytest.raises(ValueError, match='4 x 6'):
        FlowCache(tmp_path, hw=(768, 768)).pair(paths[1])


def test_flow_cache_without_meta_loads_as_before(tmp_path):
    paths, fwd = _flow_cache(tmp_path, meta=False)
    c = FlowCache(tmp_path, hw=(768, 768))
    np.testing.assert_array_equal(c.pair(paths[1])[0], fwd[1])
    assert c.meta == {}
