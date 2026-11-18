"""C++ frame-loop skeleton (plan v3 item 5.4): mailbox mask parity with dvslam.runtime.replay, drop policy, CSV I/O.

Skipped unless the binary is built:
    cmake -S cpp -B cpp/build -DCUDAToolkit_ROOT=/usr/local/cuda-13.2 \
          -DTENSORRT_ROOT=/usr/local/cuda-13.2/targets/x86_64-linux && cmake --build cpp/build -j
"""
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from dvslam.export.inc_csv import csv_to_inc, read_increments_csv, read_processed_mask
from dvslam.geometry.se3 import se3_exp
from dvslam.odometry.base import load_increments
from dvslam.runtime.replay import apply_drop_policy, simulate_mailbox

REPO = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get('DVSLAM_CPP_LOOP', REPO / 'cpp' / 'build' / 'dvslam_loop'))
pytestmark = pytest.mark.skipif(not BIN.exists(), reason=f'C++ loop not built ({BIN})')

TWIST = np.array([0.012, -0.003, 0.02, 0.004, 0.011, -0.002])   # (v, w) per frame


def run_loop(tmp_path, latencies, *args):
    lat = tmp_path / 'lat.txt'
    np.savetxt(lat, latencies, fmt='%.17g')
    src = [] if '--images-list' in args else ['--synthetic', str(len(latencies))]
    cmd = [str(BIN), 'run', '--out-dir', str(tmp_path), '--latencies', str(lat), *src, *args]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr
    return tmp_path


def _profiles():
    rng = np.random.default_rng(7)
    dt = 1000 / 30
    return {
        'fast': np.full(200, 10.0),
        'slow': np.full(200, 50.0),
        'exact_dt_ties': np.full(200, dt),
        'two_dt_ties': np.full(200, 2 * dt),
        'gamma': rng.gamma(4.0, 9.0, 400),
        'spiky': np.where(rng.random(400) < 0.1, 120.0, 12.0),
        'very_slow': np.full(100, 250.0),
    }


@pytest.mark.parametrize('name', list(_profiles()))
def test_mask_matches_python_mailbox(tmp_path, name):
    lat = _profiles()[name]
    out = run_loop(tmp_path, lat, '--tracker', 'identity')
    np.testing.assert_array_equal(read_processed_mask(out / 'frames.csv'), simulate_mailbox(lat))


def test_increments_match_python_drop_policy(tmp_path):
    lat = _profiles()['gamma']
    out = run_loop(tmp_path, lat, '--tracker', 'cv', '--cv-twist', ','.join(f'{v:.17g}' for v in TWIST))
    mask = simulate_mailbox(lat)
    V = se3_exp(TWIST)
    ref = apply_drop_policy(np.repeat(V[None], len(lat) - 1, 0), mask)
    est = read_increments_csv(out / 'increments.csv')
    np.testing.assert_allclose(np.stack([e.rel_increment for e in est]), ref['rel_increment'], atol=1e-9)
    assert [e.source for e in est] == list(ref['source'])
    assert [e.index for e in est] == list(range(1, len(lat)))
    s = json.loads((out / 'summary.json').read_text())
    assert s['processed'] == mask.sum() and s['frame_drops'] == len(lat) - mask.sum()


def test_csv_converts_to_inc_npz(tmp_path):
    lat = _profiles()['spiky'][:120]
    out = run_loop(tmp_path, lat, '--tracker', 'cv')
    csv_to_inc(out / 'increments.csv', tmp_path / 'x.inc.npz', {'scene': 'synthetic', 'split': 'none'})
    d = load_increments(tmp_path / 'x.inc.npz')
    assert d['rel_increment'].shape == (119, 4, 4) and d['index'][0] == 1 and d['meta']['n_frames'] == 120
    np.testing.assert_allclose(d['timestamp'], np.arange(1, 120) / 30.0)
    assert set(d['source']) <= {'tracked', 'motion_model'} and (d['tracking_state'] == 'OK').all()


def test_depth_thread_latest_wins(tmp_path):
    lat = np.full(150, 12.0)
    out = run_loop(tmp_path, lat, '--tracker', 'identity', '--depth-sim-ms', '20', '--depth-every', '1',
                   '--clock', 'realtime')
    s = json.loads((out / 'summary.json').read_text())
    assert s['depth_requests_done'] + s['depth_requests_dropped'] == s['processed']
    assert s['depth_requests_done'] > 0 and s['ring_waits'] == 0
    assert s['capture_to_increment_ms_p50'] >= 12.0


def test_redwood_validation_sequence(tmp_path):
    from dvslam.data.redwood import load_sequence
    try:
        seq = load_sequence('apartment', 'validation')
    except (FileNotFoundError, ValueError) as e:
        pytest.skip(f'Redwood validation not available: {e}')
    n = 300
    (tmp_path / 'list.txt').write_text('\n'.join(map(str, seq.image_paths[:n])) + '\n')
    lat = np.random.default_rng(3).gamma(3.0, 12.0, n)
    out = run_loop(tmp_path, lat, '--images-list', str(tmp_path / 'list.txt'), '--tracker', 'cv',
                   '--depth-sim-ms', '60', '--depth-every', '3')
    mask = simulate_mailbox(lat)
    np.testing.assert_array_equal(read_processed_mask(out / 'frames.csv'), mask)
    ref = apply_drop_policy(np.repeat(se3_exp(np.array([0.01, 0, 0, 0, 0.002, 0]))[None], n - 1, 0), mask)
    est = read_increments_csv(out / 'increments.csv')
    np.testing.assert_allclose(np.stack([e.rel_increment for e in est]), ref['rel_increment'], atol=1e-9)


@pytest.mark.gpu
def test_cpp_trt_depth_matches_pytorch_fp32(tmp_path):
    """C++ TensorRT stage vs the PyTorch FP32 graph on Redwood frames (plan 5.1 stage parity, 5.2 gates)."""
    import cv2

    from dvslam.data.redwood import load_depth, load_sequence
    from dvslam.export.depth_engine import (OUT_HW, TorchDepthRunner, depth_parity, preprocess,
                                            summarize_parity)
    cands = [Path(os.environ['DVSLAM_DEPTH_ENGINE'])] if 'DVSLAM_DEPTH_ENGINE' in os.environ else [
        REPO / 'runs' / 'p5_depth' / f'dav2s_{s}_fp16_head.engine' for s in ('476x630', '378x504')]
    engine = next((c for c in cands if c.exists()), cands[0])
    info = subprocess.run([str(BIN), 'info'], capture_output=True, text=True).stdout
    if not engine.exists() or '"trt": true' not in info:
        pytest.skip('no engine or binary built without TensorRT')
    seq = load_sequence('bedroom', 'validation')
    idx = np.linspace(0, len(seq) - 1, 30).round().astype(int)
    (tmp_path / 'list.txt').write_text('\n'.join(str(seq.image_paths[i]) for i in idx) + '\n')
    (tmp_path / 'dump').mkdir()
    out = run_loop(tmp_path, np.full(len(idx), 200.0), '--images-list', str(tmp_path / 'list.txt'),
                   '--depth-engine', str(engine), '--depth-every', '1', '--depth-dump', str(tmp_path / 'dump'),
                   '--clock', 'realtime')
    dumps = sorted((out / 'dump').glob('*.bin'))
    assert len(dumps) >= 5
    hw = tuple(int(v) for v in engine.stem.split('_')[1].split('x'))
    ref = TorchDepthRunner(hw)
    per = []
    for d in dumps:
        k = idx[int(d.stem)]
        pred = np.fromfile(d, np.float32).reshape(OUT_HW)
        bgr = cv2.imread(str(seq.image_paths[k]))
        gt = cv2.resize(load_depth(seq.depth_paths[k]), OUT_HW[::-1], interpolation=cv2.INTER_NEAREST)
        per.append(depth_parity(pred, ref.infer(preprocess(bgr, hw)), gt))
    s = summarize_parity(per)
    assert s['pass'], s                         # fp16_head engine: all 5.2 gates incl. b_t <= 0.2 %
