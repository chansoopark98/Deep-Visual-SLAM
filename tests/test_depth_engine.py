"""Depth engine export and parity (plan v3 item 5.2). CPU tests are fast; `gpu` tests need the 5090, Redwood,
the DAv2 weights and the engines from `scripts/export_depth.py export` (TensorRT tests also need the TRT venv)."""
import os
from pathlib import Path

import numpy as np
import pytest

from dvslam.export.depth_engine import OUT_HW, SIZES, depth_parity, preprocess, summarize_parity

REPO = Path(__file__).resolve().parents[1]
ENGINE_DIR = Path(os.environ.get('DVSLAM_DEPTH_ENGINE_DIR', REPO / 'runs' / 'p5_depth'))
WEIGHTS = REPO / 'runs' / 'e1_weights' / 'dav2' / 'depth_anything_v2_metric_hypersim_vits.pth'


def test_parity_metrics_identity_and_scale():
    rng = np.random.default_rng(0)
    ref = rng.uniform(0.5, 8, OUT_HW).astype(np.float32)
    gt = ref * rng.uniform(0.8, 1.2, OUT_HW).astype(np.float32)
    same = depth_parity(ref.copy(), ref, gt)
    assert same['rel_median'] == 0 and same['scale'] == 0 and same['b_diff'] == 0 and same['nan_frac'] == 0
    up = depth_parity(ref * 1.001, ref, gt)
    assert up['scale'] == pytest.approx(1e-3, rel=1e-3) and up['b_diff'] == pytest.approx(1e-3, rel=1e-3)
    assert summarize_parity([same, up])['pass']
    bad = depth_parity(ref * 1.003, ref, gt)                       # 0.3 % scale change fails the b_t gate
    assert not summarize_parity([same, bad])['pass']
    nan = ref.copy()
    nan[0, 0] = np.nan
    assert not summarize_parity([depth_parity(nan, ref, gt)])['pass']


def test_preprocess_matches_training_pipeline():
    depth_data = pytest.importorskip('dvslam.train.depth_data')
    bgr = np.random.default_rng(1).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    np.testing.assert_array_equal(preprocess(bgr, SIZES['476x630'])[0], depth_data.prepare_image(bgr))
    assert preprocess(bgr, SIZES['378x504']).shape == (1, 3, 378, 504)


@pytest.mark.skipif(not WEIGHTS.exists(), reason='DAv2 metric weights missing')
def test_export_net_is_the_original_network_on_cpu():
    import torch

    from dvslam.export.depth_engine import DepthExportNet, load_dav2_metric
    net = load_dav2_metric()
    x = torch.randn(1, 3, 378, 504, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = net(x)
        full = DepthExportNet(net, (378, 504), out_hw=None)(x)
        quarter = DepthExportNet(load_dav2_metric(), (378, 504))(x)
    assert torch.equal(full, ref)                   # constant pos-embed + classic attention change nothing
    assert quarter.shape == (1, *OUT_HW)


@pytest.mark.gpu
@pytest.mark.parametrize('size', list(SIZES))
def test_onnx_graph_is_fixed_shape_and_sdpa_free(size):
    import onnx
    path = ENGINE_DIR / f'dav2s_{size}.onnx'
    if not path.exists():
        pytest.skip(f'{path} missing (run scripts/export_depth.py export)')
    m = onnx.load(str(path))
    ops = {n.op_type for n in m.graph.node}
    assert m.opset_import[0].version == 17
    assert not ops & {'Attention', 'ScaledDotProductAttention', 'If', 'Loop'}
    dims = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    assert dims == [1, 3, *SIZES[size]]
    assert [d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim] == [1, *OUT_HW]


def _redwood_inputs(hw, n=10):
    import cv2

    from dvslam.data.redwood import load_depth, load_sequence
    seq = load_sequence('lobby', 'validation')
    out = []
    for i in np.linspace(0, len(seq) - 1, n).round().astype(int):
        gt = cv2.resize(load_depth(seq.depth_paths[i]), OUT_HW[::-1], interpolation=cv2.INTER_NEAREST)
        out.append((preprocess(cv2.imread(str(seq.image_paths[i])), hw), gt))
    return out


@pytest.mark.gpu
@pytest.mark.parametrize('runtime', ['trt_fp32', 'trt_fp16_head', 'trt_fp16', 'ort_cpu_fp32'])
def test_runtime_matches_pytorch_fp32(runtime):
    """fp32 / fp16_head / ORT pass every 5.2 gate; plain fp16 passes median and p99 but not b_t <= 0.2 % (report)."""
    from dvslam.export.depth_engine import ORTDepthRunner, TorchDepthRunner, TRTDepthEngine
    size = '378x504'
    if runtime.startswith('trt'):
        pytest.importorskip('tensorrt' if _has_trt_meta() else 'tensorrt_bindings')
        path = ENGINE_DIR / f'dav2s_{size}_{runtime[4:]}.engine'
        make = lambda: TRTDepthEngine(path)                         # noqa: E731
    else:
        path = ENGINE_DIR / f'dav2s_{size}.onnx'
        make = lambda: ORTDepthRunner(path, 'cpu')                  # noqa: E731
    if not path.exists():
        pytest.skip(f'{path} missing')
    ref = TorchDepthRunner(SIZES[size])
    items = _redwood_inputs(SIZES[size])
    refs = [ref.infer(x) for x, _ in items]
    runner = make()
    s = summarize_parity([depth_parity(runner.infer(x), r, gt) for (x, gt), r in zip(items, refs)])
    if runtime == 'trt_fp16':
        assert s['nan_frac'] == 0 and s['rel_median_pct'] < 1.0 and s['rel_p99_pct'] < 2.0, s
    else:
        assert s['pass'], s


def _has_trt_meta() -> bool:
    try:
        import tensorrt  # noqa: F401
        return True
    except ImportError:
        return False
