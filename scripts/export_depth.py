"""Plan v3 item 5.2: export DAv2-S metric to ONNX, build TensorRT engines, check parity and latency.

Run in the TensorRT venv (base env torch/numpy, plus tensorrt-cu12 10.13.2.6):
    runs/p5_venv/bin/python scripts/export_depth.py export [--ckpt runs/phase3/.../best.pt] --out runs/p5_depth
    runs/p5_venv/bin/python scripts/export_depth.py eval --out runs/p5_depth
Outputs: <out>/dav2s_<HxW>.onnx, <out>/dav2s_<HxW>_<prec>.engine (+ .json metadata), <out>/summary.json, tables.md.
Parity reference: the exported graph in PyTorch FP32 (same preprocessing, same 1/4 head), on Redwood validation
frames evenly spaced over the 5 scenes. Gates (plan 5.2): median rel < 1 %, p99 rel < 2 %, |b_t change| <= 0.2 %.
"""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

from dvslam.data.redwood import SCENES, load_depth, load_sequence
from dvslam.depth import read_bgr
from dvslam.export.depth_engine import (OUT_HW, SIZES, ORTDepthRunner, TorchDepthRunner, TRTDepthEngine,
                                        build_engine, cuda_event_latency, depth_parity, export_onnx, preprocess,
                                        summarize_parity, wall_latency)


def _paths(out: Path, size: str, prec: str | None = None) -> Path:
    return out / (f'dav2s_{size}.onnx' if prec is None else f'dav2s_{size}_{prec}.engine')


def _nan_probe(engine: Path, net_hw) -> bool:
    eng = TRTDepthEngine(engine)
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (1, 3, *net_hw)).astype(np.float32)
    return bool(np.isnan(eng.infer(x)).any())


def cmd_export(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for size in a.sizes:
        hw = SIZES[size]
        if a.reuse_onnx and _paths(out, size).exists():
            meta = json.loads(_paths(out, size).with_suffix('.json').read_text())
        else:
            meta = export_onnx(_paths(out, size), hw, a.ckpt)
        print('onnx', size, meta['nodes'], 'nodes, ops:', ' '.join(meta['ops']), 'sdpa_free', meta['sdpa_free'])
        for prec in a.precisions:
            m = build_engine(_paths(out, size), _paths(out, size, prec), prec, a.workspace_gb)
            nan = _nan_probe(_paths(out, size, prec), hw)
            print('engine', size, prec, f"{m['build_s']} s {m['size_mb']} MB nan={nan}")
            if prec == 'fp16' and nan and 'bf16' not in a.precisions:
                print('  FP16 gives NaN -> building BF16 fallback')
                build_engine(_paths(out, size), _paths(out, size, 'bf16'), 'bf16', a.workspace_gb)


def _frames(per_scene: int):
    for scene in SCENES:
        seq = load_sequence(scene, 'validation')
        for i in np.linspace(0, len(seq) - 1, per_scene).round().astype(int):
            yield scene, int(seq.frame_ids[i]), seq.image_paths[i], seq.depth_paths[i]


def _retry_oom(make, tries: int = 40, wait_s: float = 30.0):
    """The 5090 is shared and often nearly full: retry GPU allocations that fail with out-of-memory."""
    import time
    for k in range(tries):
        try:
            return make()
        except (torch.OutOfMemoryError, RuntimeError) as e:
            if 'out of memory' not in str(e).lower() or k == tries - 1:
                raise
            torch.cuda.empty_cache()
            time.sleep(wait_s)


def _gpu_util() -> str:
    try:
        return subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return 'n/a'


def cmd_eval(a):
    out = Path(a.out)
    frames = list(_frames(a.per_scene))
    images = [read_bgr(p) for _, _, p, _ in frames]
    gts = [cv2.resize(load_depth(p), OUT_HW[::-1], interpolation=cv2.INTER_NEAREST) for _, _, _, p in frames]
    summary = {'frames': len(frames), 'per_scene': a.per_scene, 'device': torch.cuda.get_device_name(),
               'accuracy': {}, 'latency': {}}
    if (out / 'summary.json').exists():          # merge: sizes can be evaluated in separate runs
        old = json.loads((out / 'summary.json').read_text())
        summary['accuracy'].update(old.get('accuracy', {}))
        summary['latency'].update(old.get('latency', {}))
    for size in a.sizes:
        hw = SIZES[size]
        xs = [preprocess(im, hw) for im in images]
        ref_runner = _retry_oom(lambda: TorchDepthRunner(hw, a.ckpt))
        refs = [ref_runner.infer(x) for x in xs]
        xt = torch.from_numpy(xs[0]).cuda()
        lat = {'gpu_util_before': _gpu_util()}
        lat['torch_fp32'] = cuda_event_latency(lambda: ref_runner.forward_t(xt), torch.cuda.Stream(), a.warmup,
                                               a.iters)
        del ref_runner
        torch.cuda.empty_cache()

        # One candidate runner alive at a time (the shared 5090 is close to full).
        makers = {f'trt_{p}': (lambda p=p: TRTDepthEngine(_paths(out, size, p))) for p in a.precisions
                  if _paths(out, size, p).exists()}
        makers['ort_cuda_fp32'] = lambda: ORTDepthRunner(_paths(out, size), 'cuda')
        makers['ort_cpu_fp32'] = lambda: ORTDepthRunner(_paths(out, size), 'cpu')
        makers['torch_fp16_autocast'] = lambda: TorchDepthRunner(hw, a.ckpt, fp16=True)
        acc, errors = {}, {}
        for name, make in makers.items():
            try:
                runner = _retry_oom(make)
            except Exception as e:                       # noqa: BLE001  (recorded in the summary)
                errors[name] = f'{type(e).__name__}: {str(e)[:300]}'
                continue
            per = [depth_parity(runner.infer(x), r, g) for x, r, g in zip(xs, refs, gts)]
            acc[name] = summarize_parity(per)
            # Latency: GPU time per call with CUDA events (TRT enqueue / torch forward on a device tensor).
            if name.startswith('trt_'):
                runner.buffers['image'].copy_(xt)
                lat[name] = cuda_event_latency(runner.enqueue, runner.stream, a.warmup, a.iters)

                def e2e(r=runner):          # pageable H2D of the 1x3xHxW input + engine + D2H of the 1/4 output
                    r.buffers['image'].copy_(torch.from_numpy(xs[0]).cuda(non_blocking=True))
                    r.enqueue()
                    r.buffers['depth'].cpu()
                lat[name + '_with_copies'] = cuda_event_latency(e2e, runner.stream, a.warmup, a.iters)
            elif name.startswith('torch'):
                lat[name] = cuda_event_latency(lambda r=runner: r.forward_t(xt), torch.cuda.Stream(), a.warmup,
                                               a.iters)
            else:  # onnxruntime owns its stream: host wall time incl. numpy I/O
                lat[name + '_wall'] = wall_latency(lambda r=runner: r.infer(xs[0]), 3, 20)
            del runner
            torch.cuda.empty_cache()
        # Model change of the 1/4 head (not engine numerics): standard full-resolution head, area-downsampled.
        full = _retry_oom(lambda: TorchDepthRunner(hw, a.ckpt, out_hw=None))
        fulls = [cv2.resize(full.infer(x), OUT_HW[::-1], interpolation=cv2.INTER_AREA) for x in xs]
        per = [depth_parity(f, r, g) for f, r, g in zip(fulls, refs, gts)]
        acc['head_full_vs_quarter (model change)'] = summarize_parity(per)
        acc['full_head_absrel_vs_gt_pct'] = float(np.mean(
            [np.mean(np.abs(f[v] - g[v]) / g[v]) for f, g in zip(fulls, gts) for v in [(g > 0.1) & (g < 10)]]) * 100)
        del full
        torch.cuda.empty_cache()
        abs_rel = []
        for r, g in zip(refs, gts):
            v = (g > 0.1) & (g < 10)
            abs_rel.append(np.mean(np.abs(r[v] - g[v]) / g[v]))
        acc['ref_fp32_absrel_vs_gt_pct'] = float(np.mean(abs_rel) * 100)
        acc['errors'] = errors
        summary['accuracy'][size] = acc
        for k, v in acc.items():
            print(size, k, v if not isinstance(v, dict) else
                  {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()})
        lat['gpu_util_after'] = _gpu_util()
        summary['latency'][size] = lat
        for k, v in lat.items():
            print(size, k, v if isinstance(v, str) else f"p50 {v['p50_ms']:.2f} p95 {v['p95_ms']:.2f}")
        (out / 'summary.json').write_text(json.dumps(summary, indent=1))

    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    lines = ['| size | runtime | median rel % | p99 rel % | worst-frame p99 % | max abs scale % | max abs b_t change % '
             '| NaN | pass |', '|---|---|---|---|---|---|---|---|---|']
    for size, acc in summary['accuracy'].items():
        for name, s in acc.items():
            if isinstance(s, dict) and 'rel_median_pct' in s:
                lines.append(f"| {size} | {name} | {s['rel_median_pct']:.3f} | {s['rel_p99_pct']:.3f} | "
                             f"{s['rel_p99_worst_frame_pct']:.3f} | {s['scale_max_abs_pct']:.3f} | "
                             f"{s['b_diff_max_abs_pct']:.3f} | {s['nan_frac']:.0e} | {s['pass']} |")
    lines += ['', '| size | runtime | p50 ms | p95 ms | mean ms |', '|---|---|---|---|---|']
    for size, lat in summary['latency'].items():
        for name, v in lat.items():
            if isinstance(v, dict):
                lines.append(f"| {size} | {name} | {v['p50_ms']:.2f} | {v['p95_ms']:.2f} | {v['mean_ms']:.2f} |")
    (out / 'tables.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmd', choices=['export', 'eval', 'all'])
    p.add_argument('--ckpt', default=None, help='Phase-3 checkpoint ({"model": sd} or raw sd); default Hypersim')
    p.add_argument('--out', default='runs/p5_depth')
    p.add_argument('--sizes', nargs='+', default=list(SIZES), choices=list(SIZES))
    p.add_argument('--precisions', nargs='+', default=['fp16', 'fp16_head', 'bf16', 'fp32'], choices=['fp16', 'fp16_head', 'bf16', 'fp32'])
    p.add_argument('--per-scene', type=int, default=40)
    p.add_argument('--workspace-gb', type=float, default=4.0, help='TensorRT builder workspace limit')
    p.add_argument('--reuse-onnx', action='store_true', help='skip the ONNX export if the file exists')
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--iters', type=int, default=300)
    a = p.parse_args()
    if a.cmd in ('export', 'all'):
        cmd_export(a)
    if a.cmd in ('eval', 'all'):
        cmd_eval(a)


if __name__ == '__main__':
    main()
