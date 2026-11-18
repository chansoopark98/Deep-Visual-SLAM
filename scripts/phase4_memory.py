"""Plan v3 item 4.2: peak memory of tracker C over full validation scenes, optionally under an address-space cap.

  python scripts/phase4_memory.py --out runs/p4-memory --limit-gb 6          # RLIMIT_AS = 6 GB for the whole process
  systemd-run --user --scope -p MemoryMax=6G python scripts/phase4_memory.py --out runs/p4-memory-cg --limit-gb 0

Writes <out>/memory.json (VmPeak, VmHWM, GPU use, parity against the recorded run) and <out>/timings.csv (per-frame
total_ms: a re-measurement of C's speed on whatever load the host has now).
"""
import resource
import sys

if '--limit-gb' in sys.argv:          # before numpy / cv2 are imported, so the cap covers the whole process
    _gb = float(sys.argv[sys.argv.index('--limit-gb') + 1])
    if _gb > 0:
        resource.setrlimit(resource.RLIMIT_AS, (int(_gb * 2 ** 30),) * 2)

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.phase4_common import SCENES, load_val, make_c  # noqa: E402
from dvslam.odometry.base import Frame, PinholeCamera, load_increments  # noqa: E402


def proc_status() -> dict:
    out = {}
    for line in open('/proc/self/status'):
        k, _, v = line.partition(':')
        if k in ('VmPeak', 'VmHWM', 'VmRSS', 'VmSize'):
            out[k] = int(v.split()[0]) / 2 ** 20          # kB -> GiB
    return out


def cgroup_mem() -> dict:
    """memory.max / memory.peak (GiB) of this process's cgroup, when readable (systemd-run --scope)."""
    try:
        c = open('/proc/self/cgroup').read().strip().split(':')[-1]
        d = Path('/sys/fs/cgroup') / c.lstrip('/')
        return {k: (None if (v := (d / k).read_text().strip()) == 'max' else int(v) / 2 ** 30)
                for k in ('memory.max', 'memory.peak')}
    except (OSError, ValueError):
        return {}


def gpu_mem_mib() -> float | None:
    try:
        r = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    import os
    return sum(float(l.split(',')[1]) for l in r.splitlines() if l.split(',')[0].strip() == str(os.getpid()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limit-gb', type=float, default=0.0, help='RLIMIT_AS in GiB (0 = none); applied at start-up')
    ap.add_argument('--scenes', nargs='+', default=list(SCENES))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ref', type=Path, default=Path('runs/p1c-base07-s0'), help='recorded run for the parity check')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    odo = make_c(args.seed)
    res = {'limit_gb': args.limit_gb, 'rlimit_as': list(resource.getrlimit(resource.RLIMIT_AS)), 'scenes': {}}
    peak_gpu = gpu_mem_mib()
    with open(args.out / 'timings.csv', 'w', newline='') as tf:
        w = csv.writer(tf)
        w.writerow(['scene', 'index', 'total_ms'])
        for scene in args.scenes:
            seq = load_val(scene)
            odo.reset(PinholeCamera(seq.K, seq.width, seq.height))
            inc, ms, t0 = [], [], time.time()
            for i in range(len(seq)):
                est = odo.step(Frame(i, i / 30, seq.image_paths[i], seq.depth_paths[i]))
                if est is not None:
                    inc.append(est.rel_increment)
                    ms.append(est.timings_ms['total'])
                    w.writerow([scene, i, f'{ms[-1]:.3f}'])
            inc = np.stack(inc)
            ref = load_increments(args.ref / f'{scene}.inc.npz')['rel_increment'] if (args.ref / f'{scene}.inc.npz').exists() else None
            g = gpu_mem_mib()
            peak_gpu = max(x for x in (peak_gpu, g) if x is not None) if g is not None else peak_gpu
            res['scenes'][scene] = {
                'frames': len(seq), 'wall_s': time.time() - t0, 'ms_median': float(np.median(ms)),
                'ms_p95': float(np.percentile(ms, 95)), 'ms_p99': float(np.percentile(ms, 99)),
                'max_abs_diff_vs_recorded': None if ref is None else float(np.abs(inc - ref).max()),
                **{k: round(v, 3) for k, v in proc_status().items()}}
            print(scene, res['scenes'][scene], flush=True)
    res['proc'] = {k: round(v, 3) for k, v in proc_status().items()}
    res['gpu_mem_mib_peak'] = peak_gpu
    res['cgroup_gib'] = cgroup_mem()
    res['torch_imported'] = 'torch' in sys.modules
    res['completed'] = True
    (args.out / 'memory.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ('proc', 'cgroup_gib', 'gpu_mem_mib_peak', 'torch_imported', 'completed')}))


if __name__ == '__main__':
    main()
