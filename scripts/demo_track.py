"""Run the tracker (KLTWindowOdometry) on one sequence of any dataset and log what scripts/render_demo.py draws.

  python scripts/demo_track.py --dataset euroc --scene MH_01_easy --out runs/demo/euroc --opt prior=ssl ... [--opt k=v]
  python scripts/demo_track.py --dataset tumvi --scene room6 --out runs/demo/tumvi --opt ...

-> <out>/<scene>.inc.npz (apps/run_sequence.py format), <scene>.extra.npz, <scene>.tracks.npz (every KLT observation,
   scripts/exp003_track_log.py format), <scene>.map.npz (map_frame, map_gid, map_xc: every mapped landmark of the
   window at the end of each step, in the camera frame of that frame, metres = BA units · e^-g of that output),
   <scene>.seq.npz (GT poses, timestamps, image paths: everything the renderer needs without the dataset loaders),
   <scene>.cpu.json (wall / CPU seconds).

Camera. TUM-VI runs on its own rectified 512 x 512 pinhole images (K = tumvi.K_OUT), as in every TUM-VI result.
The depth network (prior=ssl) only accepts that camera, so the other datasets are re-projected into it (--camera
tumvi, their default): the dataset's pinhole image (EuRoC: the loader's undistorted cam0 cache; Redwood: the colour
image as gray, no distortion model) is scaled to f = 190 px and centred on the 512 x 512 canvas (INTER_AREA, then a
sub-pixel shift), i.e. a pure change of intrinsics with no rotation, so depths and poses are unchanged. The image then
covers only the middle of the canvas (EuRoC about 312 x 199 px, Redwood about 232 x 174 px) and is downsampled by
2.4-2.8x. Corners are detected and kept only inside the valid region eroded by --erode px (the tracker's
detect / track masks), and the photometric refinement uses a flat vignette equal to that mask with exposure ratio 1
(TUM-VI's own photometric calibration does not exist for these cameras). The canvases are cached under
DVSLAM_CACHE_ROOT/<dataset>/<scene>/tumvi_512_f190. TUM-VI-trained depth on these images is zero-shot.
"""
import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from apps.run_sequence import parse_opts
from dvslam.data import base, registry
from dvslam.odometry.base import Frame, PinholeCamera, save_increments
from scripts.exp003_track_log import LoggedKLTWindow

CANVAS = 512
K_CANVAS = np.array([[190.0, 0.0, 255.5], [0.0, 190.0, 255.5], [0.0, 0.0, 1.0]])   # = dvslam.data.tumvi.K_OUT


def canvas_affine(K_src: np.ndarray, K_dst: np.ndarray = K_CANVAS) -> np.ndarray:
    """(2, 3) A with u_dst = A @ [u_src, v_src, 1]: the same ray under K_dst (pure change of intrinsics, zero skew)."""
    return np.asarray(K_dst, np.float64)[:2] @ np.linalg.inv(np.asarray(K_src, np.float64))


def to_canvas(img: np.ndarray, K_src: np.ndarray, size: int = CANVAS, K_dst: np.ndarray = K_CANVAS) -> np.ndarray:
    """Re-project an image from K_src to K_dst (size x size). Downsampling goes through INTER_AREA first (no
    aliasing), the remaining sub-pixel shift is bilinear; pixels outside the source are 0."""
    A = canvas_affine(K_src, K_dst)
    sx, sy = A[0, 0], A[1, 1]
    if sx < 1 and sy < 1:
        h, w = img.shape[:2]
        wr, hr = max(1, round(w * sx)), max(1, round(h * sy))
        small = cv2.resize(img, (wr, hr), interpolation=cv2.INTER_AREA)
        rx, ry = wr / w, hr / h                       # actual resize factors; u_r = (u_s + 0.5) r - 0.5
        A = A @ np.array([[1 / rx, 0, 0.5 / rx - 0.5], [0, 1 / ry, 0.5 / ry - 0.5], [0, 0, 1]])
        img = small
    return cv2.warpAffine(img, A, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def valid_mask(src_valid: np.ndarray, K_src: np.ndarray, erode: int, size: int = CANVAS) -> np.ndarray:
    """uint8 255 / 0 canvas mask: canvas pixels fully inside the valid source region, eroded by `erode` px."""
    m = to_canvas(src_valid.astype(np.float32), K_src, size) > 0.999
    if erode > 0:
        m = cv2.erode(m.astype(np.uint8), np.ones((2 * erode + 1, 2 * erode + 1), np.uint8)) > 0
    return np.where(m, 255, 0).astype(np.uint8)


class FlatPhotoNorm:
    """PhotoNorm stand-in for cameras without TUM-VI's photometric calibration: vignette = the canvas mask
    (pixels outside it invalid), every exposure 1."""

    def __init__(self, mask: np.ndarray, v_min: float = 0.25, sat_level: int = 250):
        self.V, self.v_min, self.sat_level = (mask > 0).astype(np.float32), v_min, sat_level

    def exposure(self, image_path) -> float:
        return 1.0

    def __call__(self, image_path, gray: np.ndarray, ratio: float) -> tuple[np.ndarray, np.ndarray]:
        from dvslam.ssl.tumvi_data import photometric
        img, ok = photometric(gray, self.V, ratio, self.v_min, self.sat_level)
        return img, ok > 0


class MapLoggedKLT(LoggedKLTWindow):
    """LoggedKLTWindow that also logs the mapped landmarks (camera frame, metres) and optionally restricts corners
    to a canvas mask."""

    def __init__(self, mask: np.ndarray | None = None, **opts):
        super().__init__(**opts)
        self.canvas_mask = mask
        if mask is not None and self.photo != 'off':
            self.photo_norm = FlatPhotoNorm(mask)     # reset() keeps a photo_norm that is already set

    def reset(self, camera: PinholeCamera) -> None:
        super().reset(camera)
        if self.canvas_mask is not None:
            if self.detect_mask is not None:
                raise ValueError('--camera tumvi cannot be combined with wide_fov_deg / pt_subset')
            self.detect_mask = self.track_mask = self.canvas_mask
        self.map_log = []

    def step(self, frame):
        est = super().step(frame)
        m = np.isfinite(self.X[:, 0])
        if m.any():
            g = self.diag['g'][-1] if self.diag['g'] else 0.0
            R, t = self.T_prev[:3, :3], self.T_prev[:3, 3]
            xc = (self.X[m] - t) @ R * np.exp(-g)      # R^T (X - t), BA units -> metres
            self.map_log.append((np.full(int(m.sum()), frame.index, np.int32),
                                 (self.uid[m] + self._gid_base).astype(np.int64), xc.astype(np.float32)))
        return est


def build_canvas(seq, dataset: str, scene: str, erode: int):
    """(canvas paths, mask, display paths, A canvas->display px, display size). Builds the canvas cache once."""
    disp = [Path(p) for p in seq.image_paths]
    K = np.asarray(seq.K, np.float64)
    first = cv2.imread(str(disp[0]), cv2.IMREAD_GRAYSCALE)
    if first is None:
        raise FileNotFoundError(disp[0])
    h, w = first.shape
    src_valid = np.ones((h, w), np.float32)
    if dataset == 'euroc':                             # the undistorted cache is black where the raw image ends
        from dvslam.data import euroc
        mx, my = euroc.rectify_maps(K, euroc.read_camera(
            Path(registry.default_root('euroc')) / 'machine_hall' / scene / 'mav0' / 'cam0' / 'sensor.yaml')[1], (w, h))
        src_valid = cv2.remap(src_valid, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = valid_mask(src_valid, K, erode)
    dst = base.cache_dir(dataset, scene, 'tumvi_512_f190')
    dst.mkdir(parents=True, exist_ok=True)
    meta = {'K_src': K.tolist(), 'K': K_CANVAS.tolist(), 'size': CANVAS, 'n_frames': len(disp)}
    mp = dst / 'meta.json'
    if mp.exists() and json.loads(mp.read_text()) != meta:
        raise ValueError(f'{dst}: canvas cache was built with different parameters; delete it')
    mp.write_text(json.dumps(meta))
    out = []
    threads = cv2.getNumThreads()
    cv2.setNumThreads(2)
    try:
        for p in disp:
            q = dst / f'{p.stem}.png'
            if not q.exists():
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise FileNotFoundError(p)
                tmp = q.with_name(f'{q.name}.{os.getpid()}.tmp.png')
                cv2.imwrite(str(tmp), to_canvas(img, K))
                os.replace(tmp, q)
            out.append(q)
    finally:
        cv2.setNumThreads(threads)
    A_disp = np.linalg.inv(np.r_[canvas_affine(K), [[0, 0, 1]]])[:2]
    return out, mask, disp, A_disp, (w, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=registry.DATASETS, required=True)
    ap.add_argument('--scene', required=True)
    ap.add_argument('--split', default=None, help='default: inferred (TUM-VI / EuRoC), test (Redwood)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--opt', action='append', default=[], help='tracker option k=v (repeatable)')
    ap.add_argument('--camera', choices=['native', 'tumvi'], default=None,
                    help='tumvi: re-project into the TUM-VI 512 x 512 f = 190 camera (default except for TUM-VI)')
    ap.add_argument('--erode', type=int, default=8, help='px the canvas mask is eroded by (--camera tumvi)')
    ap.add_argument('--frames', type=int, default=0, help='only the first N frames (smoke tests; 0 = all)')
    ap.add_argument('--cv-threads', type=int, default=None)
    args = ap.parse_args()
    if args.cv_threads is not None:
        cv2.setNumThreads(args.cv_threads)
    camera = args.camera or ('native' if args.dataset == 'tumvi' else 'tumvi')
    split = args.split or ('test' if args.dataset == 'redwood' else None)
    root = registry.default_root(args.dataset)
    seq = registry.load_sequence(args.dataset, args.scene, split, root, 0.0)
    n = min(len(seq), args.frames) if args.frames > 0 else len(seq)
    if camera == 'tumvi':
        paths, mask, disp, A_disp, disp_size = build_canvas(seq, args.dataset, args.scene, args.erode)
        K, size = K_CANVAS, (CANVAS, CANVAS)
    else:
        paths, mask, disp = list(seq.image_paths), None, list(seq.image_paths)
        A_disp, disp_size = np.eye(3)[:2], (seq.width, seq.height)
        K, size = np.asarray(seq.K, np.float64), (seq.width, seq.height)
    gt_valid = getattr(seq, 'gt_valid', np.ones(n, bool))
    rate = registry.frame_rate(args.dataset)[0]

    args.out.mkdir(parents=True, exist_ok=True)
    opts = parse_opts(args.opt)
    odo = MapLoggedKLT(mask=mask, **opts)
    odo.reset(PinholeCamera(K, *size))
    odo.log = []
    est = []
    r0, w0 = resource.getrusage(resource.RUSAGE_SELF), time.perf_counter()
    for i in range(n):
        e = odo.step(Frame(i, float(seq.timestamps[i]), paths[i]))
        if e is not None:
            est.append(e)
    est += odo.flush()
    r1, wall = resource.getrusage(resource.RUSAGE_SELF), time.perf_counter() - w0
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    meta = {'dataset': args.dataset, 'scene': args.scene, 'split': seq.split, 'odometry': 'klt_window', 'opts': opts,
            'camera': camera, 'data_root': str(root), 'n_frames': n, 'time_offset_s': 0.0, 'frame_rate_hz': rate}
    s = args.scene
    save_increments(args.out / f'{s}.inc.npz', est, meta)
    np.savez(args.out / f'{s}.extra.npz', **odo.finalize())
    f, g, p = (np.concatenate([x[k] for x in odo.log]) for k in range(3))
    np.savez(args.out / f'{s}.tracks.npz', obs_frame=f, obs_gid=g, obs_px=p, K=K)
    f, g, x = (np.concatenate([m[k] for m in odo.map_log]) for k in range(3))
    np.savez_compressed(args.out / f'{s}.map.npz', map_frame=f, map_gid=g, map_xc=x)
    np.savez(args.out / f'{s}.seq.npz', dataset=args.dataset, scene=s, camera=camera, T_wc=np.asarray(seq.T_wc[:n]),
             gt_valid=np.asarray(gt_valid[:n], bool), timestamps=np.asarray(seq.timestamps[:n], np.float64),
             frame_rate_hz=rate, K=K, size=np.array(size), image_paths=np.array([str(q) for q in paths[:n]]),
             display_paths=np.array([str(q) for q in disp[:n]]), A_disp=A_disp, display_size=np.array(disp_size),
             mask=mask if mask is not None else np.zeros((0, 0), np.uint8))
    info = {'wall_s': wall, 'cpu_s': cpu, 'cores': cpu / wall, 'ms_per_frame': 1e3 * wall / n, 'n_frames': n,
            'cv_threads': cv2.getNumThreads(), 'omp': os.environ.get('OMP_NUM_THREADS')}
    (args.out / f'{s}.cpu.json').write_text(json.dumps(info))
    print(s, json.dumps(info))


if __name__ == '__main__':
    main()
