"""Plan v3 item 0.9 on real data: is the Redwood GT consistent with the sensor streams at sub-frame level?

For each scene, find the fractional time offset τ (in frames) that best aligns the GT per-pair rotation vectors
with (a) image-based trackers (KLT depth-free, DPVO) and (b) optionally depth-only point-to-plane ICP.
Rotation is scale-free, so no scale fitting is involved. Also reports the item-03 MAPE between the GT steps and
the GT steps shifted by τ (what a perfect estimator at the sensor's time base would score), and the stride/lag
GT-noise probe of dvslam.eval.gt_noise_probe on the two depth-free trackers.

  python scripts/gt_timing_probe.py --klt runs/klt-df-val --dpvo runs/dpvo-val-M96-s0 --split validation [--icp 600]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dvslam.data import redwood  # noqa: E402
from dvslam.eval.diagnostics import _pct_errors, best_rescale  # noqa: E402
from dvslam.eval.gt_noise_probe import estimate_sigma_g, probe_scene  # noqa: E402
from dvslam.geometry.se3 import relative_pose  # noqa: E402
from dvslam.odometry.base import load_increments  # noqa: E402

TAUS = np.round(np.linspace(-1, 1, 41), 2)


def rotvec(inc):
    return Rotation.from_matrix(inc[:, :3, :3]).as_rotvec()


def shift(X, tau):
    """Per-pair series X evaluated at fractional pair index i + tau (linear interpolation)."""
    i = np.arange(len(X))
    return np.stack([np.interp(i + tau, i, X[:, c]) for c in range(X.shape[1])], 1)


def best_tau(G, X, mask):
    cs = [np.mean([np.corrcoef(shift(G, t)[mask, c], X[mask, c])[0, 1] for c in range(3)]) for t in TAUS]
    k = int(np.argmax(cs))
    return float(TAUS[k]), float(cs[len(TAUS) // 2]), float(cs[k])


def icp_increments(seq, n):
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    K = seq.K
    intr = o3d.camera.PinholeCameraIntrinsic(seq.width, seq.height, K[0, 0], K[1, 1], K[0, 2], K[1, 2])

    def pcd(i):
        d = o3d.geometry.Image(redwood.load_depth(seq.depth_paths[i]))
        p = o3d.geometry.PointCloud.create_from_depth_image(d, intr, depth_scale=1.0, depth_trunc=4.0)
        p = p.voxel_down_sample(0.01)
        p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        return p
    P = [pcd(i) for i in range(n)]
    est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    return np.stack([o3d.pipelines.registration.registration_icp(P[t], P[t - 1], 0.03, np.eye(4), est).transformation
                     for t in range(1, n)])


def window_scaled_world_steps(inc, gt_inc, T_wc, window=30):
    """Depth-free increments rescaled per window to GT (oracle), rotated into the world with GT orientation."""
    t = inc[:, :3, 3].copy()
    for s in range(0, len(t), window):
        sc, _ = best_rescale(gt_inc[s:s + window, :3, 3], t[s:s + window])
        t[s:s + window] *= sc
    return np.einsum('nij,nj->ni', T_wc[:-1, :3, :3], t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--klt', type=Path, required=True)
    ap.add_argument('--dpvo', type=Path, required=True)
    ap.add_argument('--split', default='validation')
    ap.add_argument('--icp', type=int, default=0, help='frames per scene for depth-only ICP (0 = skip; needs open3d)')
    ap.add_argument('--out', type=Path, default=REPO_ROOT / 'reports/phase0/09_gt_timing_and_noise.md')
    args = ap.parse_args()

    rows, probes = [], []
    for sc in redwood.SCENES:
        seq = redwood.load_sequence(sc, args.split)
        g = np.stack([relative_pose(seq.T_wc[t - 1], seq.T_wc[t]) for t in range(1, len(seq))])
        k, d = load_increments(args.klt / f'{sc}.inc.npz'), load_increments(args.dpvo / f'{sc}.inc.npz')
        ok = (k['source'] == 'tracked') & (d['source'] == 'tracked')
        G = rotvec(g)
        tk, ck0, ckm = best_tau(G, rotvec(k['rel_increment']), ok)
        td, cd0, cdm = best_tau(G, rotvec(d['rel_increment']), ok)
        floor = float(_pct_errors(g[:, :3, 3], shift(g[:, :3, 3], td)).mean())
        row = {'scene': sc, 'tau_klt': tk, 'corr_klt': (ck0, ckm), 'tau_dpvo': td, 'corr_dpvo': (cd0, cdm),
               'floor': floor}
        if args.icp:
            n = min(args.icp, len(seq))
            ti, ci0, cim = best_tau(G[:n - 1], rotvec(icp_increments(seq, n)), np.ones(n - 1, bool))
            row |= {'tau_icp': ti, 'corr_icp': (ci0, cim)}
        rows.append(row)
        sa = window_scaled_world_steps(k['rel_increment'], g, seq.T_wc)
        sb = window_scaled_world_steps(d['rel_increment'], g, seq.T_wc)
        probes.append(probe_scene(sa[ok], sb[ok], seq.T_wc[:, :3, 3][np.r_[True, ok]]))
        print(row)
    est = estimate_sigma_g(probes)

    cmd = f'--klt {args.klt} --dpvo {args.dpvo} --split {args.split}' + (f' --icp {args.icp}' if args.icp else '')
    L = [f'# Item 0.9 on real data: GT timing and noise ({args.split})', '',
         f'Generated by `scripts/gt_timing_probe.py {cmd}`.', '',
         'τ = fractional offset (frames) at which the GT per-pair rotation best matches the estimator '
         '(positive: the estimator at pair i matches GT at i + τ). r = mean per-axis Pearson r of rotation vectors '
         'at τ = 0 → at τ*. Floor = item-03 MAPE of GT steps vs GT steps shifted by τ*(DPVO): what a perfect '
         'estimator on the image time base would score against this GT.', '',
         '| scene | τ* KLT | r KLT | τ* DPVO | r DPVO |' + (' τ* depth ICP | r ICP |' if args.icp else '')
         + ' floor % |', '|---|---|---|---|---|' + ('---|---|' if args.icp else '') + '---|']
    for r in rows:
        L.append(f'| {r["scene"]} | {r["tau_klt"]:+.2f} | {r["corr_klt"][0]:.3f}→{r["corr_klt"][1]:.3f} | '
                 f'{r["tau_dpvo"]:+.2f} | {r["corr_dpvo"][0]:.3f}→{r["corr_dpvo"][1]:.3f} |'
                 + (f' {r["tau_icp"]:+.2f} | {r["corr_icp"][0]:.3f}→{r["corr_icp"][1]:.3f} |' if args.icp else '')
                 + f' {r["floor"]:.1f} |')
    L += ['', f'Mean timing floor (mean of scene means): **{np.mean([r["floor"] for r in rows]):.1f} %**.', '',
          '## GT-noise probe on the two depth-free trackers (window-rescaled to GT, world frame)', '',
          f'σ_g estimates (mm/axis): intercept {est["intercept"]*1e3:.3f}, lag-1 {est["lag1"]*1e3:.3f}, '
          f'k=1 {est["k1"]*1e3:.3f}; C by stride (mm²) {[round(c*1e6, 4) for c in est["C_by_stride"]]}.', '',
          'Caveat: the synthetic validation (`gt_noise_probe_validation.md`) assumed white, time-aligned GT noise. '
          'A timing offset is neither, so these σ_g values mix timing error and noise and are an upper-bound reading.', '']
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text('\n'.join(L))
    print(args.out)


if __name__ == '__main__':
    main()
