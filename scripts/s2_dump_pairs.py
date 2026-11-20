"""Plan v3.4 §14.4 S2: run tracker C (scripts/exp003_track_log.py, same arguments and outputs) and also dump, per
photometrically refined pair with photo_bidir, what the refinement got and returned, so apps/train_photo_s2.py can
replay the refinement offline with exactly the tracker's inputs. Numerics of the run are untouched.

  python scripts/s2_dump_pairs.py --scene room1 --out DIR --opt ... (exp003_track_log.py arguments)
-> DIR/<scene>.s2pairs.npz, one row per pair whose two directions both ran their LM:
   index / prev   frame t / t-1
   inc            (4, 4) BA-unit increment the refinement started from (X_{t-1} = inc X_t)
   g, s           log map scale (metric = map · e^{-g}) and the map-unit ratio _photo_s of both directions' depths
   T_f, T_b       refined increment of the forward (t-1 -> t, returned as inc form) / backward direction
   ok_f, ok_b     each direction's acceptance; out = the increment the tracker kept (NaN when none was accepted)
   cost0_* / cost_* / n_*   LM info of each direction
   opt_<name>     scalar: the run's refinement options OPT_KEYS (photo_levels, photo_max_pts, depth limits, LM
                  iters / huber / tol, photo_bidir, point selection), which apps/train_photo_s2.py checks
Inputs not stored are recomputed from the images and the depth cache (the tracker's own point selection).
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts import exp003_track_log as mod

KW = mod.KLTWindowOdometry
_refine0, _ok0 = KW._photo_refine, KW._photo_ok
REC, CUR, OPT = [], {}, {}
OPT_KEYS = ('photo_levels', 'photo_max_pts', 'depth_min', 'depth_max', 'depth_edge', 'photo_iters', 'photo_huber',
            'photo_tol', 'photo_bidir', 'photo_cell', 'photo_grad')


def _ok(self, info, T, inc):
    res = _ok0(self, info, T, inc)
    CUR.setdefault('dirs', []).append((np.array(T, float), info, res))
    return res


def _refine(self, frame, gray, inc, head_prev):
    CUR.clear()
    OPT.update({f'opt_{k}': getattr(self, k) for k in OPT_KEYS})
    prev, g = self._prev_frame.index, self.g
    out = _refine0(self, frame, gray, inc, head_prev)
    d = CUR.get('dirs', [])
    if len(d) == 2 and d[1][1] is not None:
        (Tf, i_f, ok_f), (Tb, i_b, ok_b) = d
        REC.append({'index': frame.index, 'prev': prev, 'inc': np.array(inc, float), 'g': g, 's': self._photo_s,
                    'T_f': Tf, 'T_b': Tb, 'ok_f': ok_f, 'ok_b': ok_b,
                    'out': np.full((4, 4), np.nan) if out is None else np.array(out, float),
                    **{f'{k}_{n}': float(i[k]) for n, i in (('f', i_f), ('b', i_b)) for k in ('cost0', 'cost', 'n')}})
    return out


def main():
    KW._photo_refine, KW._photo_ok = _refine, _ok
    mod.main()
    argv = sys.argv[1:]
    out, scene = Path(argv[argv.index('--out') + 1]), argv[argv.index('--scene') + 1]
    if not REC:
        raise SystemExit(f's2pairs {scene}: no pair ran both refinement directions (needs photo=all|fallback and '
                         'photo_bidir=true); nothing written')
    np.savez(out / f'{scene}.s2pairs.npz', **{k: np.array([r[k] for r in REC]) for k in REC[0]},
             **{k: np.array(v) for k, v in OPT.items()})
    print(f's2pairs {scene}: {len(REC)} pairs')


if __name__ == '__main__':
    main()
