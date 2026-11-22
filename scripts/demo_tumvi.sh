#!/usr/bin/env bash
# End-to-end demo on one TUM-VI room with the released weights: final-candidate tracker (EXP-020 options) with the
# live SSL depth prior, translation step / rotation errors, and an animated GIF.
#
#   scripts/demo_tumvi.sh [ROOM]          # ROOM = room1..room6, default room5 (validation room)
#
# Needs: a CUDA GPU (torchvision RAFT-large flow every frame + SSL depth net), the TUM-VI 512_16 sequence
# <DVSLAM_TUMVI_ROOT>/dataset-<ROOM>_512_16 (scripts/fetch_datasets.sh tumvi <ROOM>), the weights
# (scripts/download_weights.sh). RAFT-large weights are fetched by torchvision on first use (~/.cache/torch).
#
# Environment variables
#   DVSLAM_TUMVI_ROOT  TUM-VI root (default: <repo>/data/tum_vi)
#   DVSLAM_CACHE_ROOT  rectified-image cache, built on the first run of a room (~0.5 GB; default: <repo>/runs/cache)
#   DVSLAM_WEIGHTS     weights directory (default: <repo>/weights/dvslam)
#   DEPTH              hints_core (default; EXP-020 'hints') | core (EXP-020 'core')
#   PHOTONET=1         also use the S2 learned photometric weights (dvslam_s2_photonet_v4.pt; EXP-028, room5 only)
#   OUT                output directory (default: <repo>/runs/demo/<ROOM>-<DEPTH>-<date>)
#   PYTHON             interpreter (default: python); DVSLAM_THREADS (default 4) caps OMP / BLAS threads
#   GIF_ARGS           extra arguments for scripts/render_demo.py (e.g. "--stride 6 --fps 12")
# The rectified cache is built by the TUM-VI loader on first use (dvslam/data/tumvi.py, cv2 fisheye remap).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
ROOM=${1:-room5}
DEPTH=${DEPTH:-hints_core}
PY=${PYTHON:-python}
N=${DVSLAM_THREADS:-4}
export OMP_NUM_THREADS=$N MKL_NUM_THREADS=$N OPENBLAS_NUM_THREADS=$N NUMEXPR_NUM_THREADS=$N
export DVSLAM_TUMVI_ROOT=${DVSLAM_TUMVI_ROOT:-$REPO/data/tum_vi}
export DVSLAM_CACHE_ROOT=${DVSLAM_CACHE_ROOT:-$REPO/runs/cache}
W=${DVSLAM_WEIGHTS:-$REPO/weights/dvslam}
OUT=${OUT:-$REPO/runs/demo/$ROOM-$DEPTH-$(date +%Y%m%d-%H%M%S)}

case $ROOM in room[1-6]) ;; *) echo "ROOM must be room1..room6, got $ROOM"; exit 1 ;; esac
case $DEPTH in hints_core|core) ;; *) echo "DEPTH must be hints_core or core, got $DEPTH"; exit 1 ;; esac
CKPT=$W/dvslam_depth_$DEPTH.pt
[ -f "$CKPT" ] || { echo "missing $CKPT: run scripts/download_weights.sh (or set DVSLAM_WEIGHTS)"; exit 1; }
[ -d "$DVSLAM_TUMVI_ROOT/dataset-${ROOM}_512_16" ] || {
  echo "missing $DVSLAM_TUMVI_ROOT/dataset-${ROOM}_512_16: set DVSLAM_TUMVI_ROOT or run scripts/fetch_datasets.sh tumvi $ROOM"
  exit 1; }
"$PY" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' \
  || { echo "no CUDA GPU visible to torch: the demo needs one (RAFT-large + SSL depth)"; exit 1; }

# final candidate (EXP-020): runs/eval/exp009b_A.txt + RAFT-large tracks + dense bidirectional photometric refinement;
# photo_backend=torch = the GPU port of the same refinement (S1, EXP-024: equal to numpy within 1e-9 pp)
OPTS=(prior=ssl "prior_ckpt=$CKPT"
      quality=0.001 lk_win=15 jump_ref_tracked=true photo=all photo_cell=4 photo_tol=0.01 photo_levels=1 ba_nfev=10
      n_features=250 kf_ratio=0.3 prior_per_kf=false g_tau=5
      flow_cache=raft_large kf_refine=0 photo_cell=1 photo_grad=0.002 photo_bidir=true photo_backend=torch)  # last wins
if [ "${PHOTONET:-0}" = 1 ]; then
  [ -f "$W/dvslam_s2_photonet_v4.pt" ] || { echo "missing $W/dvslam_s2_photonet_v4.pt"; exit 1; }
  OPTS+=("photo_feat=$W/dvslam_s2_photonet_v4.pt")
fi
ARGS=()
for o in "${OPTS[@]}"; do ARGS+=(--opt "$o"); done

mkdir -p "$OUT"
echo "== $ROOM, depth $DEPTH ($CKPT) -> $OUT"
T0=$SECONDS
nice -n 10 "$PY" scripts/exp003_track_log.py --scene "$ROOM" --out "$OUT" "${ARGS[@]}" 2>&1 | tee "$OUT/track.log"
T1=$SECONDS
echo "== score (translation step error / rotation error)"
nice -n 10 "$PY" apps/eval_pairs.py --runs "$OUT"    # writes eval_pairs.txt, summary.csv
nice -n 10 "$PY" scripts/exp003_score.py "$OUT" --json "$OUT/score.json" | tee "$OUT/score.txt"
echo "== render"
# shellcheck disable=SC2086
nice -n 10 "$PY" scripts/render_demo.py --run "$OUT" --scene "$ROOM" --out "$OUT/${ROOM}_demo.gif" ${GIF_ARGS:-}
echo
echo "tracker: $((T1 - T0)) s, total: $((SECONDS - T0)) s"
echo "outputs in $OUT:"
echo "  $ROOM.inc.npz (increments), $ROOM.tracks.npz (KLT tracks), $ROOM.extra.npz, $ROOM.cpu.json (timing)"
echo "  eval_pairs.txt (translation / rotation errors), score.txt / score.json (translation error by GT-step bin, mm/frame), track.log,"
echo "  ${ROOM}_demo.gif"
