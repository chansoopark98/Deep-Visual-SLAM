# Evaluation on TUM-VI

Metrics, full results, and how to reproduce them. For the pipeline see [PIPELINE.md](PIPELINE.md); for results on
other datasets see [OTHER_DATASETS.md](OTHER_DATASETS.md).

## Metric

Pair-wise relative-pose errors over consecutive frame pairs:

- **Translation step error**: mean over consecutive frame pairs of `|‖t_gt‖ − ‖t_pred‖| / ‖t_gt‖ × 100` (%).
  Translations are in metres; no scale alignment. Target threshold: <= 2.96 %.
- **Rotation error**: mean absolute roll / pitch / yaw error of the relative rotation, in degrees. Target threshold: <= 3.4°.
- **ATE-SE(3)**: RMSE of the absolute trajectory error after SE(3) alignment (reported for reference only).

## Conditions

- Dataset: TUM-VI `room1`–`room6`, cam0 (512_16 export, rectified to a 512 x 512 pinhole image).
- Official set: `room1`–`room4` (training rooms of the depth network) plus `room6` (unseen), 12,447 pairs.
  `room5` is the validation room; it was used to select the configuration.

## Final pipeline

| Metric | Value | Setting |
|---|---|---|
| Translation step error | **2.482 ± 0.010 %** (worst 2.497 %) | official set, 3 depth-training seeds x 3 start frames |
| Mean step-length error | 0.687 mm/frame | same runs |
| Rotation error | **0.100°** | official set, pair-weighted, start frame 0, 3 seeds |

## Per room

Mean of 3 depth-training seeds, start frame 0.

| Room | Translation step error | Rotation error | ATE-SE(3) |
|---|---|---|---|
| room1 | 2.041 % | 0.090° | 0.098 m |
| room2 | 1.712 % | 0.120° | 0.301 m |
| room3 | 2.081 % | 0.121° | 0.125 m |
| room4 | 2.764 % | 0.086° | 0.071 m |
| room5 (validation) | 3.279 % | 0.090° | 0.413 m |
| room6 (unseen) | 3.918 % | 0.084° | 0.174 m |

## Unseen-room study

`room4` is held out from depth training (depth network trained on `room1`–`room3`) and scored as an unseen room.

| Variant | Translation step error on room4 |
|---|---|
| Baseline | 4.203 % |
| + S3 test-time depth adaptation | 3.987 % |
| + S2 learned weights (trained on room1–3) | 3.952–4.013 % |
| S2 + S3 | 3.765–3.839 % |

The rotation error is unchanged across these variants (0.086–0.087°).

## Caveats

- The configuration was selected on `room5`.
- `room6` was scored many times during development, so it is not a pristine test set.
- The unseen-room study covers a single room.
- Runtime is about 137 ms/frame on an RTX 5090 with the GPU photometric path (about 7 Hz), and about 300 ms/frame with the
  numpy path. This is not real time. Depth inference is included in the demo timing.

## Reproducing the evaluation

Run the final-candidate tracker over the official set (`room1`–`room4` from the `train` split, `room6` from `test`).
Set the data paths first (see [DATASETS.md](DATASETS.md)):

```bash
export DVSLAM_TUMVI_ROOT=$PWD/data/tum_vi DVSLAM_CACHE_ROOT=$PWD/runs/cache

python apps/run_sequence.py --odometry klt_window --dataset tumvi --split train test --out runs/eval/final \
    --opt prior=ssl --opt prior_ckpt=weights/dvslam/dvslam_depth_hints_core.pt \
    --opt quality=0.001 --opt lk_win=15 --opt jump_ref_tracked=true --opt photo=all --opt photo_tol=0.01 \
    --opt photo_levels=1 --opt ba_nfev=10 --opt n_features=250 --opt kf_ratio=0.3 --opt prior_per_kf=false \
    --opt g_tau=5 --opt flow_cache=raft_large --opt kf_refine=0 --opt photo_cell=1 --opt photo_grad=0.002 \
    --opt photo_bidir=true --opt photo_backend=torch
```

This writes `runs/eval/final/<scene>.inc.npz`, `<scene>.causal.tum` and `timings.csv`. Add `--scenes room5` to run a
single room, or `--split validation` for `room5`. Score the run:

```bash
python apps/eval_pairs.py --runs runs/eval/final      # translation / rotation errors, PASS/FAIL on the official set
python scripts/exp003_score.py runs/eval/final        # official set pooled by pairs, per room, by GT-step bin, mm/frame
```

`apps/eval_pairs.py` writes `eval_pairs.txt` and `summary.csv` into the run directory.
`scripts/exp003_score.py` accepts several run directories and `--json <file>`.

`scripts/demo_tumvi.sh` uses `scripts/exp003_track_log.py` instead of `apps/run_sequence.py` for one room; it takes
`--scene`, `--out` and the same `--opt` options and additionally logs the KLT tracks that `render_demo.py` draws.

## Demo outputs

`bash scripts/demo_tumvi.sh <room>` writes to `runs/demo/<room>-<depth>-<timestamp>/` (override with `OUT`):

| File | Content |
|---|---|
| `<room>.inc.npz` | Per-frame relative pose increments |
| `<room>.tracks.npz`, `<room>.extra.npz`, `<room>.cpu.json` | Tracks, diagnostics, timing |
| `eval_pairs.txt`, `summary.csv` | Translation step error, rotation error |
| `score.txt`, `score.json` | Translation step error by ground-truth step bin, mean step-length error (mm/frame) |
| `track.log` | Tracker log |
| `<room>_demo.gif` | Camera frame with tracked points, top-down trajectory vs. ground truth |

Expected for `room5` with the released `dvslam_depth_hints_core.pt` on an RTX 5090: translation step error **3.210 %**,
rotation error **0.090°**, about 410 s. (The per-room table above averages three depth-training seeds, hence 3.279 %.)

| Variable | Default | Meaning |
|---|---|---|
| `DEPTH` | `hints_core` | `hints_core` or `core` (`dvslam_depth_<DEPTH>.pt`) |
| `PHOTONET=1` | off | Also use the S2 weights (`dvslam_s2_photonet_v4.pt`). Not benchmarked end to end in this form. |
| `OUT` | `runs/demo/...` | Output directory |
| `GIF_ARGS` | empty | Extra arguments for `scripts/render_demo.py`, e.g. `"--stride 6 --fps 12"` |
| `DVSLAM_THREADS` | `4` | Cap for OMP / BLAS threads |

Rendering a GIF from a tracker run is described in [OTHER_DATASETS.md](OTHER_DATASETS.md#rendering).
