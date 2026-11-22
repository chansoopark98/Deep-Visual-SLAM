# Deep-Visual-SLAM-Monodepth2

[![Github All Releases](https://img.shields.io/github/downloads/chansoopark98/Deep-Visual-SLAM-Monodepth2/total.svg)]()

<p align="center">
 <img src="https://img.shields.io/github/issues/chansoopark98/Deep-Visual-SLAM-Monodepth2">
 <img src="https://img.shields.io/github/forks/chansoopark98/Deep-Visual-SLAM-Monodepth2">
 <img src="https://img.shields.io/github/stars/chansoopark98/Deep-Visual-SLAM-Monodepth2">
 <img src="https://img.shields.io/github/license/chansoopark98/Deep-Visual-SLAM-Monodepth2">
</p>

<p align="center">
 <img alt="Python" src ="https://img.shields.io/badge/Python-3776AB.svg?&style=for-the-badge&logo=Python&logoColor=white"/>
 <img src ="https://img.shields.io/badge/PyTorch-EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white"/>
 <img src ="https://img.shields.io/badge/OpenCV-5C3EE8.svg?&style=for-the-badge&logo=OpenCV&logoColor=white"/>
 <img src ="https://img.shields.io/badge/Numpy-013243.svg?&style=for-the-badge&logo=Numpy&logoColor=white"/>
 <img src ="https://img.shields.io/badge/CUDA-76B900.svg?&style=for-the-badge&logo=NVIDIA&logoColor=white"/>
</p>

<p align="center">
  <img src="media/tumvi_room6_demo.gif" alt="TUM-VI room6 demo: input frame with tracked features, 3D map with keyframes and current camera, top-down trajectory against ground truth" width="900">
</p>

<p align="center"><sub>TUM-VI <code>room6</code>, a room not used for training or tuning. Left: input frame with tracked features. Middle: map points, keyframes (blue) and current camera (green). Right: top-down trajectory, estimate vs. ground truth, metric, no alignment.</sub></p>

### Results on other datasets

The same pipeline, run without retraining on EuRoC and Redwood. Each sequence has an animated GIF and a static geometry
figure (`media/*_geometry.png`: evo-style 2D trajectory and the final 3D map).

<table>
  <tr>
    <td align="center"><img src="media/euroc_MH_01_easy_demo.gif" width="420" alt="EuRoC MH_01_easy demo"><br><sub>EuRoC <code>MH_01_easy</code>: the track diverges and is mis-scaled.</sub></td>
    <td align="center"><img src="media/euroc_MH_01_easy_geometry.png" width="420" alt="EuRoC MH_01_easy geometry: top-down trajectory and final 3D map"><br><sub>Geometry, <code>MH_01_easy</code>.</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="media/euroc_MH_03_medium_demo.gif" width="420" alt="EuRoC MH_03_medium demo"><br><sub>EuRoC <code>MH_03_medium</code>: the track diverges and is mis-scaled.</sub></td>
    <td align="center"><img src="media/euroc_MH_03_medium_geometry.png" width="420" alt="EuRoC MH_03_medium geometry: top-down trajectory and final 3D map"><br><sub>Geometry, <code>MH_03_medium</code>.</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="media/redwood_bedroom_demo.gif" width="420" alt="Redwood bedroom demo"><br><sub>Redwood <code>bedroom</code>: mis-scaled; the reference poses are inaccurate.</sub></td>
    <td align="center"><img src="media/redwood_bedroom_geometry.png" width="420" alt="Redwood bedroom geometry: top-down trajectory and final 3D map"><br><sub>Geometry, <code>bedroom</code>.</sub></td>
  </tr>
</table>

Pairs t >= 1 with ground truth at both ends, start frame 0, final configuration, live depth from
`dvslam_depth_hints_core.pt`. ATE is the RMSE after SE(3) alignment; the other two metrics use no alignment
(see [Metric](#metric)).

| Sequence | Frames | Translation step error | Rotation error | ATE |
|---|---|---|---|---|
| TUM-VI room6 (unseen room) | 2617 | 3.98 % | 0.084° | 0.175 m |
| EuRoC MH_01_easy (zero-shot) | 3638 | 178.97 % | 0.188° | 3.06 m |
| EuRoC MH_03_medium (zero-shot) | 2631 | 116.34 % | 0.183° | 3.18 m |
| Redwood bedroom (zero-shot) | 1097 | 64.19 % | 0.120° | 0.31 m |

**Limitations.** The depth network was trained only on TUM-VI `room1`–`room4`. On EuRoC and Redwood the images are
re-projected into its training camera (pinhole, 512 x 512, f = 190) and its metric scale does not transfer: the median
predicted / ground-truth step length is 0.30–0.40 on EuRoC and 1.35 on Redwood. The trajectories are therefore
mis-scaled, and the EuRoC tracks diverge. The Redwood reference poses are known to be inaccurate. The rotation error
stays small on all sequences.

## Overview

`dvslam` is a monocular metric visual odometry system for low-parallax, high-frame-rate indoor sequences. Geometry
carries the pose path: dense optical-flow tracks, PnP and a sliding-window bundle adjustment, refined by a dense
photometric step. A self-supervised depth network provides the metric scale prior. Inference uses a single RGB camera;
stereo images and the gyroscope are used only while training the depth network.

The original Monodepth2-style visual odometry is kept on a separate branch (see [Legacy](#legacy-monodepth2-style-vo)).

Contents: [Other datasets](#results-on-other-datasets) · [Pipeline](#pipeline) · [Results](#results) · [Installation](#installation) · [Data](#data) ·
[Pretrained weights](#pretrained-weights) · [Quick demo](#quick-demo) · [Tests](#tests) ·
[Reproducing the evaluation](#reproducing-the-evaluation) · [Training](#training) · [Repository layout](#repository-layout) ·
[Legacy](#legacy-monodepth2-style-vo) · [References](#references)

## Pipeline

```mermaid
flowchart LR
    A["cam0 frames<br/>512x512 rectified pinhole, 20 Hz"] --> B["RAFT-large<br/>dense-flow tracks"]
    B --> C["PnP + sliding-window BA<br/>depth prior + one global log-scale g"]
    C --> D["Dense bidirectional<br/>photometric refinement<br/>GPU, photo_backend=torch"]
    D --> E["Causal metric<br/>relative poses"]
    F["SSL depth net<br/>self-supervised, metric"] --> C
    G["S2 learned per-pixel weights<br/>photo_feat, optional"] -.-> D
    H["S3 test-time depth adaptation<br/>LoRA, offline, optional"] -.-> F
```

| Stage | What it does | Code |
|---|---|---|
| Tracking and window BA | RAFT-large flow every frame gives point tracks; PnP plus a sliding-window bundle adjustment estimate poses. The depth prior constrains the metric scale through one global log-scale `g`. | `dvslam/odometry/klt_window.py` |
| Photometric refinement | Dense bidirectional photometric alignment refines each relative pose. `photo_backend=torch` runs it on the GPU. | `dvslam/odometry/klt_photo_torch.py` |
| S2 learned weights (optional) | A small U-Net (`PhotoNet`) predicts per-pixel log weights for the refinement (`photo_feat=<file>`). | `dvslam/odometry/photo_learned.py` |
| SSL depth network | ResNet-18 depth and pose networks trained self-supervised on TUM-VI. Stereo images, gyroscope and stereo depth hints are used only for training; inference is cam0 only. | `dvslam/ssl/` |
| S3 test-time adaptation (optional) | One offline pass adapts the depth decoder to a test sequence with a LoRA adapter, using the tracker's own poses as pseudo-labels. No cam1, IMU or ground truth. | `dvslam/ssl/adapter.py`, `apps/calib_s3.py` |

The output is a causal chain of metric relative poses, one per frame.

## Results

### Metric

Pair-wise relative-pose errors over consecutive frame pairs:

- **Translation step error**: mean over consecutive frame pairs of `|‖t_gt‖ − ‖t_pred‖| / ‖t_gt‖ × 100` (%).
  Translations are in metres; no scale alignment. Target threshold: <= 2.96 %.
- **Rotation error**: mean absolute roll / pitch / yaw error of the relative rotation, in degrees. Target threshold: <= 3.4°.

### Conditions

- Dataset: TUM-VI `room1`–`room6`, cam0 (512_16 export, rectified to a 512 x 512 pinhole image).
- Official set: `room1`–`room4` (training rooms of the depth network) plus `room6` (unseen), 12,447 pairs.
  `room5` is the validation room; it was used to select the configuration.

### Final pipeline

| Metric | Value | Setting |
|---|---|---|
| Translation step error | **2.482 ± 0.010 %** (worst 2.497 %) | official set, 3 depth-training seeds x 3 start frames |
| Mean step-length error | 0.687 mm/frame | same runs |
| Rotation error | **0.100°** | official set, pair-weighted, start frame 0, 3 seeds |

### Per room

Mean of 3 depth-training seeds, start frame 0.

| Room | Translation step error | Rotation error | ATE-SE(3) |
|---|---|---|---|
| room1 | 2.041 % | 0.090° | 0.098 m |
| room2 | 1.712 % | 0.120° | 0.301 m |
| room3 | 2.081 % | 0.121° | 0.125 m |
| room4 | 2.764 % | 0.086° | 0.071 m |
| room5 (validation) | 3.279 % | 0.090° | 0.413 m |
| room6 (unseen) | 3.918 % | 0.084° | 0.174 m |

### Unseen-room study

`room4` is held out from depth training (depth network trained on `room1`–`room3`) and scored as an unseen room.

| Variant | Translation step error on room4 |
|---|---|
| Baseline | 4.203 % |
| + S3 test-time depth adaptation | 3.987 % |
| + S2 learned weights (trained on room1–3) | 3.952–4.013 % |
| S2 + S3 | 3.765–3.839 % |

The rotation error is unchanged across these variants (0.086–0.087°).

### Caveats

- The configuration was selected on `room5`.
- `room6` was scored many times during development, so it is not a pristine test set.
- The unseen-room study covers a single room.
- Runtime is about 137 ms/frame on an RTX 5090 with the GPU photometric path (about 7 Hz), and about 300 ms/frame with the
  numpy path. This is not real time. Depth inference is included in the demo timing.

## Installation

Requirements: Linux, Python >= 3.10, an NVIDIA GPU (needed for RAFT-large and the live depth network; the unit tests run
on the CPU). The setup script installs PyTorch from a CUDA 12.8 wheel index, which RTX 50xx GPUs require.

```bash
git clone https://github.com/chansoopark98/Deep-Visual-SLAM-Monodepth2.git
cd Deep-Visual-SLAM-Monodepth2
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 bash scripts/setup_env.sh
source .venv/bin/activate
```

`scripts/setup_env.sh` creates a venv, installs PyTorch (if missing), `requirements.txt` and the package (`pip install -e .`),
then runs an import and CUDA check. Without `TORCH_INDEX_URL` it prints the PyTorch install command and stops.

| Variable | Default | Meaning |
|---|---|---|
| `PYTHON` | `python3` | Interpreter that creates the venv (must be >= 3.10) |
| `DVSLAM_VENV` | `<repo>/.venv` | Venv directory |
| `DVSLAM_CONDA_ENV` | unset | If set, create or reuse this conda env (Python 3.11) instead of a venv |
| `TORCH_INDEX_URL` | unset | Wheel index used to install `torch` and `torchvision` when missing |
| `DVSLAM_SKIP_PIP=1` | unset | Only check an existing environment; install nothing |

Conda variant: `DVSLAM_CONDA_ENV=dvslam TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 bash scripts/setup_env.sh`.

Manual equivalent:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
```

## Data

The evaluation uses TUM-VI (CC BY 4.0, [dataset page](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset)),
exported / euroc / 512_16 variant. Datasets live outside the repository. The loaders find them through environment
variables:

| Variable | Used by |
|---|---|
| `DVSLAM_TUMVI_ROOT` | TUM-VI root, containing `dataset-room{N}_512_16/` |
| `DVSLAM_CACHE_ROOT` | Rectified-image cache, built on the first run of a room |
| `DVSLAM_EUROC_ROOT`, `DVSLAM_REDWOOD_ROOT` | EuRoC and Redwood roots (only for [other datasets](#other-datasets)) |

The defaults of the loaders and of `scripts/fetch_datasets.sh` are paths on the author's workstation, so set the
variables explicitly. `scripts/demo_tumvi.sh` defaults to `<repo>/data/tum_vi` and `<repo>/runs/cache`; use the same
values everywhere:

```bash
export DVSLAM_TUMVI_ROOT=$PWD/data/tum_vi
export DVSLAM_CACHE_ROOT=$PWD/runs/cache
bash scripts/fetch_datasets.sh tumvi room1 room2 room3 room4 room5 room6   # room5 1.7 GB, room6 1.6 GB, about 1.6-1.7 GB each
```

The download is resumable and checks size and archive integrity. After a download, and for `bash scripts/fetch_datasets.sh
check`, the script prints how many sequences of each EuRoC, TUM-VI and Redwood split the loaders can open; it exits
non-zero while any EuRoC or Redwood sequence is missing, which does not matter for the TUM-VI results. The demo needs only
the room it runs (`room5` by default).
Layouts, source URLs, the cache and troubleshooting are in [docs/DATASETS.md](docs/DATASETS.md).

## Pretrained weights

Release [`v0.1.0`](https://github.com/chansoopark98/Deep-Visual-SLAM-Monodepth2/releases/tag/v0.1.0) (research release;
details in the model card published with it).

| File | Size | What it is |
|---|---|---|
| `dvslam_depth_hints_core.pt` | 163.7 MB | Final-candidate depth network (trained with stereo depth hints) |
| `dvslam_depth_core.pt` | 163.7 MB | Same depth network trained without stereo hints |
| `dvslam_s2_photonet_v4.pt` | 1.2 MB | S2 `PhotoNet` per-pixel weight network (optional) |

```bash
bash scripts/download_weights.sh      # -> weights/dvslam/, SHA-256 checksums verified
```

| Variable | Default | Meaning |
|---|---|---|
| `DVSLAM_WEIGHTS` | `<repo>/weights/dvslam` | Destination directory |
| `DVSLAM_WEIGHTS_URL` | the `v0.1.0` release | Base URL of the files; a local mirror (`file:///...`) also works |

Files that are already present with a matching checksum are not downloaded again. The script also fetches `SHA256SUMS`
and `MODEL_CARD.md`. RAFT-large weights are not part of the release; torchvision downloads them on first use.

## Quick demo

```bash
bash scripts/demo_tumvi.sh room5      # room1..room6, default room5
```

Needs a CUDA GPU, the room's TUM-VI sequence (see [Data](#data)) and the downloaded weights. The script:

1. runs the final-candidate tracker with the live depth prior over the whole room,
2. scores the translation step error and the rotation error,
3. renders an animated GIF.

Outputs go to `runs/demo/<room>-<depth>-<timestamp>/` (override with `OUT`):

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

Options (environment variables):

| Variable | Default | Meaning |
|---|---|---|
| `DEPTH` | `hints_core` | `hints_core` or `core` (`dvslam_depth_<DEPTH>.pt`) |
| `PHOTONET=1` | off | Also use the S2 weights (`dvslam_s2_photonet_v4.pt`). Not benchmarked end to end in this form. |
| `OUT` | `runs/demo/...` | Output directory |
| `GIF_ARGS` | empty | Extra arguments for `scripts/render_demo.py`, e.g. `"--stride 6 --fps 12"` |
| `DVSLAM_THREADS` | `4` | Cap for OMP / BLAS threads |

### Rendering

`scripts/render_demo.py` draws a tracker run as an animated GIF (`--out`) and a static geometry figure (`--png`: top-down
trajectory and final 3D map):

```bash
python scripts/render_demo.py --run runs/demo/<run> --scene room6 \
    --out media/room6_demo.gif --png media/room6_geometry.png \
    [--start 0 --end -1 --stride 11 --fps 10 --width 1080 --dpi 150]
```

At least one of `--out` and `--png` is required. Other options: `--grays`, `--azim`, `--elev`, `--margin`, `--min-obs`,
`--kf-size`, `--kf-spacing` (3D view) and `--data-root` (TUM-VI root for runs without `<scene>.seq.npz`); `-h` lists all.
The script prints the frame count, translation step error, rotation error and ATE (SE(3)) of the rendered run.

### Other datasets

`scripts/demo_track.py` runs the tracker on one sequence of any dataset (`--dataset tumvi|euroc|redwood`) and writes
everything the renderer needs (`<scene>.inc.npz`, `.extra.npz`, `.tracks.npz`, `.map.npz`, `.seq.npz`, `.cpu.json`).
It takes the tracker options as repeated `--opt k=v`; use the final-candidate list from
[Reproducing the evaluation](#reproducing-the-evaluation). Set `DVSLAM_EUROC_ROOT` / `DVSLAM_REDWOOD_ROOT` first.

```bash
OPTS="--opt prior=ssl --opt prior_ckpt=weights/dvslam/dvslam_depth_hints_core.pt --opt quality=0.001 --opt lk_win=15
  --opt jump_ref_tracked=true --opt photo=all --opt photo_tol=0.01 --opt photo_levels=1 --opt ba_nfev=10
  --opt n_features=250 --opt kf_ratio=0.3 --opt prior_per_kf=false --opt g_tau=5 --opt flow_cache=raft_large
  --opt kf_refine=0 --opt photo_cell=1 --opt photo_grad=0.002 --opt photo_bidir=true --opt photo_backend=torch"

python scripts/demo_track.py --dataset euroc --scene MH_01_easy --out runs/demo/euroc $OPTS
python scripts/render_demo.py --run runs/demo/euroc --scene MH_01_easy \
    --out media/euroc_MH_01_easy_demo.gif --png media/euroc_MH_01_easy_geometry.png --stride 17 --fps 10
```

| Option | Default | Meaning |
|---|---|---|
| `--dataset` | required | `redwood`, `tumvi` or `euroc` |
| `--scene` | required | Sequence name, e.g. `room6`, `MH_01_easy`, `bedroom` |
| `--out` | required | Output directory |
| `--split` | inferred (`test` for Redwood) | Dataset split |
| `--camera` | `tumvi` except for TUM-VI | `tumvi` re-projects the images into the TUM-VI 512 x 512, f = 190 pinhole camera, which the depth network requires; `native` keeps the dataset camera |
| `--erode` | `8` | Pixels the valid-region mask is eroded by (`--camera tumvi`) |
| `--frames` | `0` | Track only the first N frames (`0` = all) |
| `--cv-threads` | unset | OpenCV thread count |

The scores in [Results on other datasets](#results-on-other-datasets) come from such runs: the depth network sees the
re-projected image, so its depth is zero-shot there. Re-projected canvases are cached under
`DVSLAM_CACHE_ROOT/<dataset>/<scene>/tumvi_512_f190`.

## Tests

```bash
bash scripts/run_tests.sh                       # unit tests + ruff report
bash scripts/run_tests.sh tests/test_release_tools.py -x   # extra arguments go to pytest
```

Tests that need data skip themselves (reasons are listed). `ruff` findings are printed but do not change the exit status. `DVSLAM_THREADS` (default 4) caps the thread count.

## Reproducing the evaluation

Run the final-candidate tracker over the official set (`room1`–`room4` from the `train` split, `room6` from `test`):

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

## Training

All training code runs on a CUDA GPU. Run directories go to `runs/` (git-ignored).

### SSL depth network

```bash
python apps/train_ssl.py --config configs/ssl_tumvi.yaml --name base [--set loss.flow=0 --set train.epochs=30]
python apps/train_ssl.py --name base --resume          # continue runs/ssl/base/last.pt
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 apps/train_ssl.py --name base   # one process per GPU
```

Writes `runs/ssl/<name>/{config.yaml,train_log.jsonl,val_log.jsonl,last.pt,best.pt}`. The config trains on `room1`–`room4`
and validates on `room5`. Stereo depth hints (the `hints_core` release) are made once per room with
`python scripts/cache_stereo_hints.py --out $DVSLAM_CACHE_ROOT/tumvi_stereo_hints room1 room2 room3 room4 room5`
(cam0/cam1 SGBM, training-time signal only) and enabled with `--set data.stereo_hints=$DVSLAM_CACHE_ROOT/tumvi_stereo_hints
--set loss.stereo_hint=1.0` (the released `hints_core` used weight 1.0, 20 epochs, batch 8, lr 1e-4, seed 0).

### S2 learned photometric weights

Dump the tracker's refinement inputs per room (same arguments as `scripts/exp003_track_log.py`; use the final-candidate
`--opt` list from above), then train `PhotoNet` through the unrolled refinement:

```bash
python scripts/s2_dump_pairs.py --scene room1 --out runs/s2/dump <final-candidate --opt list>   # once per room
python apps/train_photo_s2.py --dump runs/s2/dump --depth <depth cache dir> --out runs/s2/train/pilot --mode w --steps 2000
```

`<depth cache dir>` is the frozen depth cache of the tracker (`python scripts/exp003_cache_depth.py --ckpt <ckpt>
--out <dir>`). Use the result with `--opt photo_feat=<file>` (requires `photo_backend=torch`, `photo_levels=1`).

### S3 test-time depth adaptation

Uses the pseudo-label poses of one tracker run on the test sequence (`--labels`, start frame 0), cam0 images only:

```bash
python apps/calib_s3.py train --ckpt runs/ssl/<run>/last.pt --scene room5 --labels <run>/room5.inc.npz --out runs/s3/room5
python apps/calib_s3.py cache --ckpt runs/ssl/<run>/last.pt --adapter runs/s3/room5/adapter.pt --scene room5 --out runs/s3/depth
```

`cache` writes a depth cache that the tracker reads with `--opt prior=cache --opt prior_ckpt=runs/s3/depth`.
Run `python apps/calib_s3.py <command> -h` for all options.

### Exporting release weights

```bash
python scripts/export_release_weights.py SRC DST [--verify-images IMG.png ...] [--device cuda]
```

Strips a training checkpoint (EMA weights, config, K, size, epoch, step) or a PhotoNet file to an inference-only file and,
with `--verify-images`, checks that the depth output is bit-identical.

## Repository layout

```
apps/                  Entry points: run_sequence, eval_pairs, train_ssl, train_photo_s2, calib_s3 (train, cache)
configs/               ssl_tumvi.yaml (SSL depth training)
dvslam/
  data/                TUM-VI, EuRoC, Redwood loaders, dataset registry, rectified-image cache
  depth/               Depth priors behind the tracker (SSL network; perturbed GT depth for tests)
  odometry/            Tracker: klt_window.py (final pipeline) and its parts (klt_pnp, klt_flow, klt_photo,
                       klt_photo_torch, klt_depth_cache, klt_lag, photo_learned, base)
  ssl/                 Self-supervised depth + pose training, predictor, LoRA adapter (adapter.py)
  eval/                Pair-wise pose-error metrics and diagnostics
  geometry/            SE(3) helpers
scripts/               setup_env.sh, download_weights.sh, fetch_datasets.sh, run_tests.sh, demo_tumvi.sh,
                       demo_track.py (any dataset), render_demo.py (GIF and geometry PNG), export_release_weights.py,
                       exp003_*.py / exp009_stereo_depth.py / cache_stereo_hints.py / s2_dump_pairs.py
                       (tracking log, scoring, depth caches, S2 dumps)
tests/                 pytest suite
docs/DATASETS.md       Dataset layouts, download, cache
media/                 Demo GIFs and geometry figures (TUM-VI room6, EuRoC MH_01 / MH_03, Redwood bedroom)
model/                 ResNet encoder and depth decoder used by dvslam/ssl
weights/dvslam/        Downloaded release weights (git-ignored)
```

Top level: `README.md`, `LICENSE`, `CODE_OF_CONDUCT.md`, `pyproject.toml`, `requirements.txt`.

## Legacy: Monodepth2-style VO

The legacy Monodepth2 code (depth and pose networks trained on Redwood and NYU Depth V2, ROS 2 and C++ tools) is on
branch `legacy/monodepth2`. Earlier stereo-training and PyTorch-port work is on `legacy/stereo-training` and
`legacy/torch`.

## References

- [Monodepth2](https://github.com/nianticlabs/monodepth2): Godard et al., *Digging into Self-Supervised Monocular Depth Prediction*, ICCV 2019
- [RAFT](https://github.com/princeton-vl/RAFT): Teed and Deng, *RAFT: Recurrent All-Pairs Field Transforms for Optical Flow*, ECCV 2020 (torchvision `raft_large`)
- [TUM-VI](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset): Schubert et al., *The TUM VI Benchmark for Evaluating Visual-Inertial Odometry*, IROS 2018
- [LoRA](https://arxiv.org/abs/2106.09685): Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021
- [Redwood Indoor Dataset](http://redwood-data.org/indoor_lidar_rgbd/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
