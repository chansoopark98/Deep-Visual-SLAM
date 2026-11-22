# SLAM evaluation datasets: layout, download, repository setup

Datasets live **outside the repository** (they are tens of GB). The loaders in `dvslam/data/` find them through
environment variables; by default they look under `data/` in the repository root (git-ignored), so either
download there or point the variables at an existing copy (a symlink `data/tum_vi -> /path/to/tum_vi` also works).

| Variable | Default | Used by |
|---|---|---|
| `DVSLAM_TUMVI_ROOT` | `data/tum_vi` | `dvslam/data/tumvi.py` |
| `DVSLAM_EUROC_ROOT` | `data/euroc` | `dvslam/data/euroc.py` |
| `DVSLAM_CACHE_ROOT` | `data/cache` | `dvslam/data/base.py` (rectified-image cache) |

Redwood is read from `DVSLAM_REDWOOD_ROOT` (default `data/redwood`) or `--data-root`; see [OTHER_DATASETS.md](OTHER_DATASETS.md). Keep working data on
**local disk**, not on the NFS mounts (`/mnt/*`): the NFS volume has filled up before, and rectified-image reads are
random-access.

## 1. Check what is ready

```bash
scripts/fetch_datasets.sh check          # exit 0 only if every sequence of every split is usable
```

`check` goes through `dvslam.data.registry` (the same code the apps use), so a pass means the loaders can open the
sequence, not just that a directory exists. Required sequences:

| Dataset | Split | Sequences | Used by |
|---|---|---|---|
| EuRoC (machine hall) | validation | MH_01_easy, MH_02_easy | `apps/run_sequence.py --dataset euroc` |
| EuRoC (machine hall) | test | MH_03_medium, MH_04_difficult, MH_05_difficult | same |
| TUM-VI (512_16, cam0) | train | room1, room2, room3, room4 | `--dataset tumvi` |
| TUM-VI (512_16, cam0) | validation | room5 | same |
| TUM-VI (512_16, cam0) | test | room6 | same |
| Redwood | validation / test | apartment, bedroom, boardroom, lobby, loft | `--dataset redwood` |

Splits are defined in `SPLITS` of each loader (`euroc.py`, `tumvi.py`), not here; if they change, rerun `check`.
TUM-VI (since 2026-11-19): split by whole sequence; all six rooms are the same motion-capture room. The
official evaluation set is train + test:
`apps/run_sequence.py --dataset tumvi --split train test` then `apps/eval_pairs.py` (PASS/FAIL only on that set).
State on 2026-11-19 before the download below: everything present except TUM-VI room5 and room6.

## 2. Download

```bash
scripts/fetch_datasets.sh tumvi room5 room6      # ~3.3 GB, official TUM server
scripts/fetch_datasets.sh euroc MH_03_medium     # any missing EuRoC sequence (see the EuRoC note below)
```

The script is idempotent (skips sequences already extracted), resumes interrupted downloads, checks the byte count
against the server's `Content-Length`, checks archive integrity (`tar -t` / `unzip -t`) and the top-level directory
name, extracts, appends `<date> <url> <bytes> <sha256>` to `<root>/MANIFEST.txt`, then deletes the archive.
It runs one `curl` stream and no parallel jobs, so it does not load the CPU.

### TUM-VI

- Page: <https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset> (CC BY 4.0, cite Schubert et al., IROS 2018).
- Variant used: **exported / euroc / 512_16** (512x512 fisheye, 16-bit, EuRoC folder layout):
  `https://cdn3.vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-room{N}_512_16.tar` (room5 1.70 GB, room6 1.60 GB).
- The server publishes no checksums. Integrity is the size and tar check above; the sha256 recorded in
  `MANIFEST.txt` is for later comparison only.
- Layout the loader needs, per sequence `<root>/dataset-room{N}_512_16/`:
  `mav0/cam0/data.csv` + `mav0/cam0/data/<ns>.png`, `mav0/mocap0/data.csv` (120 Hz GT), `dso/camchain.yaml`.
  Room sequences have mocap GT for the whole run; corridor/outdoor/slides sequences only at the start and end.

### EuRoC MAV

- Page: <https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets>
  (also published as a DOI record in the ETH Research Collection; cite Burri et al., IJRR 2016).
- Layout the loader needs, per scene `<root>/machine_hall/<scene>/mav0/`:
  `cam0/data.csv` + `cam0/data/*.png`, `cam0/sensor.yaml`, `state_groundtruth_estimate0/data.csv`.
- **Direct links were not verifiable on 2026-11-19**: `robotics.ethz.ch` did not answer and the Research Collection
  returned HTTP 429. All five machine-hall sequences were already on disk, so nothing needed downloading. The
  `euroc` mode of the script uses the classic `robotics.ethz.ch/~asl-datasets/...` URLs and fails with a clear
  message if they are unreachable; in that case download the `.zip` by hand from the page above and unzip it to
  `<root>/machine_hall/<scene>/` (so that `<scene>/mav0/cam0/data.csv` exists). The Vicon-room sequences (V1/V2)
  are not used by the loaders.

## 3. Repository setup

```bash
conda activate vslam                     # or any env with numpy, scipy, pyyaml, opencv-python (requirements.txt)
pip install -r requirements.txt          # pyyaml is required by the TUM-VI and EuRoC loaders
export DVSLAM_TUMVI_ROOT=/path/to/tum_vi   # only if the data is not at the defaults above
export DVSLAM_EUROC_ROOT=/path/to/euroc
export DVSLAM_CACHE_ROOT=/path/to/cache
scripts/fetch_datasets.sh check
python apps/run_sequence.py --odometry <name> --dataset tumvi --split test --scenes room6
```

On a shared workstation cap the math libraries first (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`);
the rectification step is CPU-bound and an uncapped run once froze the machine.

### Rectified-image cache

TUM-VI (equidistant fisheye) and EuRoC (radial-tangential) images are remapped once to a pinhole model and cached at
`$DVSLAM_CACHE_ROOT/<dataset>/<scene>/<tag>/` (about 3.3 GB for the current sequences; `tag` encodes the
parameters, e.g. `cam0_pinhole_512_f190`). The first run of a sequence builds its cache; later runs reuse it. If
the rectification parameters change the loader raises an error instead of reusing a stale cache: delete that
`<tag>` directory and rerun.
