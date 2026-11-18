# third_party/dpvo — NOTICE

## Origin

- Upstream: DPVO / DPV-SLAM, https://github.com/princeton-vl/DPVO (Teed, Lipson, Deng; Lipson, Teed, Deng).
- Copied from the read-only reference checkout `assets/references/dpvo` at commit **0ac95b6**
  ("Update download_models_and_data.sh", 2026-09-20) on 2026-11-18, for plan v3 item 0.6.
- Licence: **MIT** — `LICENSE` in this directory (unchanged, copyright Princeton Vision & Learning Lab).

Not copied (not needed for VO mode; each has its own licence upstream): the `Pangolin` and `DBoW2`
git submodules, `DPViewer/` and `DPRetrieval/` (both vendor pybind11), `logs/`, `.git`.

## Bundled third-party code

| Component | Where | Licence | Note |
|---|---|---|---|
| lietorch (Teed & Deng) | `dpvo/lietorch/` | BSD-3-Clause | Upstream DPVO vendors it **without** its licence file. `dpvo/lietorch/LICENSE` was added here, copied from `assets/references/droid-slam/thirdparty/lietorch/LICENSE` (same authors, same code base). |
| Eigen 3.4.0 | not copied; headers used at build time | MPL-2.0 (headers only, unmodified) | Built against the system copy `/usr/include/eigen3` (Ubuntu `libeigen3-dev`, 3.4.0). `setup.py` prefers `thirdparty/eigen-3.4.0/` if the upstream zip is unpacked there. |
| torch_scatter (upstream runtime dependency) | **replaced** | (MIT upstream) | `dpvo/scatter.py` re-implements `scatter_sum` / `scatter_softmax` with native torch ops. No torch_scatter code is included. |

## Weights

| Item | Value |
|---|---|
| File | `models/dpvo.pth` (git-ignored, not redistributed with this repo) |
| Source | `models.zip` from the URL in upstream `download_models_and_data.sh` line 4: `https://www.dropbox.com/scl/fo/2g6e134ruh1acmaye5bmy/AK0DY9WLZ25elq-7N7sJcB0/models.zip?rlkey=4j336vcc4h324j4p4ji3ppl0t` (fetched with `&dl=1` appended) |
| sha256 `models.zip` | `89f43de2c92676ddcf7f49e8ae3f8940b7af73f549a7a5b8308850b67de78479` (13,100,005 B) |
| sha256 `dpvo.pth` | `30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12` (14,167,743 B, zip entry dated 2022-07-24) |
| Licence status | **Unstated.** Neither the README nor the repo gives a separate licence for the weights; by default they would follow the code (MIT) but this is an inference, not a grant. They were trained on TartanAir (synthetic; check TartanAir's terms before any redistribution). Treat as research-use only until confirmed. |

## Local modifications (all marked `dvslam:` in the source)

1. `dpvo/altcorr/correlation_kernel.cu`: `#include <THC/THCAtomics.cuh>` → `#include <ATen/cuda/Atomic.cuh>` (THC removed from libtorch).
2. `dpvo/altcorr/correlation_kernel.cu`, `dpvo/lietorch/src/lietorch_gpu.cu`, `dpvo/lietorch/src/lietorch_cpu.cpp`: dispatch on `x.scalar_type()` instead of the removed `x.type()` (torch 2.9 no longer converts `DeprecatedTypeProperties` to `ScalarType`).
3. `setup.py`: `TORCH_CUDA_ARCH_LIST` defaults to `12.0` (RTX 5090); Eigen include path falls back to `/usr/include/eigen3`.
4. `dpvo/scatter.py` (new) and imports in `dpvo/net.py`, `dpvo/ba.py`, `dpvo/blocks.py`: native replacement for torch_scatter. (`dpvo/loop_closure/long_term.py` still imports torch_scatter; it is only used with `CLASSIC_LOOP_CLOSURE`, which is off.)
5. `dpvo/patchgraph.py`: `reduce_edges` (numba/pypose) imported lazily inside `edges_loop`, so VO mode needs neither.
6. `dpvo/config.py` + `dpvo/dpvo.py:__call__`: new `LATEST_UPDATES` option (default 1 = upstream behaviour): number of `update()` calls on each new frame after initialisation. Used by the plan's A-N3 ρ trigger.

Build steps: `BUILD_sm120.md`.
7. **Plan v3 items 1.1 / 1.2 (2026-11-18), used by `dvslam/odometry/dpvo_metric.py` only:**
   - `dpvo/fastba/ba_cuda.cu`: `cuda_ba` takes four extra arguments (`prior_disp`, `prior_w`, `prior_lambda`,
     `prior_huber`) and, when `prior_lambda > 0`, adds a per-patch log-depth unary
     (`depth_prior_unary_kernel`: r = log z_prior − log z, w = λ · prior_w · Huber(r; δ), Gauss-Newton terms into
     the patch blocks C, u before the Schur complement). **The idea follows ViPE MetricTrail**
     (`assets/references/vipe/vipe/metrictrail/csrc/dpvo/fastba/ba_cuda.cu:402-437`, NVIDIA, **Apache-2.0**);
     the kernel was re-implemented here, no ViPE code was copied.
   - `dpvo/fastba/ba.cpp`: the existing `forward` binding is unchanged in signature (passes an empty prior);
     new binding `forward_prior`. `dpvo/fastba/ba.py`, `dpvo/fastba/__init__.py`: new wrapper `BA_prior`.
   - `setup_metric.py` (new): builds the same BA sources as module **`cuda_ba_metric`** so the frozen
     `cuda_ba` build used by `dpvo_mono` is never rebuilt or overwritten.
   - `dpvo/static_graph.py` (new): fixed-shape (E_max-padded) update operator — static group ids, GPU
     neighbour table (no CPU sort), static-size `scatter_softmax` / `scatter_sum` (no `torch.unique`, no
     `index.max()` host syncs). Equal to `Update.forward` on the real edges up to summation order (tested).
   - `dpvo/altcorr/correlation_kernel.cu`, `dpvo/lietorch/src/lietorch_gpu.cu`: every kernel launch passes
     `at::cuda::getCurrentCUDAStream()` (upstream launched on the legacy default stream, which breaks CUDA-graph
     capture and side-stream use). Behaviour on the default stream is unchanged. `setup_stream.py` (new)
     builds these two modules into `build/stream_ext/` (used only with `cuda_graph=1`); the in-place
     `cuda_corr` / `lietorch_backends` builds were **not** rebuilt.
   - `dpvo/dpvo.py` itself is unchanged: `dvslam/odometry/dpvo_metric.py` subclasses `DPVO` (prior init at
     patch creation, prior/weight buffers shifted with the patch slots on culling, BA failure recorded instead
     of swallowed, ring-buffer edges).
