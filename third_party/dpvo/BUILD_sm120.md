# Building DPVO for sm_120 (RTX 5090)

Environment used (2026-11-18): Ubuntu 22.04, gcc 11.4.0, CUDA 12.8 (`/usr/local/cuda-12.8`, nvcc V12.8.93),
conda **base** env: Python 3.13, torch 2.9.0+cu128, numpy 2.2.6 (both unchanged). No separate conda env was needed.

## Steps

```bash
cd third_party/dpvo

# 1. Python deps DPVO's VO path imports (pure Python; torch/numpy untouched)
pip install --no-deps yacs==0.1.8 einops==0.8.1      # einops resolved to 0.8.2

# 2. Eigen 3.4.0: system headers (apt libeigen3-dev, EIGEN_*_VERSION = 3.4.0) are used; no download.
#    setup.py falls back to /usr/include/eigen3 when thirdparty/eigen-3.4.0 is absent (EIGEN_INCLUDE_DIR overrides).

# 3. Build the three CUDA extensions in place (no pip install; *.so land next to setup.py, git-ignored)
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
MAX_JOBS=8 python setup.py build_ext --inplace        # ~2 min; TORCH_CUDA_ARCH_LIST defaults to 12.0 in setup.py
#   -> cuda_corr.*.so, cuda_ba.*.so, lietorch_backends.*.so
cuobjdump --list-elf cuda_ba.cpython-313-x86_64-linux-gnu.so   # shows *.sm_120.cubin

# 4. Weights (official link from download_models_and_data.sh)
mkdir -p models && cd models
curl -sSL -o models.zip "https://www.dropbox.com/scl/fo/2g6e134ruh1acmaye5bmy/AK0DY9WLZ25elq-7N7sJcB0/models.zip?rlkey=4j336vcc4h324j4p4ji3ppl0t&dl=1"
unzip models.zip && sha256sum dpvo.pth   # 30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12
```

`dvslam/odometry/dpvo_mono.py` puts `third_party/dpvo` on `sys.path`, so neither the package nor the
extensions are installed into site-packages.

## Source fixes needed (see NOTICE.md, "Local modifications")

| Failure | Fix |
|---|---|
| `THC/THCAtomics.cuh` include (THC removed) | `ATen/cuda/Atomic.cuh` |
| `error: no suitable conversion function from "const at::DeprecatedTypeProperties" to "c10::ScalarType"` (4× in `correlation_kernel.cu`, also lietorch dispatch) | `x.type()` → `x.scalar_type()` in the `AT_DISPATCH_*` / `DISPATCH_GROUP_AND_FLOATING_TYPES` calls |
| no arch flags (would default to the visible GPU or all arches) | `TORCH_CUDA_ARCH_LIST=12.0` default in `setup.py` |
| `torch_scatter` has no torch 2.9 / cp313 / sm_120 wheel | `dpvo/scatter.py` native-torch `scatter_sum`, `scatter_softmax` (unit-tested against a reference in `tests/test_dpvo_mono.py`) |
| `import numba` at import time via `patchgraph.py` → `loop_closure/optim_utils.py` | import made lazy inside `PatchGraph.edges_loop` (loop closure only) |

Remaining warnings: `torch.cuda.amp.autocast` FutureWarning (harmless).

## Check

```bash
python -m pytest tests/test_dpvo_mono.py -m gpu -q    # smoke: 60 frames of validation/bedroom
```

## Plan 1.1 / 1.2 extensions (dpvo_metric)

```bash
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
MAX_JOBS=8 python setup_metric.py build_ext --inplace --build-temp build/metric          # -> cuda_ba_metric.*.so
MAX_JOBS=8 python setup_stream.py build_ext --build-lib build/stream_ext --build-temp build/stream_tmp
#   -> build/stream_ext/{cuda_corr,lietorch_backends}.*.so (current-stream launches; only for cuda_graph=1)
```
Neither command touches the frozen in-place `cuda_ba` / `cuda_corr` / `lietorch_backends` builds.
