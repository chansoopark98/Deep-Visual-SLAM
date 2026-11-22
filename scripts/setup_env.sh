#!/usr/bin/env bash
# Create a Python environment for the dvslam tracker, tests and demo.
#
#   scripts/setup_env.sh                       # venv at .venv (or $DVSLAM_VENV)
#   DVSLAM_CONDA_ENV=dvslam scripts/setup_env.sh   # conda env instead of a venv
#
# Environment variables
#   PYTHON            interpreter that creates the venv (default: python3, must be >= 3.10)
#   DVSLAM_VENV       venv directory (default: <repo>/.venv)
#   DVSLAM_CONDA_ENV  if set: create / reuse this conda env (python 3.11) instead of a venv
#   TORCH_INDEX_URL   if torch is missing, install torch + torchvision from this wheel index, e.g.
#                     https://download.pytorch.org/whl/cu128 (CUDA 12.8 builds; needed for RTX 50xx).
#                     Unset: print the command and stop, so the CUDA build is your choice.
#   DVSLAM_SKIP_PIP=1 only check an existing environment (no pip installs)
#
# Then: source the env (printed at the end), scripts/download_weights.sh, scripts/run_tests.sh,
# scripts/demo_tumvi.sh.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"

if [ -n "${DVSLAM_CONDA_ENV:-}" ]; then
  command -v conda > /dev/null || { echo "conda not found (unset DVSLAM_CONDA_ENV to use a venv)"; exit 1; }
  if ! conda run -n "$DVSLAM_CONDA_ENV" true 2> /dev/null; then
    conda create -y -n "$DVSLAM_CONDA_ENV" python=3.11
  fi
  PY=$(conda run -n "$DVSLAM_CONDA_ENV" python -c 'import sys; print(sys.executable)')
  ACTIVATE="conda activate $DVSLAM_CONDA_ENV"
else
  VENV=${DVSLAM_VENV:-$REPO/.venv}
  BASE_PY=${PYTHON:-python3}
  "$BASE_PY" -c 'import sys; assert sys.version_info >= (3, 10), sys.version' \
    || { echo "$BASE_PY is older than Python 3.10"; exit 1; }
  [ -x "$VENV/bin/python" ] || "$BASE_PY" -m venv "$VENV"
  PY=$VENV/bin/python
  ACTIVATE="source $VENV/bin/activate"
fi
echo "python: $PY ($("$PY" -c 'import sys; print(sys.version.split()[0])'))"

if [ "${DVSLAM_SKIP_PIP:-0}" != 1 ]; then
  if ! "$PY" -c 'import torch, torchvision' 2> /dev/null; then
    if [ -z "${TORCH_INDEX_URL:-}" ]; then
      cat <<EOF
PyTorch is not installed in this environment. Install a CUDA build that matches your driver, e.g. (CUDA 12.8):
  $PY -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
or re-run with TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 $0
EOF
      exit 2
    fi
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"
  fi
  "$PY" -m pip install -r requirements.txt
  "$PY" -m pip install -e .
fi

# sanity check
"$PY" - <<'EOF'
import cv2, numpy, PIL, scipy, torch, torchvision, yaml  # noqa: F401
import dvslam
from dvslam.odometry.klt_window import KLTWindowOdometry  # noqa: F401
from dvslam.ssl.predict import Predictor  # noqa: F401
print(f'torch {torch.__version__}, torchvision {torchvision.__version__}, opencv {cv2.__version__}, '
      f'numpy {numpy.__version__}')
print(f'dvslam from {dvslam.__path__[0]}')
cuda = torch.cuda.is_available()
print(f'torch.cuda.is_available(): {cuda}' + (f' ({torch.cuda.get_device_name(0)})' if cuda else
      '  -> the demo (RAFT-large + live SSL depth) needs a CUDA GPU; unit tests run on the CPU'))
EOF
echo "OK. Activate with: $ACTIVATE"
