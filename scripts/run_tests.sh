#!/usr/bin/env bash
# Run the unit test suite (thread-capped) and ruff (report only).
#
#   scripts/run_tests.sh                 # pytest -q over tests/ (pyproject: -m 'not gpu') + ruff check
#   scripts/run_tests.sh tests/test_release_tools.py -x   # extra arguments go to pytest
#
# Environment variables
#   PYTHON        interpreter (default: python)
#   DVSLAM_THREADS  OMP / MKL / OpenBLAS / numexpr threads (default 4)
#   DVSLAM_TUMVI_ROOT, DVSLAM_REDWOOD_ROOT, DVSLAM_EUROC_ROOT, DVSLAM_CACHE_ROOT: dataset roots of the loaders
# Tests that need data, a GPU or optional packages skip themselves (reasons are listed by -rs): the Redwood / TUM-VI /
# EuRoC loader and app tests skip when the dataset is not on disk; tests marked `gpu` (RTX + data + baseline
# weights) are deselected by pyproject's addopts (run them with: scripts/run_tests.sh -m gpu); open3d / cuvslam /
# TensorRT tests skip when the package or the built C++ loop is missing. A few unmarked tests use CUDA when it is
# available and the CPU otherwise.
# Exit status: pytest's. ruff findings are printed but do not fail the script.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
PY=${PYTHON:-python}
N=${DVSLAM_THREADS:-4}
export OMP_NUM_THREADS=$N MKL_NUM_THREADS=$N OPENBLAS_NUM_THREADS=$N NUMEXPR_NUM_THREADS=$N

status=0
nice -n 10 "$PY" -m pytest -q -rs "$@" || status=$?

echo "== ruff check dvslam apps scripts (report only)"
if command -v ruff > /dev/null; then RUFF=(ruff); elif "$PY" -m ruff --version > /dev/null 2>&1; then RUFF=("$PY" -m ruff)
else RUFF=(); echo "ruff not installed (pip install ruff): skipped"; fi
if [ ${#RUFF[@]} -gt 0 ]; then
  "${RUFF[@]}" check dvslam apps scripts --statistics || echo "ruff: findings above (not an error for this script)"
fi
exit $status
