#!/bin/bash
# Fetch / verify the SLAM evaluation datasets (TUM-VI, EuRoC). See docs/DATASETS.md.
#
#   scripts/fetch_datasets.sh check                 # which sequences of each split are usable (no download)
#   scripts/fetch_datasets.sh tumvi [room5 room6]   # TUM-VI exported/euroc/512_16 sequences (default: room5 room6)
#   scripts/fetch_datasets.sh euroc [MH_01_easy..]  # EuRoC machine-hall sequences (default: all five)
#
# Roots match the loaders: DVSLAM_TUMVI_ROOT, DVSLAM_EUROC_ROOT (dvslam/data/{tumvi,euroc}.py).
# Single sequential curl stream, resumable (-C -). Archives are deleted after a verified extract.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TUMVI_ROOT=${DVSLAM_TUMVI_ROOT:-/home/park-ubuntu/park/datasets/tum_vi}
EUROC_ROOT=${DVSLAM_EUROC_ROOT:-/home/park-ubuntu/park/datasets/euroc}
TUMVI_URL=https://cdn3.vision.in.tum.de/tumvi/exported/euroc/512_16
EUROC_URL=http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall

# download <url> <dest>; verifies the byte count against the server's Content-Length.
download() {
  local url=$1 dest=$2 want
  want=$(curl -sIL --retry 3 "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{n=$2} END{print n}')
  [ -n "$want" ] || { echo "FAIL $url: no Content-Length (server unreachable or rate-limited)"; return 1; }
  curl -fSL --retry 5 -C - -o "$dest" "$url"
  [ "$(stat -c %s "$dest")" = "$want" ] || { echo "FAIL $dest: size $(stat -c %s "$dest") != $want"; return 1; }
}

manifest() {  # manifest <root> <url> <archive>
  echo "$(date -I) $2 $(stat -c %s "$3") $(sha256sum "$3" | cut -d' ' -f1)" >> "$1/MANIFEST.txt"
}

fetch_tumvi() {
  mkdir -p "$TUMVI_ROOT"
  for s in "${@:-room5 room6}"; do
    for r in $s; do
      d="$TUMVI_ROOT/dataset-${r}_512_16"
      [ -s "$d/mav0/cam0/data.csv" ] && { echo "have $r"; continue; }
      tar="$TUMVI_ROOT/dataset-${r}_512_16.tar"
      download "$TUMVI_URL/dataset-${r}_512_16.tar" "$tar"
      tar -tf "$tar" >/dev/null                                   # archive integrity
      [ "$(tar -tf "$tar" | head -1 | cut -d/ -f1)" = "dataset-${r}_512_16" ] || { echo "FAIL $r: unexpected layout"; return 1; }
      tar -xf "$tar" -C "$TUMVI_ROOT"
      manifest "$TUMVI_ROOT" "$TUMVI_URL/dataset-${r}_512_16.tar" "$tar"
      rm -f "$tar"; echo "ok $r"
    done
  done
}

fetch_euroc() {
  mkdir -p "$EUROC_ROOT/machine_hall"
  for s in "${@:-MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult}"; do
    for r in $s; do
      d="$EUROC_ROOT/machine_hall/$r"
      [ -s "$d/mav0/cam0/data.csv" ] && { echo "have $r"; continue; }
      zip="$d.zip"
      download "$EUROC_URL/$r/$r.zip" "$zip"
      unzip -tq "$zip" >/dev/null
      mkdir -p "$d" && unzip -q -o "$zip" -d "$d"
      manifest "$EUROC_ROOT" "$EUROC_URL/$r/$r.zip" "$zip"
      rm -f "$zip"; echo "ok $r"
    done
  done
}

check() {
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -I - "$REPO" <<'EOF'
import sys; sys.path.insert(0, sys.argv[1])
from dvslam.data import registry as r
ok = True
for ds in ('euroc', 'tumvi', 'redwood'):
    for sp in ('validation', 'test'):
        exp, got = r.expected_scenes(ds, sp), r.scenes(ds, sp)
        miss = [s for s in exp if s not in got]
        ok &= not miss
        print(f'{ds:8s} {sp:10s} {len(got)}/{len(exp)}' + (f'  MISSING {miss}' if miss else ''))
sys.exit(0 if ok else 1)
EOF
}

cmd=${1:-check}; shift || true
case $cmd in
  check) check ;;
  tumvi) fetch_tumvi "$@"; check ;;
  euroc) fetch_euroc "$@"; check ;;
  *) echo "usage: $0 {check|tumvi|euroc} [sequences...]"; exit 2 ;;
esac
