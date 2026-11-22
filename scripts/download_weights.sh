#!/usr/bin/env bash
# Download the dvslam release weights (v0.1.0) and verify their SHA-256 checksums.
#
#   scripts/download_weights.sh            # -> weights/dvslam/ (or $DVSLAM_WEIGHTS)
#
# Environment variables
#   DVSLAM_WEIGHTS      destination directory (default: <repo>/weights/dvslam)
#   DVSLAM_WEIGHTS_URL  base URL of the release files (default: the GitHub release v0.1.0 below); a local mirror,
#                       file:///path/to/release/v0.1.0 or http://host:port/ works too
# Files: dvslam_depth_hints_core.pt, dvslam_depth_core.pt, dvslam_s2_photonet_v4.pt, SHA256SUMS, MODEL_CARD.md.
# Files already present with the right checksum are not downloaded again.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE=${DVSLAM_WEIGHTS_URL:-https://github.com/chansoopark98/Deep-Visual-SLAM/releases/download/v0.1.0}
DEST=${DVSLAM_WEIGHTS:-$REPO/weights/dvslam}
FILES=(dvslam_depth_hints_core.pt dvslam_depth_core.pt dvslam_s2_photonet_v4.pt)

if command -v sha256sum > /dev/null; then SHA=(sha256sum); else SHA=(shasum -a 256); fi
fetch() {  # fetch <name>: BASE/<name> -> DEST/<name> (via .part)
  local url=${BASE%/}/$1
  echo "download $url"
  if command -v curl > /dev/null; then
    curl -fSL --retry 3 -o "$DEST/$1.part" "$url"
  else
    wget -q -O "$DEST/$1.part" "$url"
  fi
  mv "$DEST/$1.part" "$DEST/$1"
}

mkdir -p "$DEST"
fetch SHA256SUMS
fetch MODEL_CARD.md
for f in "${FILES[@]}"; do
  want=$(awk -v f="$f" '$2 == f || $2 == "*"f {print $1}' "$DEST/SHA256SUMS")
  [ -n "$want" ] || { echo "FAIL: $f is not listed in SHA256SUMS"; exit 1; }
  if [ -f "$DEST/$f" ] && [ "$("${SHA[@]}" "$DEST/$f" | cut -d' ' -f1)" = "$want" ]; then
    echo "ok (present) $f"
    continue
  fi
  fetch "$f"
done
(cd "$DEST" && "${SHA[@]}" -c SHA256SUMS) || { echo "FAIL: checksum mismatch in $DEST"; exit 1; }
echo "weights verified in $DEST"
