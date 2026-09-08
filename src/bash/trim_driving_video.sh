#!/usr/bin/env bash
# One-time asset prep — NOT part of the runtime path. Trims a driving
# video down to an Apple-Live-Photo-scale burst (~1.5s) before the first
# LivePortrait call builds and caches its .pkl motion template.
#
# Usage: bash src/image_gen/trim_driving_video.sh assets/d19.mp4 1.5
set -euo pipefail

SRC="$1"
DURATION="${2:-1.5}"  # seconds; 1.5s ~ Apple Live Photo scale, try 0.75s for half that
OUT="${SRC%.*}_short.mp4"

ffmpeg -y -i "${SRC}" -t "${DURATION}" -c copy "${OUT}"
echo "Trimmed clip: ${OUT}"
echo "First call against this file will build+cache ${OUT%.*}.pkl automatically."
echo "Point --driving at that .pkl for all calls after the first."
