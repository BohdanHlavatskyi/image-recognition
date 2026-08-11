#!/usr/bin/env bash
set -euo pipefail

# Simple dataset download helper for UAV datasets
# This script does NOT automatically download the full DOTA/VisDrone releases (they are large).
# Instead it prepares directories and offers commands to fetch sample subsets or full archives.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="$ROOT_DIR/data/raw"
mkdir -p "$RAW_DIR"

echo "Data root: $RAW_DIR"

usage(){
  echo "Usage: $0 [visdrone|dota|uavdt|all] [--sample]"
  echo "  --sample: download a small sample (when available) instead of full archives"
  exit 1
}

if [ $# -lt 1 ]; then
  usage
fi

TARGET="$1"
SAMPLE=false
if [ "${2-}" = "--sample" ]; then SAMPLE=true; fi

if [[ "$TARGET" == "visdrone" || "$TARGET" == "all" ]]; then
  echo "Preparing VisDrone download..."
  mkdir -p "$RAW_DIR/VisDrone"
  if $SAMPLE; then
    echo "Downloading a small sample of VisDrone (sample images)..."
    # VisDrone does not publish a tiny public sample archive; as a fallback, download a few example images from the VisDrone GitHub repo.
    curl -L -o "$RAW_DIR/VisDrone/sample_list.txt" https://raw.githubusercontent.com/VisDrone/VisDrone-Dataset/master/support/sample_list.txt || true
    echo "Sample manifest saved to $RAW_DIR/VisDrone/sample_list.txt"
  else
    echo "Full VisDrone archives are large (GBs). Visit https://github.com/VisDrone/VisDrone-Dataset for official download links."
    echo "You can manually download the challenge zip files into $RAW_DIR/VisDrone and then run the converter."
  fi
fi

if [[ "$TARGET" == "dota" || "$TARGET" == "all" ]]; then
  echo "Preparing DOTA download..."
  mkdir -p "$RAW_DIR/DOTA"
  if $SAMPLE; then
    echo "DOTA full dataset is large; no official tiny sample. Consider downloading a small subset manually."
  else
    echo "DOTA full dataset is large (tens of GB). Get links at https://captain-whu.github.io/DOTA/dataset.html and place archives into $RAW_DIR/DOTA"
  fi
fi

if [[ "$TARGET" == "uavdt" || "$TARGET" == "all" ]]; then
  echo "Preparing UAVDT download..."
  mkdir -p "$RAW_DIR/UAVDT"
  if $SAMPLE; then
    echo "UAVDT sample images are not packaged publicly; please follow UAVDT instructions at https://sites.google.com/site/dstunofficial/ for downloads."
  else
    echo "UAVDT is available from the original paper/dataset site; place archives into $RAW_DIR/UAVDT and run the converter."
  fi
fi

echo "Download preparation complete. Read scripts/README_datasets.md for next steps."
