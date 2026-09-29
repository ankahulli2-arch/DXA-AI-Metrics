#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

OUT_DIR=""

if [ "$1" = "save" ]; then
  OUT_DIR="${2:-visual_out}"
elif [ "$1" = "--save" ]; then
  OUT_DIR="${2:-visual_out}"
elif [ $# -eq 0 ]; then
  OUT_DIR=""
else
  echo "Usage:"
  echo "  ./run_visual.sh                    # интерактивный просмотр"
  echo "  ./run_visual.sh save [OUT_DIR]     # сохранить PNG в OUT_DIR"
  echo "  ./run_visual.sh --save [OUT_DIR]   # то же самое"
  exit 1
fi

if [ -n "$OUT_DIR" ]; then
  mkdir -p "$OUT_DIR"
  python Visual.py --save "$OUT_DIR"
else
  python Visual.py
fi
