#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[ERROR] $PYTHON_BIN not found. Set PYTHON_BIN environment variable." >&2
    exit 1
fi

if [ ! -f "requirements.txt" ]; then
    echo "[ERROR] requirements.txt not found in $SCRIPT_DIR" >&2
    exit 1
fi

if [ ! -d ".venv" ]; then
    "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir -r requirements.txt

mkdir -p config
mkdir -p out
mkdir -p visual_out

if [ ! -f "config/input.txt" ]; then
    echo "$SCRIPT_DIR/input" > config/input.txt
fi

for f in build.sh run.sh run_visual.sh; do
    if [ -f "$f" ]; then
        chmod +x "$f"
    fi
done

echo "[OK] build complete"
echo "     venv:      .venv"
echo "     config:    ./config"
echo "     out:       ./out"
echo "     visual:    ./visual_out"