#!/usr/bin/env bash
set -e

MODE="${1:-batch}"

case "$MODE" in
  batch)
    exec python /app/Final_sum.py
    ;;
  visual)
    shift
    exec python /app/Visual.py "$@"
    ;;
  api)
    shift
    exec python /app/api.py "$@"
    ;;
  *)
    echo "Usage: docker run dxa-ai [batch|visual|api] [args...]"
    exit 1
    ;;
esac