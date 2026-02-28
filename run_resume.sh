#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

timestamp=$(date +%Y%m%d_%H%M%S)

if [ -f log/demo.log ]; then
    mv log/demo.log "log/demo_${timestamp}.log"
fi

if [ -f log/bug.csv ]; then
    cp log/bug.csv "log/bug_${timestamp}.csv"
fi

# Clean up leftover temp/ dirs from a crashed run
find target_dataset -type d -name temp -exec rm -rf {} + 2>/dev/null || true

exec python main.py --resume "$@"
