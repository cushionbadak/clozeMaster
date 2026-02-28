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

exec python main.py --resume "$@"
