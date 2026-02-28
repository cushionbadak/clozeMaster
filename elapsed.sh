#!/usr/bin/env bash
# Show elapsed time of a running (or finished) main.py session from its log.
set -euo pipefail

log="${1:-./log/demo.log}"

if [ ! -f "$log" ]; then
    echo "Log not found: $log" >&2
    exit 1
fi

first=$(head -1 "$log" | grep -oE '^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}')
last=$(tail -1 "$log" | grep -oE '^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}')

if [ -z "$first" ] || [ -z "$last" ]; then
    echo "Could not parse timestamps from $log" >&2
    exit 1
fi

t1=$(date -j -f "%Y-%m-%d %H:%M:%S" "$first" +%s 2>/dev/null || date -d "$first" +%s)
t2=$(date -j -f "%Y-%m-%d %H:%M:%S" "$last" +%s 2>/dev/null || date -d "$last" +%s)
diff=$((t2 - t1))

h=$((diff / 3600))
m=$(( (diff % 3600) / 60 ))
s=$((diff % 60))

printf "Start : %s\nLatest: %s\nElapsed: %dh %02dm %02ds\n" "$first" "$last" "$h" "$m" "$s"
