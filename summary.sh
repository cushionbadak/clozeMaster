#!/usr/bin/env bash
# Summarize all runs: elapsed time, files processed, bugs found.
set -euo pipefail

cd "$(dirname "$0")"

logdir="./log"

if [ ! -d "$logdir" ]; then
    echo "Log directory not found: $logdir" >&2
    exit 1
fi

parse_ts() {
    date -j -f "%Y-%m-%d %H:%M:%S" "$1" +%s 2>/dev/null || date -d "$1" +%s
}

fmt_duration() {
    local diff=$1
    printf "%dh %02dm %02ds" $((diff/3600)) $(((diff%3600)/60)) $((diff%60))
}

total_time=0
total_files=0
total_bugs=0
run_count=0

printf "%-32s  %-21s  %-14s  %6s  %4s\n" "Log" "Period" "Elapsed" "Files" "Bugs"
printf "%-32s  %-21s  %-14s  %6s  %4s\n" "---" "------" "-------" "-----" "----"

for log in "$logdir"/demo*.log; do
    [ -f "$log" ] || continue

    first=$(head -1 "$log" | grep -oE '^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}' || true)
    last=$(tail -1 "$log" | grep -oE '^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}' || true)

    files=$(grep -c 'filename:' "$log" 2>/dev/null || echo 0)
    bugs=$(grep -c 'status:ice\|status:crash\|status:mem err\|status:timeout' "$log" 2>/dev/null || echo 0)

    if [ -z "$first" ] || [ -z "$last" ]; then
        printf "%-32s  (no timestamps)\n" "$(basename "$log")"
        continue
    fi

    t1=$(parse_ts "$first")
    t2=$(parse_ts "$last")
    diff=$((t2 - t1))

    total_time=$((total_time + diff))
    total_files=$((total_files + files))
    total_bugs=$((total_bugs + bugs))
    run_count=$((run_count + 1))

    period="${first%:*} ~ ${last%:*}"
    printf "%-32s  %-21s  %-14s  %6d  %4d\n" "$(basename "$log")" "$period" "$(fmt_duration $diff)" "$files" "$bugs"
done

if [ "$run_count" -eq 0 ]; then
    echo "No log files with valid timestamps found."
    exit 1
fi

printf "%-32s  %-21s  %-14s  %6s  %4s\n" "---" "" "-------" "-----" "----"
printf "%-32s  %-21s  %-14s  %6d  %4d\n" "Total ($run_count runs)" "" "$(fmt_duration $total_time)" "$total_files" "$total_bugs"
