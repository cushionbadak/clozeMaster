#!/bin/bash
set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_NAME="clozeMaster_results_${TIMESTAMP}"

echo "=== Archiving ClozeMaster results ==="

# Check if there's anything to archive
if [ ! -d "log" ] && [ ! -d "target_dataset" ]; then
    echo "Nothing to archive: log/ and target_dataset/ not found."
    exit 1
fi

# Create a wrapper directory so tar extracts into a single folder
mkdir -p "$ARCHIVE_NAME"
[ -d "log" ] && cp -r log "$ARCHIVE_NAME/"
[ -d "target_dataset" ] && cp -r target_dataset "$ARCHIVE_NAME/"

# Archive
tar -czf "${ARCHIVE_NAME}.tar.gz" "$ARCHIVE_NAME"
rm -rf "$ARCHIVE_NAME"

echo "Created: ${ARCHIVE_NAME}.tar.gz"
