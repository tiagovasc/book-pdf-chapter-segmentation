#!/usr/bin/env bash
# ============================================================
#  Book Chapter Summarizer - Batch Runner (Linux / macOS)
#
#  Usage:
#    ./run_batch.sh                   Process every .pdf inside the "pdfs" folder
#    ./run_batch.sh path/to/a.pdf     Process a single PDF
#    ./run_batch.sh one.pdf two.pdf   Process each PDF listed on the command line
#
#  Output goes to $OUTPUT_BASE_DIR (default: ./output). Each book lands in
#  output/[BookName_timestamp]/.
# ============================================================

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/book_chapter_summarizer.py"

if [[ ! -f "$SCRIPT" ]]; then
    echo "[ERROR] Script not found: $SCRIPT"
    exit 1
fi

export OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-$SCRIPT_DIR/output}"
mkdir -p "$OUTPUT_BASE_DIR"

if [[ -z "${GEMINI_API_KEY:-}" ]]; then
    echo "[WARN] GEMINI_API_KEY is not set in the environment."
    echo "       export GEMINI_API_KEY=your_key_here"
    echo "       or put it in a .env file next to the script (see .env.example)."
    echo
fi

# Collect PDFs from args or from ./pdfs
declare -a PDFS
if [[ $# -gt 0 ]]; then
    PDFS=("$@")
else
    INBOX="$SCRIPT_DIR/pdfs"
    if [[ ! -d "$INBOX" ]]; then
        echo "[ERROR] No PDF paths given and default inbox not found: $INBOX"
        echo "        Create a 'pdfs' folder next to this script and drop PDFs there,"
        echo "        or pass PDF paths as arguments."
        exit 1
    fi
    shopt -s nullglob
    PDFS=("$INBOX"/*.pdf)
    shopt -u nullglob
fi

TOTAL=${#PDFS[@]}
SUCCESS=0
FAIL=0

for i in "${!PDFS[@]}"; do
    PDF="${PDFS[$i]}"
    NUM=$((i + 1))
    echo
    echo "============================================================"
    echo "  [$NUM/$TOTAL] Processing: $PDF"
    echo "============================================================"
    echo
    if [[ ! -f "$PDF" ]]; then
        echo "  [SKIP] File not found: $PDF"
        FAIL=$((FAIL + 1))
        continue
    fi
    if python3 "$SCRIPT" "$PDF"; then
        echo "  [OK] Finished $PDF"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "  [FAIL] Exit code $? on $PDF"
        FAIL=$((FAIL + 1))
    fi
done

echo
echo "============================================================"
echo "  BATCH COMPLETE"
echo "  Succeeded: $SUCCESS / $TOTAL"
echo "  Failed:    $FAIL / $TOTAL"
echo "  Output:    $OUTPUT_BASE_DIR"
echo "============================================================"
