#!/bin/bash
# Start the Screenshot OCR Tool
# This script runs the screenshot_ocr.py in the background

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda environment if available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate base 2>/dev/null || true
fi

echo "🚀 Starting Screenshot OCR Tool..."
echo "   Hotkey: Win + Shift + E"
echo "   Output: ~/Pictures/Screenshots_OCR/"
echo ""
echo "   Press Ctrl+C to stop"
echo ""

python3 screenshot_ocr.py
