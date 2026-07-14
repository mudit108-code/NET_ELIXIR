#!/usr/bin/env bash
set -euo pipefail

# AIgnition 3.0 -- Probabilistic Revenue Forecasting


DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "=================================================================="
echo "AIgnition 3.0 -- Probabilistic Revenue Forecasting"
echo "  DATA_DIR    : $DATA_DIR"
echo "  MODEL_PATH  : $MODEL_PATH"
echo "  OUTPUT_PATH : $OUTPUT_PATH"
echo "=================================================================="

# 1. Ingest the (possibly replaced) data/ folder and engineer features.
#    This does NOT retrain the model -- it only prepares the held-out data
#    for the already-trained model to score, per Section 5 of the guide.
python3 src/generate_features.py \
    --data-dir "$DATA_DIR" \
    --out features.pkl

# 2. Load the pre-trained, committed model and produce probabilistic
#    revenue / ROAS forecasts at channel / campaign-type / campaign grain.
python3 src/predict.py \
    --features features.pkl \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_PATH"

echo "Done. Predictions written to $OUTPUT_PATH"
