#!/bin/bash
# Generate submission for RNA 3D Folding competition
# Usage: bash scripts/run_inference.sh [checkpoint] [test_csv]

set -e

CHECKPOINT=${1:-checkpoints/best_model.pt}
TEST_CSV=${2:-data/test_sequences.csv}

echo "========================================"
echo "Stanford RNA 3D Folding Part 2 - Inference"
echo "========================================"
echo "Checkpoint: $CHECKPOINT"
echo "Test CSV: $TEST_CSV"
echo ""

python3 -m src.inference \
    --checkpoint "$CHECKPOINT" \
    --test_csv "$TEST_CSV" \
    --output submission.csv

echo ""
echo "Done! Submission saved to submission.csv"
