#!/bin/bash
# Train RNA 3D Folding Model
# Usage: bash scripts/run_training.sh [config_path]

set -e

CONFIG=${1:-configs/default.yaml}

echo "========================================"
echo "Stanford RNA 3D Folding Part 2 - Training"
echo "========================================"
echo "Config: $CONFIG"
echo ""

# Check for GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Create directories
mkdir -p checkpoints data/train/structures

# Run training
python3 -m src.train --config "$CONFIG"
