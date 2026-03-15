"""
Inference pipeline for generating competition submissions.

Generates submission.csv with 5 predicted 3D structures per RNA target.
Format: ID, resname, resid, x_1, y_1, z_1, ..., x_5, y_5, z_5
"""

import os
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .model import RNAFoldModel
from .dataset import RNATestDataset, encode_sequence


def load_model(checkpoint_path: str, config: dict, device: torch.device) -> RNAFoldModel:
    """Load trained model from checkpoint."""
    model = RNAFoldModel(
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        n_layers=config["model"]["n_layers"],
        d_ff=config["model"]["d_ff"],
        dropout=0.0,  # No dropout at inference
        num_predictions=config["model"]["num_predictions"],
        max_seq_len=config["data"]["max_seq_len"],
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def predict_structure(model: RNAFoldModel, sequence: str,
                      max_seq_len: int, device: torch.device) -> dict:
    """
    Predict 3D structure for a single RNA sequence.

    Returns dict with coords (L, 5, 3) and confidence (L, 5).
    """
    tokens = encode_sequence(sequence)
    seq_len = len(tokens)

    # Handle long sequences by chunking
    if seq_len <= max_seq_len:
        padded = tokens + [0] * (max_seq_len - seq_len)
        mask = [1] * seq_len + [0] * (max_seq_len - seq_len)

        tokens_t = torch.tensor([padded], dtype=torch.long, device=device)
        mask_t = torch.tensor([mask], dtype=torch.bool, device=device)

        pred = model(tokens_t, mask_t)
        coords = pred["coords"][0, :seq_len].cpu().numpy()  # (L, 5, 3)
        confidence = pred["confidence"][0, :seq_len].cpu().numpy()  # (L, 5)
    else:
        # Sliding window with overlap for long sequences
        window = max_seq_len
        stride = max_seq_len // 2
        coords = np.zeros((seq_len, model.num_predictions, 3), dtype=np.float32)
        weights = np.zeros((seq_len, 1, 1), dtype=np.float32)
        confidence = np.zeros((seq_len, model.num_predictions), dtype=np.float32)

        for start in range(0, seq_len, stride):
            end = min(start + window, seq_len)
            chunk = tokens[start:end]
            chunk_len = len(chunk)
            padded = chunk + [0] * (window - chunk_len)
            mask = [1] * chunk_len + [0] * (window - chunk_len)

            tokens_t = torch.tensor([padded], dtype=torch.long, device=device)
            mask_t = torch.tensor([mask], dtype=torch.bool, device=device)

            pred = model(tokens_t, mask_t)
            chunk_coords = pred["coords"][0, :chunk_len].cpu().numpy()
            chunk_conf = pred["confidence"][0, :chunk_len].cpu().numpy()

            coords[start:end] += chunk_coords
            confidence[start:end] += chunk_conf
            weights[start:end] += 1.0

            if end >= seq_len:
                break

        coords = coords / np.maximum(weights, 1e-8)
        confidence = confidence / np.maximum(weights[:, :, 0], 1e-8)

    return {"coords": coords, "confidence": confidence}


def generate_submission(model: RNAFoldModel, test_csv: str,
                        max_seq_len: int, device: torch.device,
                        output_path: str = "submission.csv"):
    """
    Generate submission.csv for the competition.

    Format: ID, resname, resid, x_1, y_1, z_1, ..., x_5, y_5, z_5
    """
    df = pd.read_csv(test_csv)
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating predictions"):
        target_id = row["target_id"]
        sequence = row["sequence"]

        result = predict_structure(model, sequence, max_seq_len, device)
        coords = result["coords"]  # (L, 5, 3)

        for resid, nuc in enumerate(sequence):
            entry = {
                "ID": f"{target_id}_{resid + 1}",
                "resname": nuc.upper(),
                "resid": resid + 1,
            }
            for pred_idx in range(5):
                entry[f"x_{pred_idx + 1}"] = round(float(coords[resid, pred_idx, 0]), 3)
                entry[f"y_{pred_idx + 1}"] = round(float(coords[resid, pred_idx, 1]), 3)
                entry[f"z_{pred_idx + 1}"] = round(float(coords[resid, pred_idx, 2]), 3)
            rows.append(entry)

    submission = pd.DataFrame(rows)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"  Total rows: {len(submission)}")
    print(f"  Targets: {len(df)}")
    return submission


def main():
    parser = argparse.ArgumentParser(description="Generate RNA 3D Folding Submission")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--output", type=str, default="submission.csv")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or cfg["inference"]["checkpoint"]
    test_csv = args.test_csv or cfg["data"]["test_csv"]

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Test CSV: {test_csv}")

    model = load_model(checkpoint, cfg, device)
    generate_submission(model, test_csv, cfg["data"]["max_seq_len"], device, args.output)


if __name__ == "__main__":
    main()
