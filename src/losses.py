"""
Loss functions and evaluation metrics for RNA 3D structure prediction.

Includes:
- FAPE (Frame Aligned Point Error) loss
- Distance matrix loss
- TM-score computation
- Combined training loss
"""

import torch
import torch.nn.functional as F
import numpy as np


def compute_distance_matrix(coords: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise distance matrix from coordinates.

    Args:
        coords: (B, L, 3)
    Returns:
        dist: (B, L, L) pairwise distances
    """
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B, L, L, 3)
    return torch.sqrt((diff ** 2).sum(-1) + 1e-8)


def distance_matrix_loss(pred_coords: torch.Tensor, true_coords: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
    """
    Loss based on pairwise distance matrices.
    More rotationally invariant than direct coordinate loss.

    Args:
        pred_coords: (B, L, 3)
        true_coords: (B, L, 3)
        mask: (B, L) boolean mask
    """
    pred_dist = compute_distance_matrix(pred_coords)
    true_dist = compute_distance_matrix(true_coords)

    # Mask for valid pairs
    pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # (B, L, L)
    pair_mask = pair_mask.float()

    # L1 loss on distances
    loss = (pred_dist - true_dist).abs() * pair_mask
    return loss.sum() / (pair_mask.sum() + 1e-8)


def fape_loss(pred_coords: torch.Tensor, true_coords: torch.Tensor,
              mask: torch.Tensor, clamp_distance: float = 10.0) -> torch.Tensor:
    """
    Frame Aligned Point Error (simplified).

    Computes RMSD-like loss with clamping for robustness.

    Args:
        pred_coords: (B, L, 3) predicted coordinates
        true_coords: (B, L, 3) true coordinates
        mask: (B, L) boolean mask
        clamp_distance: maximum distance to clamp errors
    """
    # Center both coordinate sets (translation invariance)
    mask_float = mask.float().unsqueeze(-1)  # (B, L, 1)
    num_valid = mask_float.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1, 1)

    pred_center = (pred_coords * mask_float).sum(dim=1, keepdim=True) / num_valid
    true_center = (true_coords * mask_float).sum(dim=1, keepdim=True) / num_valid

    pred_centered = pred_coords - pred_center
    true_centered = true_coords - true_center

    # Per-residue distance
    diff = pred_centered - true_centered
    per_residue_dist = torch.sqrt((diff ** 2).sum(-1) + 1e-8)  # (B, L)

    # Clamp
    per_residue_dist = torch.clamp(per_residue_dist, max=clamp_distance)

    # Masked mean
    loss = (per_residue_dist * mask.float()).sum() / (mask.float().sum() + 1e-8)
    return loss


def kabsch_align(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """
    Align pred to true using Kabsch algorithm (optimal rotation).

    Args:
        pred: (N, 3) predicted coordinates
        true: (N, 3) true coordinates
    Returns:
        aligned: (N, 3) aligned predicted coordinates
    """
    # Center
    pred_center = pred.mean(axis=0)
    true_center = true.mean(axis=0)
    pred_c = pred - pred_center
    true_c = true - true_center

    # Covariance matrix
    H = pred_c.T @ true_c
    if not np.isfinite(H).all():
        return pred

    # SVD
    try:
        U, S, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return pred

    # Correct rotation for reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])

    R = Vt.T @ sign_matrix @ U.T

    aligned = pred_c @ R.T + true_center
    return aligned


def compute_tm_score(pred_coords: np.ndarray, true_coords: np.ndarray) -> float:
    """
    Compute TM-score between predicted and true structures.

    TM-score is length-normalized and ranges from 0 to 1.
    Score > 0.45 generally indicates correct global fold.

    Args:
        pred_coords: (N, 3) predicted C1' coordinates
        true_coords: (N, 3) true C1' coordinates
    Returns:
        TM-score (float)
    """
    L = len(true_coords)
    if L == 0:
        return 0.0
    if not np.isfinite(pred_coords).all():
        return 0.0

    # Length-dependent normalization factor
    d0 = 0.6 * (L - 0.5) ** (1.0 / 3.0) - 2.5
    d0 = max(d0, 0.5)

    # Align structures
    aligned = kabsch_align(pred_coords, true_coords)

    # Compute per-residue distances
    dists = np.sqrt(((aligned - true_coords) ** 2).sum(axis=1))

    # TM-score formula
    tm = (1.0 / (1.0 + (dists / d0) ** 2)).sum() / L

    return float(tm)


def best_of_n_tm_score(pred_coords_list: list[np.ndarray],
                       true_coords: np.ndarray) -> float:
    """
    Compute best-of-N TM-score (competition metric).

    Args:
        pred_coords_list: list of N predicted coordinate arrays
        true_coords: true coordinates
    Returns:
        Best TM-score among the N predictions
    """
    scores = [compute_tm_score(pred, true_coords) for pred in pred_coords_list]
    return max(scores)


class CombinedLoss(torch.nn.Module):
    """Combined training loss for RNA structure prediction."""

    def __init__(self, fape_weight: float = 1.0, dist_weight: float = 0.5,
                 confidence_weight: float = 0.1):
        super().__init__()
        self.fape_weight = fape_weight
        self.dist_weight = dist_weight
        self.confidence_weight = confidence_weight

    def forward(self, pred: dict, true_coords: torch.Tensor,
                mask: torch.Tensor) -> dict:
        """
        Args:
            pred: dict with 'coords' (B, L, N, 3) and 'confidence' (B, L, N)
            true_coords: (B, L, 3)
            mask: (B, L) boolean mask
        """
        num_preds = pred["coords"].shape[2]
        total_fape = 0.0
        total_dist = 0.0

        per_pred_losses = []
        for i in range(num_preds):
            pred_i = pred["coords"][:, :, i, :]  # (B, L, 3)
            f = fape_loss(pred_i, true_coords, mask)
            d = distance_matrix_loss(pred_i, true_coords, mask)
            total_fape += f
            total_dist += d
            per_pred_losses.append((f + d).detach())

        total_fape /= num_preds
        total_dist /= num_preds

        # Confidence loss: confidence should correlate with quality
        confidence_loss = torch.tensor(0.0, device=true_coords.device)
        if self.confidence_weight > 0:
            per_pred_losses_t = torch.stack(per_pred_losses)  # (N,)
            # Normalize to [0, 1] as targets for confidence
            with torch.no_grad():
                quality = 1.0 / (1.0 + per_pred_losses_t)
                quality = quality / (quality.max() + 1e-8)
                # Expand to match confidence shape
                quality_target = quality.unsqueeze(0).unsqueeze(0).expand_as(
                    pred["confidence"]
                )
            confidence_loss = F.mse_loss(
                pred["confidence"] * mask.unsqueeze(-1).float(),
                quality_target * mask.unsqueeze(-1).float(),
            )

        total_loss = (
            self.fape_weight * total_fape
            + self.dist_weight * total_dist
            + self.confidence_weight * confidence_loss
        )

        return {
            "loss": total_loss,
            "fape_loss": total_fape,
            "dist_loss": total_dist,
            "confidence_loss": confidence_loss,
        }
