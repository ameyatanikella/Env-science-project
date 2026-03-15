"""
Training pipeline for RNA 3D structure prediction.

Supports:
- Mixed precision training (AMP)
- Learning rate warmup + cosine decay
- Gradient clipping
- TM-score evaluation during validation
- Checkpointing
"""

import os
import time
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .model import RNAFoldModel
from .dataset import get_train_dataloader, RNATrainDataset
from .losses import CombinedLoss, compute_tm_score


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warmup then cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion,
                    scaler, device, grad_clip, epoch):
    model.train()
    total_loss = 0
    total_fape = 0
    total_dist = 0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        tokens = batch["tokens"].to(device)
        mask = batch["mask"].to(device)
        coords = batch["coords"].to(device)

        optimizer.zero_grad()

        with autocast(device_type="cuda", enabled=scaler.is_enabled()):
            pred = model(tokens, mask)
            loss_dict = criterion(pred, coords, mask)
            loss = loss_dict["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        total_fape += loss_dict["fape_loss"].item()
        total_dist += loss_dict["dist_loss"].item()
        n_batches += 1

        pbar.set_postfix({
            "loss": f"{total_loss / n_batches:.4f}",
            "fape": f"{total_fape / n_batches:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        })

    return {
        "loss": total_loss / max(n_batches, 1),
        "fape_loss": total_fape / max(n_batches, 1),
        "dist_loss": total_dist / max(n_batches, 1),
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_tm = 0
    n_batches = 0
    n_tm = 0

    for batch in tqdm(dataloader, desc="Validating"):
        tokens = batch["tokens"].to(device)
        mask = batch["mask"].to(device)
        coords = batch["coords"].to(device)

        pred = model(tokens, mask)
        loss_dict = criterion(pred, coords, mask)
        total_loss += loss_dict["loss"].item()
        n_batches += 1

        # Compute TM-score for each sample
        pred_coords = pred["coords"].cpu().numpy()  # (B, L, N, 3)
        true_coords = coords.cpu().numpy()
        masks = mask.cpu().numpy()

        for b in range(pred_coords.shape[0]):
            seq_len = masks[b].sum().astype(int)
            if seq_len < 3:
                continue
            true_c = true_coords[b, :seq_len]
            if np.abs(true_c).sum() < 1e-6:
                continue

            # Best of 5 TM-score
            pred_list = [pred_coords[b, :seq_len, i] for i in range(pred_coords.shape[2])]
            tm = max(compute_tm_score(p, true_c) for p in pred_list)
            total_tm += tm
            n_tm += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "tm_score": total_tm / max(n_tm, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train RNA 3D Folding Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["training"]["seed"])
    os.makedirs(cfg["training"]["save_dir"], exist_ok=True)

    print(f"Device: {device}")

    # Data
    train_loader = get_train_dataloader(
        cfg["data"]["train_dir"],
        max_seq_len=cfg["data"]["max_seq_len"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    # For validation, use 10% split
    full_dataset = train_loader.dataset
    n_val = max(1, len(full_dataset) // 10)
    n_train = len(full_dataset) - n_val
    train_subset, val_subset = torch.utils.data.random_split(full_dataset, [n_train, n_val])

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # Model
    model = RNAFoldModel(
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"],
        d_ff=cfg["model"]["d_ff"],
        dropout=cfg["model"]["dropout"],
        num_predictions=cfg["model"]["num_predictions"],
        max_seq_len=cfg["data"]["max_seq_len"],
    ).to(device)

    print(f"Model parameters: {model.count_parameters():,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    total_steps = cfg["training"]["num_epochs"] * len(train_loader)
    scheduler = get_lr_scheduler(optimizer, cfg["training"]["warmup_steps"], total_steps)

    criterion = CombinedLoss()
    scaler = GradScaler(enabled=device.type == "cuda")

    start_epoch = 0
    best_tm = 0.0

    # Resume
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_tm = ckpt.get("best_tm", 0.0)
        print(f"Resumed from epoch {start_epoch}, best TM: {best_tm:.4f}")

    # Train
    for epoch in range(start_epoch, cfg["training"]["num_epochs"]):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            scaler, device, cfg["training"]["grad_clip"], epoch,
        )

        val_metrics = validate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_tm={val_metrics['tm_score']:.4f} "
            f"time={elapsed:.1f}s"
        )

        # Save checkpoint
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_tm": best_tm,
            "config": cfg,
        }

        if val_metrics["tm_score"] > best_tm:
            best_tm = val_metrics["tm_score"]
            ckpt["best_tm"] = best_tm
            torch.save(ckpt, os.path.join(cfg["training"]["save_dir"], "best_model.pt"))
            print(f"  New best TM-score: {best_tm:.4f}")

        torch.save(ckpt, os.path.join(cfg["training"]["save_dir"], "last_model.pt"))

    print(f"Training complete. Best TM-score: {best_tm:.4f}")


if __name__ == "__main__":
    main()
