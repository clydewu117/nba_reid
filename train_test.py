#!/usr/bin/env python
"""
Basketball Video ReID - Training and Testing Script (with Parameter & Gradient Inspection)
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import wandb

# Import project modules
from config.defaults import get_cfg_defaults
from data.dataloader import build_dataloader
from models.build import build_model
from loss import make_loss
from metrics import R1_mAP_eval


# ----------------------------
# Utility Classes
# ----------------------------
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# Train One Epoch (with gradient diagnostics)
# ----------------------------
def train_one_epoch(
    cfg, model, train_loader, optimizer, loss_fn, scaler, epoch, device
):
    model.train()
    batch_time, data_time = AverageMeter(), AverageMeter()
    losses, id_losses, triplet_losses = AverageMeter(), AverageMeter(), AverageMeter()
    end = time.time()

    for batch_idx, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        videos = batch["video"].to(device)
        pids = batch["pid"].to(device)

        with autocast():
            outputs = model(videos, label=pids)
            loss, loss_dict = loss_fn(
                outputs["cls_score"], outputs["global_feat"], outputs["feat"], pids
            )

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        if cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            )

        scaler.step(optimizer)
        scaler.update()

        losses.update(loss.item(), videos.size(0))
        id_losses.update(loss_dict["id_loss"].item(), videos.size(0))
        if "triplet_loss" in loss_dict:
            triplet_losses.update(loss_dict["triplet_loss"].item(), videos.size(0))

        lr = optimizer.param_groups[0]["lr"]

        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_idx + 1) % cfg.SOLVER.LOG_PERIOD == 0:
            print(
                f"Epoch: [{epoch}/{cfg.SOLVER.MAX_EPOCHS}][{batch_idx + 1}/{len(train_loader)}] "
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                f"Loss {losses.val:.4f} ({losses.avg:.4f})"
            )
            wandb.log(
                {
                    "train/batch_loss": losses.val,
                    "train/batch_id_loss": id_losses.val,
                    "train/batch_triplet_loss": triplet_losses.val,
                    "train/lr": lr,
                    "epoch": epoch + batch_idx / len(train_loader),
                }
            )

    return {
        "loss": losses.avg,
        "id_loss": id_losses.avg,
        "triplet_loss": triplet_losses.avg,
        "batch_time": batch_time.avg,
    }


# ----------------------------
# Evaluation
# ----------------------------
@torch.no_grad()
def test(cfg, model, query_loader, gallery_loader, device, epoch=None):
    model.eval()
    evaluator = R1_mAP_eval(len(query_loader.dataset), max_rank=50, feat_norm=True)
    evaluator.reset()

    print(f"Extracting query features ({len(query_loader.dataset)})...")
    for batch in query_loader:
        videos = batch["video"].to(device)
        pids = batch["pid"].to(device)
        with autocast():
            feat = model(videos)
        evaluator.update((feat, pids))

    print(f"Extracting gallery features ({len(gallery_loader.dataset)})...")
    for batch in gallery_loader:
        videos = batch["video"].to(device)
        pids = batch["pid"].to(device)
        with autocast():
            feat = model(videos)
        evaluator.update((feat, pids))

    cmc, mAP = evaluator.compute()

    # Extract Rank-1, Rank-5, Rank-10
    rank1 = cmc[0]
    rank5 = cmc[4] if len(cmc) > 4 else cmc[-1]
    rank10 = cmc[9] if len(cmc) > 9 else cmc[-1]

    print(f"\n{'='*60}")
    print(f"Validation Results:")
    print(f"  mAP     : {mAP:.2%}")
    print(f"  Rank-1  : {rank1:.2%}")
    print(f"  Rank-5  : {rank5:.2%}")
    print(f"  Rank-10 : {rank10:.2%}")
    print(f"{'='*60}\n")

    if epoch is not None:
        wandb.log(
            {
                "test/mAP": mAP,
                "test/rank1": rank1,
                "test/rank5": rank5,
                "test/rank10": rank10,
                "epoch": epoch,
            }
        )

    return rank1, mAP, rank5, rank10


# ----------------------------
# Split dataset into query/gallery
# ----------------------------
def split_query_gallery(dataset, query_ratio=0.5, seed=42, min_samples=2):
    """
    Split dataset into query and gallery sets for ReID evaluation.

    Args:
        dataset: The dataset to split
        query_ratio: Ratio of samples to use as query (default: 0.5)
        seed: Random seed for reproducibility (default: 42)
        min_samples: Minimum samples required per ID (default: 2)

    Returns:
        query_indices: List of indices for query set
        gallery_indices: List of indices for gallery set
    """
    np.random.seed(seed)
    pid_to_indices = {}

    # Group indices by person ID
    for idx in range(len(dataset)):
        sample = dataset[idx]
        pid = sample["pid"]
        pid_to_indices.setdefault(pid, []).append(idx)

    query_indices, gallery_indices = [], []
    excluded_pids = []

    for pid, indices in pid_to_indices.items():
        # Exclude IDs with insufficient samples
        if len(indices) < min_samples:
            excluded_pids.append(pid)
            continue

        # Randomly split samples for this ID
        np.random.shuffle(indices)
        split_point = max(1, int(len(indices) * query_ratio))
        # Ensure gallery has at least 1 sample
        split_point = min(split_point, len(indices) - 1)

        query_indices.extend(indices[:split_point])
        gallery_indices.extend(indices[split_point:])

    # Print statistics
    print(f"\n{'='*60}")
    print(f"Query/Gallery Split Summary:")
    print(f"  Query samples    : {len(query_indices)}")
    print(f"  Gallery samples  : {len(gallery_indices)}")
    print(f"  Valid PIDs       : {len(pid_to_indices) - len(excluded_pids)}")
    print(f"  Excluded PIDs    : {len(excluded_pids)} (< {min_samples} samples)")
    print(f"{'='*60}")

    # Verify no overlap
    overlap = set(query_indices) & set(gallery_indices)
    assert (
        len(overlap) == 0
    ), f"Error: {len(overlap)} samples overlap between query and gallery!"

    return query_indices, gallery_indices


# ----------------------------
# Train Entry
# ----------------------------
def train(cfg):
    device = torch.device(
        f"cuda:{cfg.GPU_IDS[0]}" if torch.cuda.is_available() else "cpu"
    )
    set_seed(cfg.SEED)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
        f.write(cfg.dump())

    print("=" * 80)
    print(f"Training Configuration:\n  Output Dir: {cfg.OUTPUT_DIR}")
    print("=" * 80)

    train_loader, num_classes = build_dataloader(cfg, is_train=True)
    test_loader, _ = build_dataloader(cfg, is_train=False)

    cfg.defrost()
    cfg.MODEL.NUM_CLASSES = num_classes
    cfg.freeze()

    print(
        f"\nDataset Statistics:\n  Train videos: {len(train_loader.dataset)}  Test videos: {len(test_loader.dataset)}"
    )

    print("Loading Data")
    dataset = test_loader.dataset
    q_idx, g_idx = split_query_gallery(dataset, 0.5, cfg.SEED)
    q_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, q_idx),
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATA.NUM_WORKERS,
    )
    g_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, g_idx),
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATA.NUM_WORKERS,
    )

    print("Initializing WandB")
    wandb.init(
        project=f"nba-reid",
        name=f"{cfg.MODEL.ARCH}_{cfg.DATA.VIDEO_TYPE}_{cfg.DATA.SHOT_TYPE}",
        config={"num_classes": num_classes, "num_frames": cfg.DATA.NUM_FRAMES},
        dir=cfg.OUTPUT_DIR,
    )

    model = build_model(cfg).to(device)
    # Debug: confirm the actual model class built
    print("[DEBUG] BUILT MODEL:", type(model).__name__)
    wandb.watch(model, log="all", log_freq=100)
    loss_fn = make_loss(cfg, num_classes)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.SOLVER.MAX_EPOCHS, eta_min=1e-6
    )
    scaler = GradScaler()
    best_rank1, best_mAP, best_rank5, best_rank10 = 0.0, 0.0, 0.0, 0.0

    print("\n" + "=" * 80)
    print("Start training...")
    print("=" * 80)

    for epoch in range(1, cfg.SOLVER.MAX_EPOCHS + 1):
        print(f"\n{'='*80}\nEpoch [{epoch}/{cfg.SOLVER.MAX_EPOCHS}]\n{'='*80}")
        train_metrics = train_one_epoch(
            cfg, model, train_loader, optimizer, loss_fn, scaler, epoch, device
        )
        print(f"Epoch {epoch} Summary: Loss={train_metrics['loss']:.4f}")

        scheduler.step()
        if epoch % cfg.SOLVER.EVAL_PERIOD == 0:
            rank1, mAP, rank5, rank10 = test(
                cfg, model, q_loader, g_loader, device, epoch
            )
            if rank1 > best_rank1:
                best_rank1, best_mAP, best_rank5, best_rank10 = (
                    rank1,
                    mAP,
                    rank5,
                    rank10,
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "rank1": rank1,
                        "mAP": mAP,
                        "rank5": rank5,
                        "rank10": rank10,
                    },
                    os.path.join(cfg.OUTPUT_DIR, "best_model.pth"),
                )
                print(f"✓ Saved best model at epoch {epoch}")
                print(
                    f"  Rank-1: {rank1:.2%} | Rank-5: {rank5:.2%} | Rank-10: {rank10:.2%} | mAP: {mAP:.2%}"
                )

    print("\n" + "=" * 80)
    print("Training Completed! Best Results:")
    print(f"  Rank-1  : {best_rank1:.2%}")
    print(f"  Rank-5  : {best_rank5:.2%}")
    print(f"  Rank-10 : {best_rank10:.2%}")
    print(f"  mAP     : {best_mAP:.2%}")
    print("=" * 80)
    wandb.finish()


# ----------------------------
# Main Entry
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    cfg = get_cfg_defaults()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Debug: show which model is actually selected from config
    print("[DEBUG] MODEL.NAME:", cfg.MODEL.NAME, "MODEL.MODEL_NAME:", cfg.MODEL.MODEL_NAME, "MODEL.ARCH:", cfg.MODEL.ARCH)

    if args.mode == "train":
        train(cfg)
    else:
        print("Test mode not included in this minimal example.")


if __name__ == "__main__":
    main()
