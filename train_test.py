#!/usr/bin/env python
"""
Basketball Video ReID - Training and Testing Script
Adapted for unireid project structure
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
# Train One Epoch
# ----------------------------
def train_one_epoch(cfg, model, train_loader, optimizer, loss_fn, scaler, epoch, device):
    model.train()
    batch_time, data_time = AverageMeter(), AverageMeter()
    losses, id_losses, triplet_losses = AverageMeter(), AverageMeter(), AverageMeter()
    end = time.time()

    for batch_idx, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        # 从batch中获取数据
        videos = batch['video'].to(device)
        pids = batch['pid'].to(device)

        with autocast():
            # Model返回字典: {'cls_score', 'bn_feat', 'feat'}
            outputs = model(videos, label=pids)
            
            # Loss function接收单独的参数
            loss, loss_dict = loss_fn(
                outputs['cls_score'], 
                outputs['bn_feat'], 
                outputs['feat'], 
                pids
            )

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        if cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE)

        scaler.step(optimizer)
        scaler.update()

        # 更新统计
        losses.update(loss.item(), videos.size(0))
        id_losses.update(loss_dict['id_loss'].item(), videos.size(0))
        if 'triplet_loss' in loss_dict:
            triplet_losses.update(loss_dict['triplet_loss'].item(), videos.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        # 日志输出
        if (batch_idx + 1) % cfg.SOLVER.LOG_PERIOD == 0:
            print(f"Epoch: [{epoch}/{cfg.SOLVER.MAX_EPOCHS}][{batch_idx + 1}/{len(train_loader)}] "
                  f"Time {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                  f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                  f"Loss {losses.val:.4f} ({losses.avg:.4f}) "
                  f"ID {id_losses.val:.4f} Trip {triplet_losses.val:.4f}")
            
            wandb.log({
                'train/batch_loss': losses.val,
                'train/batch_id_loss': id_losses.val,
                'train/batch_triplet_loss': triplet_losses.val,
                'train/lr': optimizer.param_groups[0]['lr']
            })

    return {
        'loss': losses.avg,
        'id_loss': id_losses.avg,
        'triplet_loss': triplet_losses.avg,
        'batch_time': batch_time.avg
    }


# ----------------------------
# Evaluation
# ----------------------------
@torch.no_grad()
def test(cfg, model, query_loader, gallery_loader, device, epoch=None):
    """
    Test function supporting both ReID and Classification modes.
    
    For ReID: Uses query/gallery split and computes Rank-1, mAP
    For Classification: Uses full test set and computes Accuracy, F1
    """
    model.eval()
    
    # Check if in classification mode
    is_classification = getattr(cfg.DATA, 'SHOT_CLASSIFICATION', False)
    
    if is_classification:
        # Classification mode evaluation
        from metrics import ClassificationEvaluator
        evaluator = ClassificationEvaluator(num_classes=2)
        evaluator.reset()
        
        print(f"\nEvaluating classification on test set ({len(query_loader.dataset) + len(gallery_loader.dataset)} samples)...")
        
        # Process query set
        for batch in query_loader:
            videos = batch['video'].to(device)
            labels = batch['pid'].to(device)  # In classification mode, pid is actually shot_type label
            
            with autocast():
                logits, _ = model(videos, label=None, training=False)  # Get logits only
            
            evaluator.update(logits, labels)
        
        # Process gallery set
        for batch in gallery_loader:
            videos = batch['video'].to(device)
            labels = batch['pid'].to(device)
            
            with autocast():
                logits, _ = model(videos, label=None, training=False)
            
            evaluator.update(logits, labels)
        
        # Compute classification metrics
        metrics = evaluator.compute()
        acc = metrics['accuracy']
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1']
        cm = metrics['confusion_matrix']
        
        print(f"\n{'='*60}")
        print(f"Classification Results:")
        print(f"  Accuracy  : {acc:.2%}")
        print(f"  Precision : {precision:.2%}")
        print(f"  Recall    : {recall:.2%}")
        print(f"  F1-score  : {f1:.2%}")
        print(f"\nConfusion Matrix:")
        print(f"  {cm}")
        print(f"  (0: freethrow, 1: 3pt)")
        print(f"{'='*60}\n")
        
        if epoch is not None:
            wandb.log({
                'test/accuracy': acc,
                'test/precision': precision,
                'test/recall': recall,
                'test/f1': f1,
                'epoch': epoch
            })
        
        return acc, f1, precision, recall
    
    else:
        # ReID mode evaluation
        evaluator = R1_mAP_eval(len(query_loader.dataset), max_rank=50, feat_norm=True)
        evaluator.reset()

        print(f"\nExtracting query features ({len(query_loader.dataset)})...")
        for batch in query_loader:
            videos = batch['video'].to(device)
            pids = batch['pid'].to(device)
            
            with autocast():
                feat = model(videos)  # 测试模式返回 [B, 512] 特征
            
            evaluator.update((feat, pids))

        print(f"Extracting gallery features ({len(gallery_loader.dataset)})...")
        for batch in gallery_loader:
            videos = batch['video'].to(device)
            pids = batch['pid'].to(device)
            
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
            wandb.log({
                'test/mAP': mAP, 
                'test/rank1': rank1,
                'test/rank5': rank5,
                'test/rank10': rank10,
                'epoch': epoch
            })
        
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
        item = dataset.data[idx]  # 直接访问data列表
        pid = item['pid']
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
    assert len(overlap) == 0, f"Error: {len(overlap)} samples overlap between query and gallery!"
    
    return query_indices, gallery_indices


# ----------------------------
# Train Entry
# ----------------------------
def train(cfg):
    device = torch.device(f"cuda:{cfg.GPU_IDS[0]}" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.SEED)

    # 创建输出目录
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(cfg.OUTPUT_DIR, 'config.yaml'), 'w') as f:
        f.write(cfg.dump())

    print("=" * 80)
    print(f"Training Configuration:")
    print(f"  Output Dir: {cfg.OUTPUT_DIR}")
    print(f"  Video Type: {cfg.DATA.VIDEO_TYPE}")
    print(f"  Shot Type: {cfg.DATA.SHOT_TYPE}")
    print(f"  Num Frames: {cfg.DATA.NUM_FRAMES}")
    print(f"  Batch Size: {cfg.DATA.BATCH_SIZE}")
    print(f"  Use Sampler: {getattr(cfg.DATA, 'USE_SAMPLER', False)}")
    if hasattr(cfg.DATA, 'USE_SAMPLER') and cfg.DATA.USE_SAMPLER:
        print(f"  Num Instances: {cfg.DATA.NUM_INSTANCES}")
    print("=" * 80)

    # 构建dataloader
    train_loader, num_classes = build_dataloader(cfg, is_train=True)
    test_loader, _ = build_dataloader(cfg, is_train=False)

    # 更新num_classes到config
    cfg.defrost()
    cfg.MODEL.NUM_CLASSES = num_classes
    cfg.freeze()

    print(f"\nDataset Statistics:")
    print(f"  Num Classes: {num_classes}")
    print(f"  Train videos: {len(train_loader.dataset)}")
    print(f"  Test videos: {len(test_loader.dataset)}")

    # Split test set into query and gallery
    dataset = test_loader.dataset
    q_idx, g_idx = split_query_gallery(dataset, 0.5, cfg.SEED)
    
    # 导入collate_fn
    from data.dataloader import collate_fn
    
    q_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, q_idx),
        batch_size=cfg.TEST.BATCH_SIZE, 
        shuffle=False,
        num_workers=cfg.DATA.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    g_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, g_idx),
        batch_size=cfg.TEST.BATCH_SIZE, 
        shuffle=False,
        num_workers=cfg.DATA.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # 初始化wandb
    run_name = f"{cfg.OUTPUT_DIR.split('/')[-1]}"
    wandb.init(
        project="nba-reid", 
        name=run_name,
        config={
            "num_classes": num_classes, 
            "num_frames": cfg.DATA.NUM_FRAMES,
            "batch_size": cfg.DATA.BATCH_SIZE,
            "lr": cfg.SOLVER.BASE_LR,
            "video_type": cfg.DATA.VIDEO_TYPE,
            "shot_type": cfg.DATA.SHOT_TYPE
        }, 
        dir=cfg.OUTPUT_DIR
    )

    # 构建模型和loss
    model = build_model(cfg).to(device)
    wandb.watch(model, log='all', log_freq=100)
    loss_fn = make_loss(cfg, num_classes)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # 优化器和调度器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.SOLVER.BASE_LR,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.SOLVER.MAX_EPOCHS,
        eta_min=1e-6
    )
    
    scaler = GradScaler()
    
    # Initialize best metrics based on task mode
    is_classification = getattr(cfg.DATA, 'SHOT_CLASSIFICATION', False)
    if is_classification:
        best_acc, best_f1, best_precision, best_recall = 0.0, 0.0, 0.0, 0.0
    else:
        best_rank1, best_mAP, best_rank5, best_rank10 = 0.0, 0.0, 0.0, 0.0

    print("\n" + "=" * 80)
    print("Start training...")
    print("=" * 80)

    # 训练循环
    for epoch in range(1, cfg.SOLVER.MAX_EPOCHS + 1):
        print(f"\n{'='*80}")
        print(f"Epoch [{epoch}/{cfg.SOLVER.MAX_EPOCHS}] - LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*80}")
        
        # 训练一个epoch
        train_metrics = train_one_epoch(
            cfg, model, train_loader, optimizer, loss_fn, scaler, epoch, device
        )
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  ID Loss: {train_metrics['id_loss']:.4f}")
        print(f"  Triplet Loss: {train_metrics['triplet_loss']:.4f}")
        print(f"  Batch Time: {train_metrics['batch_time']:.3f}s")
        
        # Log epoch metrics
        wandb.log({
            'train/epoch_loss': train_metrics['loss'],
            'train/epoch_id_loss': train_metrics['id_loss'],
            'train/epoch_triplet_loss': train_metrics['triplet_loss'],
            'epoch': epoch
        })

        scheduler.step()
        
        # 评估
        if epoch % cfg.SOLVER.EVAL_PERIOD == 0:
            if is_classification:
                # Classification mode
                acc, f1, precision, recall = test(cfg, model, q_loader, g_loader, device, epoch)
                
                # 保存最佳模型
                is_best = acc > best_acc
                if is_best:
                    best_acc, best_f1, best_precision, best_recall = acc, f1, precision, recall
                    
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'accuracy': acc,
                        'f1': f1,
                        'precision': precision,
                        'recall': recall,
                        'config': cfg
                    }
                    
                    torch.save(checkpoint, os.path.join(cfg.OUTPUT_DIR, 'best_model.pth'))
                    
                    print(f"\n✅ Saved best model at epoch {epoch}")
                    print(f"  Accuracy  : {acc:.2%}")
                    print(f"  F1-score  : {f1:.2%}")
                    print(f"  Precision : {precision:.2%}")
                    print(f"  Recall    : {recall:.2%}")
            else:
                # ReID mode
                rank1, mAP, rank5, rank10 = test(cfg, model, q_loader, g_loader, device, epoch)
                
                # 保存最佳模型
                is_best = rank1 > best_rank1
                if is_best:
                    best_rank1, best_mAP, best_rank5, best_rank10 = rank1, mAP, rank5, rank10
                    
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'rank1': rank1,
                        'mAP': mAP,
                        'rank5': rank5,
                        'rank10': rank10,
                        'config': cfg
                    }
                    
                    torch.save(checkpoint, os.path.join(cfg.OUTPUT_DIR, 'best_model.pth'))
                    
                    print(f"\n✅ Saved best model at epoch {epoch}")
                    print(f"  Rank-1  : {rank1:.2%}")
                    print(f"  Rank-5  : {rank5:.2%}")
                    print(f"  Rank-10 : {rank10:.2%}")
                    print(f"  mAP     : {mAP:.2%}")
            
            # # 每隔一定epoch也保存checkpoint
            # if epoch % (cfg.SOLVER.EVAL_PERIOD * 5) == 0:
            #     checkpoint = {
            #         'epoch': epoch,
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'scheduler_state_dict': scheduler.state_dict(),
            #         'rank1': rank1,
            #         'mAP': mAP,
            #     }
            #     torch.save(checkpoint, os.path.join(cfg.OUTPUT_DIR, f'checkpoint_epoch_{epoch}.pth'))

    print("\n" + "=" * 80)
    print("Training Completed! Best Results:")
    if is_classification:
        print(f"  Accuracy  : {best_acc:.2%}")
        print(f"  F1-score  : {best_f1:.2%}")
        print(f"  Precision : {best_precision:.2%}")
        print(f"  Recall    : {best_recall:.2%}")
    else:
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
    parser = argparse.ArgumentParser(description='Basketball Video ReID Training')
    parser.add_argument('--config-file', type=str, default='', help='path to config file')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--wandb-offline', action='store_true', help='run wandb in offline mode')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER, 
                       help='modify config options using the command-line')
    args = parser.parse_args()

    if args.wandb_offline:
        os.environ['WANDB_MODE'] = 'offline'

    cfg = get_cfg_defaults()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    if args.mode == 'train':
        train(cfg)
    else:
        print("Test mode not implemented in this version.")


if __name__ == '__main__':
    main()