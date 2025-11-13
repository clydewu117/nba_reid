#!/usr/bin/env python
"""
Basketball Video ReID - Classification and ReID Testing Script
"""

import os
import random
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist

from config.defaults import get_cfg_defaults
from data.dataloader import build_dataloader, collate_fn
from models.build import build_model


def set_seed(seed):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_rank(distmat, q_pids, g_pids, max_rank=50):
    """计算 Rank-k 和 mAP"""
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
    
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis])

    all_cmc, all_AP = [], []
    
    for q_idx in range(num_q):
        orig_cmc = matches[q_idx]
        if not np.any(orig_cmc):
            continue
        
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        precision = tmp_cmc / (np.arange(len(tmp_cmc)) + 1)
        AP = (precision * orig_cmc).sum() / num_rel
        all_AP.append(AP)
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / len(all_cmc)
    mAP = np.mean(all_AP)
    
    return all_cmc, mAP


def split_query_gallery(dataset, query_ratio=0.5, seed=42, min_samples=2):
    """
    Split dataset into query and gallery sets for ReID evaluation.
    """
    np.random.seed(seed)
    pid_to_indices = {}
    
    # Group indices by person ID
    for idx in range(len(dataset)):
        item = dataset.data[idx]
        pid = item['pid']
        pid_to_indices.setdefault(pid, []).append(idx)

    query_indices, gallery_indices = [], []
    excluded_pids = []
    
    for pid, indices in pid_to_indices.items():
        if len(indices) < min_samples:
            excluded_pids.append(pid)
            continue
        
        np.random.shuffle(indices)
        split_point = max(1, int(len(indices) * query_ratio))
        split_point = min(split_point, len(indices) - 1)
        
        query_indices.extend(indices[:split_point])
        gallery_indices.extend(indices[split_point:])
    
    print(f"\n{'='*60}")
    print(f"Query/Gallery Split Summary:")
    print(f"  Query samples    : {len(query_indices)}")
    print(f"  Gallery samples  : {len(gallery_indices)}")
    print(f"  Valid PIDs       : {len(pid_to_indices) - len(excluded_pids)}")
    print(f"  Excluded PIDs    : {len(excluded_pids)} (< {min_samples} samples)")
    print(f"{'='*60}\n")
    
    overlap = set(query_indices) & set(gallery_indices)
    assert len(overlap) == 0, f"Error: {len(overlap)} samples overlap!"
    
    return query_indices, gallery_indices


@torch.no_grad()
def test_classification(cfg, model, test_loader, device):
    """测试分类性能（使用全部测试集）"""
    model.eval()
    
    all_pids = []
    all_probs = []
    all_paths = []
    all_identities = []
    
    print(f"\n{'='*60}")
    print(f"Classification Testing...")
    print(f"{'='*60}\n")
    
    # Unified feature extraction across different backbones (e.g., Uniformer, MViT)
    def _extract_features(videos_tensor):
        # Prefer model-specific eval feature path if available (e.g., MViTReID)
        if hasattr(model, "_forward_features_eval") and callable(getattr(model, "_forward_features_eval")):
            return model._forward_features_eval(videos_tensor)
        # Fallback to calling backbone directly (e.g., Uniformerv2ReID)
        return model.backbone(videos_tensor)

    for batch in tqdm(test_loader, desc="Extracting"):
        videos = batch['video'].to(device)
        pids = batch['pid'].to(device)
        
        features = _extract_features(videos)
        bn_feat = model.reid_head.bottleneck(features)
        cls_score = model.reid_head.classifier(bn_feat)
        probs = F.softmax(cls_score, dim=1)
        
        all_pids.append(pids.cpu())
        all_probs.append(probs.cpu())
        all_paths.extend(batch['video_paths'])
        all_identities.extend(batch['identities'])
    
    all_pids = torch.cat(all_pids, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    
    detailed_results = []
    topk_correct = {1: 0, 3: 0, 5: 0}
    total = len(all_pids)
    
    top5_probs, top5_preds = torch.topk(all_probs, 5, dim=1)
    
    for i in range(total):
        true_pid = all_pids[i].item()
        true_identity = all_identities[i]
        video_path = all_paths[i]
        video_name = os.path.basename(video_path)
        
        # 检查是否在top-k中 (不需要.item())
        is_top1 = (top5_preds[i, 0].item() == true_pid)
        is_top3 = any(top5_preds[i, k].item() == true_pid for k in range(3))
        is_top5 = any(top5_preds[i, k].item() == true_pid for k in range(5))
        
        if is_top1:
            topk_correct[1] += 1
        if is_top3:
            topk_correct[3] += 1
        if is_top5:
            topk_correct[5] += 1
        
        result_row = {
            'video_path': video_path,
            'video_name': video_name,
            'true_identity': true_identity,
            'true_pid': true_pid,
            'true_confidence': all_probs[i, true_pid].item(),
            'top1_correct': is_top1,
            'top3_correct': is_top3,
            'top5_correct': is_top5,
        }
        
        for k in range(5):
            pred_pid = top5_preds[i, k].item()
            pred_prob = top5_probs[i, k].item()
            pred_identity = test_loader.dataset.pid_list[pred_pid]
            
            result_row[f'top{k+1}_pid'] = pred_pid
            result_row[f'top{k+1}_identity'] = pred_identity
            result_row[f'top{k+1}_confidence'] = pred_prob
        
        detailed_results.append(result_row)
    
    print(f"\n{'='*60}")
    print("Classification Results (Top-k Accuracy):")
    print(f"{'='*60}")
    print(f"Overall:")
    for k in [1, 3, 5]:
        acc = topk_correct[k] / total * 100
        print(f"  Top-{k}: {acc:.2f}%")
    
    print(f"\nPer-Identity Top-k Accuracy:")
    identity_topk = {}
    
    for identity in sorted(set(all_identities)):
        identity_indices = [i for i, x in enumerate(all_identities) if x == identity]
        identity_total = len(identity_indices)
        identity_correct = {1: 0, 3: 0, 5: 0}
        
        for idx in identity_indices:
            if detailed_results[idx]['top1_correct']:
                identity_correct[1] += 1
            if detailed_results[idx]['top3_correct']:
                identity_correct[3] += 1
            if detailed_results[idx]['top5_correct']:
                identity_correct[5] += 1
        
        identity_topk[identity] = {
            'total': identity_total,
            'top1_correct': identity_correct[1],
            'top3_correct': identity_correct[3],
            'top5_correct': identity_correct[5],
            'top1_acc': identity_correct[1] / identity_total * 100,
            'top3_acc': identity_correct[3] / identity_total * 100,
            'top5_acc': identity_correct[5] / identity_total * 100,
        }
        
        print(f"  {identity} ({identity_total} videos):")
        print(f"    Top-1: {identity_topk[identity]['top1_acc']:.2f}%")
        print(f"    Top-3: {identity_topk[identity]['top3_acc']:.2f}%")
        print(f"    Top-5: {identity_topk[identity]['top5_acc']:.2f}%")
    
    return {
        'topk_overall': topk_correct,
        'topk_per_identity': identity_topk,
        'total': total,
        'detailed_results': detailed_results
    }


@torch.no_grad()
def test_reid(cfg, model, query_loader, gallery_loader, device):
    """测试ReID性能（使用query/gallery划分）"""
    model.eval()
    
    # Extract query features
    print(f"\n{'='*60}")
    print(f"ReID Testing (Query/Gallery)...")
    print(f"{'='*60}\n")
    
    q_feats, q_pids, q_paths, q_identities = [], [], [], []
    
    print(f"Extracting query features ({len(query_loader.dataset)})...")

    # Reuse the same unified extractor here
    def _extract_features(videos_tensor):
        if hasattr(model, "_forward_features_eval") and callable(getattr(model, "_forward_features_eval")):
            return model._forward_features_eval(videos_tensor)
        return model.backbone(videos_tensor)
    for batch in tqdm(query_loader, desc="Query"):
        videos = batch['video'].to(device)
        pids = batch['pid'].to(device)
        
        features = _extract_features(videos)
        bn_feat = model.reid_head.bottleneck(features)
        feat_norm = F.normalize(bn_feat, p=2, dim=1)
        
        q_feats.append(feat_norm.cpu())
        q_pids.append(pids.cpu())
        q_paths.extend(batch['video_paths'])
        q_identities.extend(batch['identities'])
    
    q_feats = torch.cat(q_feats, dim=0)
    q_pids = torch.cat(q_pids, dim=0)
    
    # Extract gallery features
    g_feats, g_pids, g_paths, g_identities = [], [], [], []
    
    print(f"Extracting gallery features ({len(gallery_loader.dataset)})...")
    for batch in tqdm(gallery_loader, desc="Gallery"):
        videos = batch['video'].to(device)
        pids = batch['pid'].to(device)
        
        features = _extract_features(videos)
        bn_feat = model.reid_head.bottleneck(features)
        feat_norm = F.normalize(bn_feat, p=2, dim=1)
        
        g_feats.append(feat_norm.cpu())
        g_pids.append(pids.cpu())
        g_paths.extend(batch['video_paths'])
        g_identities.extend(batch['identities'])
    
    g_feats = torch.cat(g_feats, dim=0)
    g_pids = torch.cat(g_pids, dim=0)
    
    # Compute distance matrix
    q_feats_np = q_feats.numpy()
    g_feats_np = g_feats.numpy()
    q_pids_np = q_pids.numpy()
    g_pids_np = g_pids.numpy()
    
    distmat = cdist(q_feats_np, g_feats_np, metric='euclidean')
    
    # Evaluate
    cmc, mAP = evaluate_rank(distmat, q_pids_np, g_pids_np, max_rank=50)
    
    print(f"\n{'='*60}")
    print("ReID Results (Feature Matching):")
    print(f"{'='*60}")
    print(f"Overall:")
    print(f"  mAP: {mAP:.2%}")
    print(f"  Rank-1: {cmc[0]:.2%}")
    print(f"  Rank-5: {cmc[4]:.2%}")
    print(f"  Rank-10: {cmc[9]:.2%}")
    
    # Per-identity ReID metrics
    print(f"\nPer-Identity ReID Metrics:")
    identity_reid = {}
    
    for identity in sorted(set(q_identities)):
        # Get query indices for this identity
        q_mask = np.array([i for i, x in enumerate(q_identities) if x == identity])
        # Get gallery indices for this identity
        g_mask = np.array([i for i, x in enumerate(g_identities) if x == identity])
        
        if len(q_mask) == 0 or len(g_mask) == 0:
            print(f"  {identity}: Skipped (no query or gallery samples)")
            continue
        
        # Extract submatrix
        identity_distmat = distmat[np.ix_(q_mask, g_mask)]
        identity_q_pids = q_pids_np[q_mask]
        identity_g_pids = g_pids_np[g_mask]
        
        identity_cmc, identity_mAP = evaluate_rank(
            identity_distmat, identity_q_pids, identity_g_pids,
            max_rank=min(50, len(g_mask))
        )
        
        identity_reid[identity] = {
            'query_samples': len(q_mask),
            'gallery_samples': len(g_mask),
            'mAP': identity_mAP,
            'rank1': identity_cmc[0],
            'rank5': identity_cmc[4] if len(identity_cmc) > 4 else identity_cmc[-1],
            'rank10': identity_cmc[9] if len(identity_cmc) > 9 else identity_cmc[-1]
        }
        
        print(f"  {identity} (Q:{len(q_mask)}, G:{len(g_mask)}):")
        print(f"    mAP: {identity_mAP:.2%}")
        print(f"    Rank-1: {identity_cmc[0]:.2%}")
        print(f"    Rank-5: {identity_reid[identity]['rank5']:.2%}")
    
    return {
        'reid_overall': {
            'mAP': mAP, 
            'rank1': cmc[0], 
            'rank5': cmc[4],
            'rank10': cmc[9]
        },
        'reid_per_identity': identity_reid,
        'num_query': len(q_pids),
        'num_gallery': len(g_pids)
    }


def main():
    parser = argparse.ArgumentParser(description='Basketball Video ReID Testing')
    parser.add_argument('--config-file', type=str, required=True,
                       help='path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='path to checkpoint')
    parser.add_argument('--output-dir', type=str, default='test_results',
                       help='output directory for results')
    parser.add_argument('--query-ratio', type=float, default=0.5, 
                       help='ratio of test samples to use as query (default: 0.5)')
    parser.add_argument('--seed', type=int, default=None, 
                       help='random seed (default: use cfg.SEED)')
    args = parser.parse_args()

    # Load config
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    # Set random seed
    seed = args.seed if args.seed is not None else cfg.SEED
    set_seed(seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device = torch.device(f"cuda:{cfg.GPU_IDS[0]}" if torch.cuda.is_available() else "cpu")
    
    # Extract prefix from checkpoint path
    checkpoint_dir = os.path.dirname(args.checkpoint)
    prefix = os.path.basename(checkpoint_dir)

    # Print configuration
    print("=" * 80)
    print("Testing Configuration:")
    print(f"  Config File  : {args.config_file}")
    print(f"  Checkpoint   : {args.checkpoint}")
    print(f"  Prefix       : {prefix}")
    print(f"  Video Type   : {cfg.DATA.VIDEO_TYPE}")
    print(f"  Shot Type    : {cfg.DATA.SHOT_TYPE}")
    print(f"  Output Dir   : {args.output_dir}")
    print(f"  Query Ratio  : {args.query_ratio}")
    print(f"  Random Seed  : {seed}")
    print("=" * 80)

    # Build full test dataloader
    test_loader, num_classes = build_dataloader(cfg, is_train=False)
    cfg.defrost()
    cfg.MODEL.NUM_CLASSES = num_classes
    cfg.freeze()

    print(f"\nDataset Statistics:")
    print(f"  Num Classes: {num_classes}")
    print(f"  Test videos: {len(test_loader.dataset)}")

    # Build model
    model = build_model(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\n{'='*60}")
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    if 'rank1' in checkpoint:
        print(f"  Training Best Rank-1: {checkpoint['rank1']:.2%}")
        print(f"  Training Best mAP: {checkpoint['mAP']:.2%}")
    print(f"{'='*60}")

    # 1. Test Classification (on full test set)
    cls_results = test_classification(cfg, model, test_loader, device)
    
    # 2. Split query/gallery and test ReID
    dataset = test_loader.dataset
    q_idx, g_idx = split_query_gallery(dataset, args.query_ratio, seed)
    
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
    
    reid_results = test_reid(cfg, model, q_loader, g_loader, device)
    
    # Save results
    print(f"\n{'='*60}")
    print("Saving results...")
    print(f"{'='*60}\n")
    
    # 1. Detailed per-video classification results
    detailed_df = pd.DataFrame(cls_results['detailed_results'])
    detailed_csv = os.path.join(args.output_dir, f'{prefix}_detailed_results.csv')
    detailed_df.to_csv(detailed_csv, index=False)
    print(f"✓ Detailed per-video results saved to: {detailed_csv}")
    
    # 2. Per-identity top-k accuracy
    identity_topk_df = pd.DataFrame.from_dict(cls_results['topk_per_identity'], orient='index')
    identity_topk_df.index.name = 'identity'
    topk_csv = os.path.join(args.output_dir, f'{prefix}_per_identity_topk.csv')
    identity_topk_df.to_csv(topk_csv)
    print(f"✓ Per-identity top-k accuracy saved to: {topk_csv}")
    
    # 3. Per-identity ReID metrics
    if reid_results['reid_per_identity']:
        identity_reid_df = pd.DataFrame.from_dict(reid_results['reid_per_identity'], orient='index')
        identity_reid_df.index.name = 'identity'
        reid_csv = os.path.join(args.output_dir, f'{prefix}_per_identity_reid.csv')
        identity_reid_df.to_csv(reid_csv)
        print(f"✓ Per-identity ReID metrics saved to: {reid_csv}")
    
    # 4. Overall summary
    summary_file = os.path.join(args.output_dir, f'{prefix}_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("BASKETBALL VIDEO REID - TEST RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Config: {args.config_file}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Checkpoint Epoch: {checkpoint['epoch']}\n")
        f.write(f"Random Seed: {seed}\n")
        f.write(f"Query Ratio: {args.query_ratio}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("CLASSIFICATION METRICS (Full Test Set)\n")
        f.write("="*60 + "\n")
        f.write(f"Total test videos: {cls_results['total']}\n\n")
        f.write("Overall Top-k Accuracy:\n")
        for k in [1, 3, 5]:
            acc = cls_results['topk_overall'][k] / cls_results['total'] * 100
            f.write(f"  Top-{k}: {acc:.2f}% ({cls_results['topk_overall'][k]}/{cls_results['total']})\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("REID METRICS (Query/Gallery Split)\n")
        f.write("="*60 + "\n")
        f.write(f"Query samples: {reid_results['num_query']}\n")
        f.write(f"Gallery samples: {reid_results['num_gallery']}\n")
        f.write("\nOverall ReID Metrics:\n")
        f.write(f"  mAP: {reid_results['reid_overall']['mAP']:.2%}\n")
        f.write(f"  Rank-1: {reid_results['reid_overall']['rank1']:.2%}\n")
        f.write(f"  Rank-5: {reid_results['reid_overall']['rank5']:.2%}\n")
        f.write(f"  Rank-10: {reid_results['reid_overall']['rank10']:.2%}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("DATASET INFO\n")
        f.write("="*60 + "\n")
        f.write(f"Number of identities: {len(cls_results['topk_per_identity'])}\n")
        f.write(f"Video type: {cfg.DATA.VIDEO_TYPE}\n")
        f.write(f"Shot type: {cfg.DATA.SHOT_TYPE}\n")
        f.write(f"Num frames: {cfg.DATA.NUM_FRAMES}\n")
    
    print(f"✓ Summary saved to: {summary_file}")
    
    print(f"\n{'='*60}")
    print("All results saved successfully!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()