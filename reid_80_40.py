#!/usr/bin/env python
"""
Basketball Video ReID - Cross-Dataset Testing Script
Specialized for testing models trained on N identities on a different M identities.
Only performs ReID (feature matching) evaluation, skips classification.

Fix: Per-identity ReID now evaluates each identity's query clips against the
     FULL gallery (not just that identity's own gallery clips), which is the
     correct closed-set ReID protocol.
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
from metrics import evaluate_rank  # Reuse from metrics.py — no duplicate definition


def set_seed(seed):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
def test_reid(cfg, model, query_loader, gallery_loader, device):
    """测试ReID性能（使用query/gallery划分）"""
    model.eval()

    print(f"\n{'='*60}")
    print(f"ReID Testing (Query/Gallery)...")
    print(f"{'='*60}\n")

    # Unified feature extraction across different backbones
    def _extract_features(videos_tensor):
        if hasattr(model, "_forward_features_eval") and callable(getattr(model, "_forward_features_eval")):
            return model._forward_features_eval(videos_tensor)
        return model.backbone(videos_tensor)

    # ── Extract query features ──────────────────────────────────────────────
    q_feats, q_pids, q_paths, q_identities = [], [], [], []

    print(f"Extracting query features ({len(query_loader.dataset)} samples)...")
    for batch in tqdm(query_loader, desc="Query"):
        videos = batch['video'].to(device)
        pids = batch['pid'].to(device)

        outputs = model(videos)
        if isinstance(outputs, dict):
            if "feat" in outputs:
                feat_norm = outputs["feat"]
            elif "bn_feat" in outputs:
                feat_norm = outputs["bn_feat"]
                feat_norm = F.normalize(feat_norm, p=2, dim=1)
            else:
                raise RuntimeError("Model evaluation output dict missing 'feat'/'bn_feat' keys.")
        elif isinstance(outputs, (list, tuple)):
            feat_norm = outputs[0]
        else:
            feat_norm = outputs
        feat_norm = F.normalize(feat_norm, p=2, dim=1)

        q_feats.append(feat_norm.cpu())
        q_pids.append(pids.cpu())
        q_paths.extend(batch['video_paths'])
        q_identities.extend(batch['identities'])

    q_feats = torch.cat(q_feats, dim=0)
    q_pids = torch.cat(q_pids, dim=0)

    # ── Extract gallery features ────────────────────────────────────────────
    g_feats, g_pids, g_paths, g_identities = [], [], [], []

    print(f"Extracting gallery features ({len(gallery_loader.dataset)} samples)...")
    for batch in tqdm(gallery_loader, desc="Gallery"):
        videos = batch['video'].to(device)
        pids = batch['pid'].to(device)

        outputs = model(videos)
        if isinstance(outputs, dict):
            if "feat" in outputs:
                feat_norm = outputs["feat"]
            elif "bn_feat" in outputs:
                feat_norm = outputs["bn_feat"]
                feat_norm = F.normalize(feat_norm, p=2, dim=1)
            else:
                raise RuntimeError("Model evaluation output dict missing 'feat'/'bn_feat' keys.")
        elif isinstance(outputs, (list, tuple)):
            feat_norm = outputs[0]
        else:
            feat_norm = outputs
        feat_norm = F.normalize(feat_norm, p=2, dim=1)

        g_feats.append(feat_norm.cpu())
        g_pids.append(pids.cpu())
        g_paths.extend(batch['video_paths'])
        g_identities.extend(batch['identities'])

    g_feats = torch.cat(g_feats, dim=0)
    g_pids = torch.cat(g_pids, dim=0)

    # ── Compute full distance matrix ────────────────────────────────────────
    q_feats_np = q_feats.numpy()
    g_feats_np = g_feats.numpy()
    q_pids_np = q_pids.numpy()
    g_pids_np = g_pids.numpy()

    print("\nComputing distance matrix...")
    distmat = cdist(q_feats_np, g_feats_np, metric='euclidean')

    # ── Overall ReID metrics ────────────────────────────────────────────────
    cmc, mAP = evaluate_rank(distmat, q_pids_np, g_pids_np, max_rank=50)

    print(f"\n{'='*60}")
    print("ReID Results (Feature Matching):")
    print(f"{'='*60}")
    print(f"Overall:")
    print(f"  mAP     : {mAP:.2%}")
    print(f"  Rank-1  : {cmc[0]:.2%}")
    print(f"  Rank-5  : {cmc[4]:.2%}")
    print(f"  Rank-10 : {cmc[9]:.2%}")

    # ── Per-identity ReID metrics ───────────────────────────────────────────
    #
    # CORRECT PROTOCOL:
    #   For each identity I, take only I's query rows from distmat,
    #   but keep ALL gallery columns.
    #   This asks: "given a clip of player I, can the model rank player I's
    #   gallery clips above clips of ALL other players?"
    #
    # PREVIOUS BUG (now fixed):
    #   The old code sliced distmat[np.ix_(q_mask, g_mask)] where both
    #   q_mask and g_mask were restricted to the same identity I.
    #   That turned the problem into "rank I vs I", a trivially easy task
    #   with no distractors, making per-identity Rank-1 approach ~100%
    #   and rendering the metric meaningless.
    #
    print(f"\nPer-Identity ReID Metrics:")
    identity_reid = {}

    for identity in sorted(set(q_identities)):
        # Rows: only this identity's query clips
        q_mask = np.array([i for i, x in enumerate(q_identities) if x == identity])

        if len(q_mask) == 0:
            print(f"  {identity}: Skipped (no query samples)")
            continue

        # Columns: FULL gallery (all identities)  ← KEY FIX
        identity_distmat = distmat[q_mask, :]          # shape: [n_q_this_id, n_g_total]
        identity_q_pids  = q_pids_np[q_mask]
        identity_g_pids  = g_pids_np                   # all gallery pids

        identity_cmc, identity_mAP = evaluate_rank(
            identity_distmat,
            identity_q_pids,
            identity_g_pids,
            max_rank=min(50, len(g_pids_np))
        )

        rank5  = identity_cmc[4] if len(identity_cmc) > 4 else identity_cmc[-1]
        rank10 = identity_cmc[9] if len(identity_cmc) > 9 else identity_cmc[-1]

        identity_reid[identity] = {
            'query_samples':   len(q_mask),
            'gallery_samples': len(g_pids_np),   # full gallery size
            'mAP':    identity_mAP,
            'rank1':  identity_cmc[0],
            'rank5':  rank5,
            'rank10': rank10,
        }

        print(f"  {identity} (Q:{len(q_mask)}, G:{len(g_pids_np)}):")
        print(f"    mAP: {identity_mAP:.2%}  "
              f"Rank-1: {identity_cmc[0]:.2%}  "
              f"Rank-5: {rank5:.2%}  "
              f"Rank-10: {rank10:.2%}")

    return {
        'reid_overall': {
            'mAP':    mAP,
            'rank1':  cmc[0],
            'rank5':  cmc[4],
            'rank10': cmc[9],
        },
        'reid_per_identity': identity_reid,
        'num_query':            len(q_pids),
        'num_gallery':          len(g_pids),
        'query_identities':     sorted(set(q_identities)),
        'gallery_identities':   sorted(set(g_identities)),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Basketball Video ReID Cross-Dataset Testing (e.g., train on 80, test on 40)')
    parser.add_argument('--config-file', type=str, required=True,
                        help='path to config file used during training')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='path to trained checkpoint')
    parser.add_argument('--output-dir', type=str, default='reid_80_40_results',
                        help='output directory for results')
    parser.add_argument('--query-ratio', type=float, default=0.5,
                        help='ratio of test samples to use as query (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    args = parser.parse_args()

    # ── Config ──────────────────────────────────────────────────────────────
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(f"cuda:{cfg.GPU_IDS[0]}" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = os.path.dirname(args.checkpoint)
    prefix = os.path.basename(checkpoint_dir)

    print("=" * 80)
    print("Cross-Dataset ReID Testing Configuration:")
    print(f"  Config File  : {args.config_file}")
    print(f"  Checkpoint   : {args.checkpoint}")
    print(f"  Prefix       : {prefix}")
    print(f"  Video Type   : {cfg.DATA.VIDEO_TYPE}")
    print(f"  Shot Type    : {cfg.DATA.SHOT_TYPE}")
    print(f"  Output Dir   : {args.output_dir}")
    print(f"  Query Ratio  : {args.query_ratio}")
    print(f"  Random Seed  : {args.seed}")
    print("=" * 80)

    # ── Dataset ─────────────────────────────────────────────────────────────
    test_loader, test_num_classes = build_dataloader(cfg, is_train=False)

    print(f"\nDataset Statistics:")
    print(f"  Test set identities : {test_num_classes}")
    print(f"  Test videos         : {len(test_loader.dataset)}")

    # ── Model ────────────────────────────────────────────────────────────────
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    trained_num_classes = cfg.DATA.TRAIN_IDENTITIES

    print(f"  Trained identities  : {trained_num_classes}")

    if test_num_classes != trained_num_classes:
        print(f"\n{'='*80}")
        print(f"⚠️  CROSS-DATASET TESTING MODE")
        print(f"   Model trained on {trained_num_classes} identities")
        print(f"   Testing on {test_num_classes} different identities")
        print(f"   Only ReID (feature matching) evaluation will be performed")
        print(f"   Classification evaluation is not applicable")
        print(f"{'='*80}\n")

    # Build model with trained number of classes (to match checkpoint architecture)
    cfg.defrost()
    cfg.MODEL.NUM_CLASSES = trained_num_classes
    cfg.freeze()

    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"\n{'='*60}")
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    if 'rank1' in checkpoint:
        print(f"  Training Best Rank-1: {checkpoint['rank1']:.2%}")
        print(f"  Training Best mAP   : {checkpoint['mAP']:.2%}")
    print(f"{'='*60}")

    # ── Query / Gallery split ────────────────────────────────────────────────
    dataset = test_loader.dataset
    q_idx, g_idx = split_query_gallery(dataset, args.query_ratio, args.seed)

    q_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, q_idx),
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATA.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    g_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, g_idx),
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATA.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    reid_results = test_reid(cfg, model, q_loader, g_loader, device)

    # ── Save results ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Saving results...")
    print(f"{'='*60}\n")

    # 1. Per-identity ReID metrics
    if reid_results['reid_per_identity']:
        identity_reid_df = pd.DataFrame.from_dict(
            reid_results['reid_per_identity'], orient='index'
        )
        identity_reid_df.index.name = 'identity'
        reid_csv = os.path.join(args.output_dir, f'{prefix}_per_identity_reid.csv')
        identity_reid_df.to_csv(reid_csv)
        print(f"✓ Per-identity ReID metrics → {reid_csv}")

    # 2. Overall summary
    summary_file = os.path.join(args.output_dir, f'{prefix}_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BASKETBALL VIDEO REID - CROSS-DATASET TEST RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Config     : {args.config_file}\n")
        f.write(f"Checkpoint : {args.checkpoint}\n")
        f.write(f"Epoch      : {checkpoint['epoch']}\n")
        f.write(f"Seed       : {args.seed}\n")
        f.write(f"Query Ratio: {args.query_ratio}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("MODEL INFO\n")
        f.write("="*80 + "\n")
        f.write(f"Trained on {trained_num_classes} identities\n")
        f.write(f"Tested on  {test_num_classes} identities\n")
        if 'rank1' in checkpoint:
            f.write(f"Training Best Rank-1: {checkpoint['rank1']:.2%}\n")
            f.write(f"Training Best mAP   : {checkpoint['mAP']:.2%}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("REID METRICS (Query/Gallery Split on Test Set)\n")
        f.write("="*80 + "\n")
        f.write(f"Query samples  : {reid_results['num_query']}\n")
        f.write(f"Gallery samples: {reid_results['num_gallery']}\n")
        f.write(f"Test identities: {len(reid_results['query_identities'])}\n")
        f.write("\nOverall ReID Metrics:\n")
        f.write(f"  mAP    : {reid_results['reid_overall']['mAP']:.2%}\n")
        f.write(f"  Rank-1 : {reid_results['reid_overall']['rank1']:.2%}\n")
        f.write(f"  Rank-5 : {reid_results['reid_overall']['rank5']:.2%}\n")
        f.write(f"  Rank-10: {reid_results['reid_overall']['rank10']:.2%}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("PER-IDENTITY REID SUMMARY\n")
        f.write("="*80 + "\n")
        for identity in sorted(reid_results['reid_per_identity'].keys()):
            metrics = reid_results['reid_per_identity'][identity]
            f.write(f"\n{identity}:\n")
            f.write(f"  Query: {metrics['query_samples']}, Gallery: {metrics['gallery_samples']}\n")
            f.write(f"  mAP   : {metrics['mAP']:.2%}\n")
            f.write(f"  Rank-1: {metrics['rank1']:.2%}\n")
            f.write(f"  Rank-5: {metrics['rank5']:.2%}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("DATASET INFO\n")
        f.write("="*80 + "\n")
        f.write(f"Video type  : {cfg.DATA.VIDEO_TYPE}\n")
        f.write(f"Shot type   : {cfg.DATA.SHOT_TYPE}\n")
        f.write(f"Num frames  : {cfg.DATA.NUM_FRAMES}\n")
        f.write(f"Frame stride: {cfg.DATA.FRAME_STRIDE}\n")
        f.write(f"Sample start: {cfg.DATA.SAMPLE_START}\n")

    print(f"✓ Summary → {summary_file}")

    print(f"\n{'='*60}")
    print("Cross-dataset testing completed successfully!")
    print(f"{'='*60}\n")

    print("Results Summary:")
    print(f"  Trained identities: {trained_num_classes}")
    print(f"  Test identities   : {test_num_classes}")
    print(f"  Overall mAP       : {reid_results['reid_overall']['mAP']:.2%}")
    print(f"  Overall Rank-1    : {reid_results['reid_overall']['rank1']:.2%}")
    print(f"  Overall Rank-5    : {reid_results['reid_overall']['rank5']:.2%}")
    print(f"  Overall Rank-10   : {reid_results['reid_overall']['rank10']:.2%}")


if __name__ == '__main__':
    main()
