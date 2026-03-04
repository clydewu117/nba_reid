#!/usr/bin/env python
"""
Basketball Video ReID - Classification and ReID Testing Script

Multi-run logic:
  - If temporal_shuffle_ratio == 0.0  (no shuffle):
      Test-time frame sampling is deterministic → run ONCE.

  - If temporal_shuffle_ratio > 0.0  (shuffle enabled):
      _apply_temporal_shuffle uses np.random, so results vary by seed.
      Run `--num-runs` times with seeds  base_seed, base_seed+1, ...
      Q/G split is FIXED across all runs (always uses base_seed),
      only the temporal shuffle randomness changes.
      Final metrics are mean ± std across runs.

Fix: Per-identity ReID evaluates each identity's query clips against the
     FULL gallery (not just that identity's own gallery clips).
"""

import os
import random
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial.distance import cdist

from config.defaults import get_cfg_defaults
from data.dataloader_vanilla import build_dataloader, collate_fn
from models.build import build_model
from metrics import evaluate_rank


# ─────────────────────────────────────────────
# Seed
# ─────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────
# Query / Gallery split  (fixed, uses base_seed only)
# ─────────────────────────────────────────────

def split_query_gallery(dataset, query_ratio=0.5, seed=42, min_samples=2):
    np.random.seed(seed)
    pid_to_indices = {}

    for idx in range(len(dataset)):
        pid = dataset.data[idx]['pid']
        pid_to_indices.setdefault(pid, []).append(idx)

    query_indices, gallery_indices = [], []
    excluded_pids = []

    for pid, indices in pid_to_indices.items():
        if len(indices) < min_samples:
            excluded_pids.append(pid)
            continue
        np.random.shuffle(indices)
        sp = max(1, int(len(indices) * query_ratio))
        sp = min(sp, len(indices) - 1)
        query_indices.extend(indices[:sp])
        gallery_indices.extend(indices[sp:])

    print(f"\n{'='*60}")
    print(f"Query/Gallery Split  (seed={seed}, fixed across all runs)")
    print(f"  Query samples    : {len(query_indices)}")
    print(f"  Gallery samples  : {len(gallery_indices)}")
    print(f"  Valid PIDs       : {len(pid_to_indices) - len(excluded_pids)}")
    print(f"  Excluded PIDs    : {len(excluded_pids)} (< {min_samples} samples)")
    print(f"{'='*60}\n")

    overlap = set(query_indices) & set(gallery_indices)
    assert len(overlap) == 0, f"{len(overlap)} samples overlap between query and gallery!"

    return query_indices, gallery_indices


# ─────────────────────────────────────────────
# Classification  (run once — deterministic when shuffle_ratio=0,
#                  or run multiple times when shuffle_ratio>0)
# ─────────────────────────────────────────────

def _backbone_features(model, videos):
    """Unified backbone feature extraction (handles forward_features vs __call__)."""
    backbone = model.backbone
    if hasattr(backbone, "forward_features"):
        return backbone.forward_features(videos)
    return backbone(videos)


def _head_logits(model, features):
    """Forward through ReID head to get classification logits.
    Handles heads with optional feat_proj (e.g. VideoMAEv2 768->512)."""
    head = model.reid_head
    if hasattr(head, "feat_proj") and head.feat_proj is not None:
        features = head.feat_proj(features)
    bn_feat = head.bottleneck(features)
    return head.classifier(bn_feat)


def _head_feat(model, features):
    """Forward through ReID head to get normalised ReID feature."""
    head = model.reid_head
    if hasattr(head, "feat_proj") and head.feat_proj is not None:
        features = head.feat_proj(features)
    bn_feat = head.bottleneck(features)
    neck_feat = getattr(head, "neck_feat", "after")
    if neck_feat == "after":
        return F.normalize(bn_feat, p=2, dim=1)
    return F.normalize(features, p=2, dim=1)


@torch.no_grad()
def _extract_cls(model, loader, device):
    """Extract classification probs and pids for one pass over loader."""
    all_pids, all_probs, all_paths, all_identities = [], [], [], []

    for batch in tqdm(loader, desc="  cls", leave=False):
        videos = batch['video'].to(device)
        pids   = batch['pid'].to(device)

        features  = _backbone_features(model, videos)
        cls_score = _head_logits(model, features)
        probs     = F.softmax(cls_score, dim=1)

        all_pids.append(pids.cpu())
        all_probs.append(probs.cpu())
        all_paths.extend(batch['video_paths'])
        all_identities.extend(batch['identities'])

    return (torch.cat(all_pids),
            torch.cat(all_probs),
            all_paths,
            all_identities)


def compute_cls_metrics(all_pids, all_probs, all_paths, all_identities, pid_list):
    """Compute top-k accuracy given extracted probs."""
    total = len(all_pids)
    k     = min(5, all_probs.size(1))
    top5_probs, top5_preds = torch.topk(all_probs, k, dim=1)

    topk_correct  = {1: 0, 3: 0, 5: 0}
    detailed_rows = []

    for i in range(total):
        true_pid      = all_pids[i].item()
        true_identity = all_identities[i]

        is_top1 = (top5_preds[i, 0].item() == true_pid)
        is_top3 = any(top5_preds[i, j].item() == true_pid for j in range(min(3, k)))
        is_top5 = any(top5_preds[i, j].item() == true_pid for j in range(k))

        if is_top1: topk_correct[1] += 1
        if is_top3: topk_correct[3] += 1
        if is_top5: topk_correct[5] += 1

        row = {
            'video_path':      all_paths[i],
            'video_name':      os.path.basename(all_paths[i]),
            'true_identity':   true_identity,
            'true_pid':        true_pid,
            'true_confidence': all_probs[i, true_pid].item(),
            'top1_correct':    is_top1,
            'top3_correct':    is_top3,
            'top5_correct':    is_top5,
        }
        for j in range(k):
            pred_pid = top5_preds[i, j].item()
            row[f'top{j+1}_pid']        = pred_pid
            row[f'top{j+1}_identity']   = pid_list[pred_pid]
            row[f'top{j+1}_confidence'] = top5_probs[i, j].item()
        detailed_rows.append(row)

    identity_topk = {}
    for identity in sorted(set(all_identities)):
        idxs = [i for i, x in enumerate(all_identities) if x == identity]
        n    = len(idxs)
        corr = {1: 0, 3: 0, 5: 0}
        for idx in idxs:
            if detailed_rows[idx]['top1_correct']: corr[1] += 1
            if detailed_rows[idx]['top3_correct']: corr[3] += 1
            if detailed_rows[idx]['top5_correct']: corr[5] += 1
        identity_topk[identity] = {
            'total':        n,
            'top1_correct': corr[1],
            'top3_correct': corr[3],
            'top5_correct': corr[5],
            'top1_acc':     corr[1] / n * 100,
            'top3_acc':     corr[3] / n * 100,
            'top5_acc':     corr[5] / n * 100,
        }

    return {
        'topk_overall':      topk_correct,
        'topk_per_identity': identity_topk,
        'total':             total,
        'detailed_results':  detailed_rows,
    }


def test_classification(cfg, model, test_loader, device,
                        num_runs=1, base_seed=42, shuffle_ratio=0.0):
    """
    Run classification test.
    - shuffle_ratio == 0 → deterministic, run once.
    - shuffle_ratio >  0 → stochastic frames, run num_runs times and average probs.
    """
    model.eval()
    pid_list = test_loader.dataset.pid_list

    print(f"\n{'='*60}")
    if shuffle_ratio == 0.0:
        print("Classification Testing  (deterministic, 1 run)")
        print(f"{'='*60}\n")
        set_seed(base_seed)
        pids, probs, paths, identities = _extract_cls(model, test_loader, device)
        return compute_cls_metrics(pids, probs, paths, identities, pid_list)
    else:
        print(f"Classification Testing  (shuffle={shuffle_ratio}, {num_runs} runs, "
              f"seeds {base_seed}–{base_seed+num_runs-1})")
        print(f"{'='*60}\n")

        sum_probs      = None
        ref_pids       = None
        ref_paths      = None
        ref_identities = None

        for i in range(num_runs):
            run_seed = base_seed + i
            set_seed(run_seed)
            print(f"  Run {i+1}/{num_runs}  (seed={run_seed})")
            pids, probs, paths, identities = _extract_cls(model, test_loader, device)

            if sum_probs is None:
                sum_probs      = probs.clone()
                ref_pids       = pids
                ref_paths      = paths
                ref_identities = identities
            else:
                sum_probs += probs

        avg_probs = sum_probs / num_runs
        return compute_cls_metrics(ref_pids, avg_probs, ref_paths, ref_identities, pid_list)


# ─────────────────────────────────────────────
# Single ReID run
# ─────────────────────────────────────────────

@torch.no_grad()
def run_reid_once(cfg, model, q_loader, g_loader, device):
    """One ReID evaluation pass (Q/G split already fixed externally)."""

    def extract(loader, desc):
        feats, pids, identities = [], [], []
        for batch in tqdm(loader, desc=f"  {desc}", leave=False):
            videos = batch['video'].to(device)
            pid    = batch['pid'].to(device)
            feat   = _head_feat(model, _backbone_features(model, videos))
            feats.append(feat.cpu())
            pids.append(pid.cpu())
            identities.extend(batch['identities'])
        return torch.cat(feats).numpy(), torch.cat(pids).numpy(), identities

    q_feats, q_pids, q_ids = extract(q_loader, "Query")
    g_feats, g_pids, g_ids = extract(g_loader, "Gallery")

    distmat = cdist(q_feats, g_feats, metric='euclidean')

    # Overall
    cmc, mAP = evaluate_rank(distmat, q_pids, g_pids, max_rank=50)
    overall = {
        'mAP':    mAP,
        'rank1':  float(cmc[0]),
        'rank5':  float(cmc[4] if len(cmc) > 4 else cmc[-1]),
        'rank10': float(cmc[9] if len(cmc) > 9 else cmc[-1]),
    }

    # Per-identity  (query rows vs FULL gallery)
    per_identity = {}
    for identity in sorted(set(q_ids)):
        q_mask = np.array([i for i, x in enumerate(q_ids) if x == identity])
        if len(q_mask) == 0:
            continue
        id_cmc, id_mAP = evaluate_rank(
            distmat[q_mask, :], q_pids[q_mask], g_pids,
            max_rank=min(50, len(g_pids))
        )
        per_identity[identity] = {
            'mAP':    id_mAP,
            'rank1':  float(id_cmc[0]),
            'rank5':  float(id_cmc[4] if len(id_cmc) > 4 else id_cmc[-1]),
            'rank10': float(id_cmc[9] if len(id_cmc) > 9 else id_cmc[-1]),
        }

    return {'overall': overall, 'per_identity': per_identity}


# ─────────────────────────────────────────────
# Average across runs
# ─────────────────────────────────────────────

def average_reid_runs(run_results):
    metrics = ['mAP', 'rank1', 'rank5', 'rank10']

    overall_vals = defaultdict(list)
    for r in run_results:
        for m in metrics:
            overall_vals[m].append(r['overall'][m])

    overall_avg = {m: float(np.mean(overall_vals[m])) for m in metrics}
    overall_std = {m: float(np.std(overall_vals[m]))  for m in metrics}

    all_identities = set()
    for r in run_results:
        all_identities.update(r['per_identity'].keys())

    per_identity_avg, per_identity_std = {}, {}
    for identity in sorted(all_identities):
        id_vals = defaultdict(list)
        for r in run_results:
            if identity in r['per_identity']:
                for m in metrics:
                    id_vals[m].append(r['per_identity'][identity][m])
        per_identity_avg[identity] = {m: float(np.mean(id_vals[m])) for m in metrics}
        per_identity_std[identity] = {m: float(np.std(id_vals[m]))  for m in metrics}

    return {
        'overall_avg':      overall_avg,
        'overall_std':      overall_std,
        'per_identity_avg': per_identity_avg,
        'per_identity_std': per_identity_std,
    }


def test_reid(cfg, model, dataset, q_idx, g_idx, device,
              num_runs=1, base_seed=42, shuffle_ratio=0.0):
    """
    Run ReID test.
    - shuffle_ratio == 0 → deterministic frames, run once.
    - shuffle_ratio >  0 → stochastic frames, run num_runs times and average.
    Q/G split (q_idx, g_idx) is FIXED for all runs.
    """
    def make_loaders():
        def _loader(indices):
            return torch.utils.data.DataLoader(
                torch.utils.data.Subset(dataset, indices),
                batch_size=cfg.TEST.BATCH_SIZE,
                shuffle=False,
                num_workers=cfg.DATA.NUM_WORKERS,
                collate_fn=collate_fn,
                pin_memory=True,
            )
        return _loader(q_idx), _loader(g_idx)

    model.eval()
    actual_runs = 1 if shuffle_ratio == 0.0 else num_runs
    run_results = []

    print(f"\n{'='*60}")
    if shuffle_ratio == 0.0:
        print("ReID Testing  (deterministic, 1 run)")
    else:
        print(f"ReID Testing  (shuffle={shuffle_ratio}, {num_runs} runs, "
              f"seeds {base_seed}–{base_seed+num_runs-1})")
    print(f"  Q/G split fixed at seed={base_seed}")
    print(f"{'='*60}")

    for i in range(actual_runs):
        run_seed = base_seed + i
        set_seed(run_seed)
        q_loader, g_loader = make_loaders()

        if actual_runs > 1:
            print(f"\n  Run {i+1}/{actual_runs}  (seed={run_seed})")

        result = run_reid_once(cfg, model, q_loader, g_loader, device)
        run_results.append(result)
        o = result['overall']
        print(f"  mAP: {o['mAP']:.2%}  Rank-1: {o['rank1']:.2%}"
              f"  Rank-5: {o['rank5']:.2%}  Rank-10: {o['rank10']:.2%}")

    avg = average_reid_runs(run_results)

    if actual_runs > 1:
        print(f"\n  Average over {actual_runs} runs:")
        for m in ['mAP', 'rank1', 'rank5', 'rank10']:
            print(f"    {m:<10}: {avg['overall_avg'][m]:.2%} ± {avg['overall_std'][m]:.2%}")

    return avg, run_results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Basketball Video ReID Testing')
    parser.add_argument('--config-file',  type=str, required=True)
    parser.add_argument('--checkpoint',   type=str, required=True)
    parser.add_argument('--output-dir',   type=str, default='eccv/shuffling_test_results')
    parser.add_argument('--query-ratio',  type=float, default=0.5)
    parser.add_argument('--seed',         type=int, default=None,
                        help='base seed for Q/G split and temporal shuffle runs')
    parser.add_argument('--num-runs',     type=int, default=10,
                        help='number of runs when temporal_shuffle_ratio > 0 (default: 10)')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    # Allow CPU-only runs even when training config was saved with NUM_GPUS > 0.
    if not torch.cuda.is_available():
        cfg.defrost()
        cfg.NUM_GPUS = 0
        cfg.freeze()
    else:
        cfg.freeze()

    base_seed      = args.seed if args.seed is not None else cfg.SEED
    shuffle_ratio  = getattr(cfg.DATA, 'TEMPORAL_SHUFFLE_RATIO', 0.0)
    actual_runs    = args.num_runs if shuffle_ratio > 0.0 else 1

    set_seed(base_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        f"cuda:{cfg.GPU_IDS[0]}" if torch.cuda.is_available() else "cpu"
    )
    checkpoint_dir = os.path.dirname(args.checkpoint)
    prefix = os.path.basename(checkpoint_dir)

    print("=" * 80)
    print("Testing Configuration:")
    print(f"  Config File         : {args.config_file}")
    print(f"  Checkpoint          : {args.checkpoint}")
    print(f"  Video Type          : {cfg.DATA.VIDEO_TYPE}")
    print(f"  Shot Type           : {cfg.DATA.SHOT_TYPE}")
    print(f"  Output Dir          : {args.output_dir}")
    print(f"  Query Ratio         : {args.query_ratio}")
    print(f"  Base Seed           : {base_seed}")
    print(f"  Temporal Shuffle    : {shuffle_ratio}")
    if shuffle_ratio > 0.0:
        print(f"  Num Runs            : {actual_runs}  "
              f"(seeds {base_seed}–{base_seed+actual_runs-1})")
    else:
        print(f"  Num Runs            : 1  (deterministic, no shuffle)")
    print("=" * 80)

    # Dataset & model
    test_loader, num_classes = build_dataloader(cfg, is_train=False)
    cfg.defrost()
    cfg.MODEL.NUM_CLASSES = num_classes
    cfg.freeze()

    print(f"\n  Num Classes : {num_classes}")
    print(f"  Test videos : {len(test_loader.dataset)}")

    model = build_model(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"\n  Loaded checkpoint — epoch {checkpoint['epoch']}")
    if 'rank1' in checkpoint:
        print(f"  Training Best Rank-1 : {checkpoint['rank1']:.2%}  "
              f"mAP: {checkpoint['mAP']:.2%}")

    # Q/G split — fixed with base_seed for all runs
    dataset = test_loader.dataset
    q_idx, g_idx = split_query_gallery(dataset, args.query_ratio, base_seed)

    # ── 1. Classification ────────────────────────────────────────────────────
    cls_results = test_classification(
        cfg, model, test_loader, device,
        num_runs=actual_runs,
        base_seed=base_seed,
        shuffle_ratio=shuffle_ratio,
    )

    print(f"\n  Classification Overall:")
    for kk in [1, 3, 5]:
        acc = cls_results['topk_overall'][kk] / cls_results['total'] * 100
        print(f"    Top-{kk}: {acc:.2f}%  ({cls_results['topk_overall'][kk]}/{cls_results['total']})")

    # ── 2. ReID ──────────────────────────────────────────────────────────────
    avg, run_results = test_reid(
        cfg, model, dataset, q_idx, g_idx, device,
        num_runs=actual_runs,
        base_seed=base_seed,
        shuffle_ratio=shuffle_ratio,
    )

    # ── Save results ──────────────────────────────────────────────────────────

    # 1. Classification detailed CSV
    pd.DataFrame(cls_results['detailed_results']).to_csv(
        os.path.join(args.output_dir, f'{prefix}_detailed_results.csv'), index=False
    )

    # 2. Per-identity classification CSV
    pd.DataFrame.from_dict(
        cls_results['topk_per_identity'], orient='index'
    ).rename_axis('identity').to_csv(
        os.path.join(args.output_dir, f'{prefix}_per_identity_topk.csv')
    )

    # 3. Per-run ReID CSV
    run_rows = [{'run': i+1, 'seed': base_seed+i, **r['overall']}
                for i, r in enumerate(run_results)]
    pd.DataFrame(run_rows).to_csv(
        os.path.join(args.output_dir, f'{prefix}_reid_per_run.csv'), index=False
    )

    # 4. Averaged per-identity ReID CSV
    id_rows = []
    for identity in sorted(avg['per_identity_avg'].keys()):
        row = {'identity': identity}
        for m in ['mAP', 'rank1', 'rank5', 'rank10']:
            row[f'{m}_mean'] = avg['per_identity_avg'][identity][m]
            row[f'{m}_std']  = avg['per_identity_std'][identity][m]
        id_rows.append(row)
    pd.DataFrame(id_rows).set_index('identity').to_csv(
        os.path.join(args.output_dir, f'{prefix}_per_identity_reid_avg.csv')
    )

    # 5. Summary TXT
    summary_file = os.path.join(args.output_dir, f'{prefix}_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("BASKETBALL VIDEO REID - TEST RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Config           : {args.config_file}\n")
        f.write(f"Checkpoint       : {args.checkpoint}\n")
        f.write(f"Epoch            : {checkpoint['epoch']}\n")
        f.write(f"Base Seed        : {base_seed}\n")
        f.write(f"Temporal Shuffle : {shuffle_ratio}\n")
        f.write(f"Num Runs         : {actual_runs}\n")
        f.write(f"Query Ratio      : {args.query_ratio}\n")
        f.write(f"Video type       : {cfg.DATA.VIDEO_TYPE}\n")
        f.write(f"Shot type        : {cfg.DATA.SHOT_TYPE}\n")
        f.write(f"Num frames       : {cfg.DATA.NUM_FRAMES}\n")
        f.write(f"Identities       : {num_classes}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write(f"CLASSIFICATION  ({'1 run, deterministic' if shuffle_ratio == 0.0 else f'{actual_runs} runs, probs averaged'})\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total videos: {cls_results['total']}\n")
        for kk in [1, 3, 5]:
            acc = cls_results['topk_overall'][kk] / cls_results['total'] * 100
            f.write(f"  Top-{kk}: {acc:.2f}%  ({cls_results['topk_overall'][kk]}/{cls_results['total']})\n")

        f.write("\n" + "=" * 60 + "\n")
        suffix = f"mean ± std, {actual_runs} runs" if actual_runs > 1 else "1 run"
        f.write(f"REID METRICS  ({suffix})\n")
        f.write("=" * 60 + "\n")
        f.write(f"Q/G split seed: {base_seed}  (fixed across all runs)\n\n")
        for m in ['mAP', 'rank1', 'rank5', 'rank10']:
            mu  = avg['overall_avg'][m]
            std = avg['overall_std'][m]
            if actual_runs > 1:
                f.write(f"  {m:<10}: {mu:.4f} ± {std:.4f}  ({mu:.2%} ± {std:.2%})\n")
            else:
                f.write(f"  {m:<10}: {mu:.4f}  ({mu:.2%})\n")

        if actual_runs > 1:
            f.write("\nPer-run breakdown:\n")
            f.write(f"  {'Run':>4}  {'Seed':>6}  {'mAP':>8}  {'Rank-1':>8}  {'Rank-5':>8}  {'Rank-10':>8}\n")
            f.write(f"  {'─'*4}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}\n")
            for i, r in enumerate(run_results):
                o = r['overall']
                f.write(f"  {i+1:>4}  {base_seed+i:>6}  {o['mAP']:>8.4f}"
                        f"  {o['rank1']:>8.4f}  {o['rank5']:>8.4f}  {o['rank10']:>8.4f}\n")
            f.write(f"  {'mean':>4}  {'':>6}  "
                    + "  ".join(f"{avg['overall_avg'][m]:>8.4f}"
                                for m in ['mAP', 'rank1', 'rank5', 'rank10']) + "\n")
            f.write(f"  {'std':>4}  {'':>6}  "
                    + "  ".join(f"{avg['overall_std'][m]:>8.4f}"
                                for m in ['mAP', 'rank1', 'rank5', 'rank10']) + "\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("DATASET INFO\n")
        f.write("=" * 60 + "\n")
        f.write(f"Identities : {len(cls_results['topk_per_identity'])}\n")

    print(f"\n{'='*60}")
    print("Saved:")
    print(f"  ✓ {prefix}_detailed_results.csv")
    print(f"  ✓ {prefix}_per_identity_topk.csv")
    print(f"  ✓ {prefix}_reid_per_run.csv")
    print(f"  ✓ {prefix}_per_identity_reid_avg.csv")
    print(f"  ✓ {prefix}_summary.txt")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()