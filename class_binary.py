#!/usr/bin/env python
"""
Basketball Video ReID - Binary Classification Testing Script
"""

import os
import random
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

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


@torch.no_grad()
def test_binary_classification(cfg, model, test_loader, device):
    """测试二分类性能（使用全部测试集）"""
    model.eval()
    
    all_pids = []
    all_probs = []
    all_preds = []
    all_paths = []
    all_identities = []
    
    print(f"\n{'='*60}")
    print(f"Binary Classification Testing...")
    print(f"{'='*60}\n")
    
    for batch in tqdm(test_loader, desc="Extracting"):
        videos = batch['video'].to(device)
        pids = batch['pid'].to(device)
        
        outputs = model(videos)

        if isinstance(outputs, dict):
            if "cls_score" in outputs:
                cls_score = outputs["cls_score"]
            elif "logits" in outputs:
                cls_score = outputs["logits"]
            elif "bn_feat" in outputs:
                cls_score = model.reid_head.classifier(outputs["bn_feat"])
            else:
                raise RuntimeError("Model eval output dict missing classification logits.")
        elif isinstance(outputs, (list, tuple)):
            cls_score = outputs[0]
        else:
            cls_score = outputs

        probs = F.softmax(cls_score, dim=1)
        
        # For binary classification, predict class with highest probability
        preds = torch.argmax(probs, dim=1)
        
        all_pids.append(pids.cpu())
        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_paths.extend(batch['video_paths'])
        all_identities.extend(batch['identities'])
    
    all_pids = torch.cat(all_pids, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    
    # Calculate overall metrics
    true_labels = all_pids.numpy()
    pred_labels = all_preds.numpy()
    
    # Overall accuracy
    correct = (pred_labels == true_labels).sum()
    total = len(true_labels)
    accuracy = correct / total * 100
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
    
    # Per-class metrics
    tn, fp, fn, tp = cm.ravel()
    
    # Precision, Recall, F1 for each class
    precision_class0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    recall_class0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1_class0 = 2 * precision_class0 * recall_class0 / (precision_class0 + recall_class0) if (precision_class0 + recall_class0) > 0 else 0.0
    
    precision_class1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_class1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_class1 = 2 * precision_class1 * recall_class1 / (precision_class1 + recall_class1) if (precision_class1 + recall_class1) > 0 else 0.0
    
    # Macro-averaged metrics
    macro_precision = (precision_class0 + precision_class1) / 2
    macro_recall = (recall_class0 + recall_class1) / 2
    macro_f1 = (f1_class0 + f1_class1) / 2
    
    # Print results
    print(f"\n{'='*60}")
    print("Binary Classification Results:")
    print(f"{'='*60}")
    print(f"Overall:")
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"  Macro-Precision: {macro_precision:.4f}")
    print(f"  Macro-Recall: {macro_recall:.4f}")
    print(f"  Macro-F1: {macro_f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"               Predicted")
    print(f"             Class 0  Class 1")
    print(f"Actual")
    print(f"Class 0      {cm[0,0]:6d}   {cm[0,1]:6d}")
    print(f"Class 1      {cm[1,0]:6d}   {cm[1,1]:6d}")
    
    print(f"\nPer-Class Metrics:")
    print(f"  Class 0:")
    print(f"    Precision: {precision_class0:.4f}")
    print(f"    Recall: {recall_class0:.4f}")
    print(f"    F1-Score: {f1_class0:.4f}")
    print(f"  Class 1:")
    print(f"    Precision: {precision_class1:.4f}")
    print(f"    Recall: {recall_class1:.4f}")
    print(f"    F1-Score: {f1_class1:.4f}")
    
    # Get class names based on classification mode
    is_shot_classification = getattr(cfg.DATA, 'SHOT_CLASSIFICATION', False)
    if is_shot_classification:
        # Shot classification mode: freethrow vs 3pt
        class0_name = 'freethrow'
        class1_name = '3pt'
    else:
        # Identity ReID mode
        class0_name = test_loader.dataset.pid_list[0] if len(test_loader.dataset.pid_list) > 0 else 'Unknown'
        class1_name = test_loader.dataset.pid_list[1] if len(test_loader.dataset.pid_list) > 1 else 'Unknown'
    
    # Per-identity metrics
    print(f"\nPer-Identity Accuracy:")
    identity_metrics = {}
    
    for identity in sorted(set(all_identities)):
        identity_indices = [i for i, x in enumerate(all_identities) if x == identity]
        identity_total = len(identity_indices)
        identity_correct = sum(1 for idx in identity_indices if pred_labels[idx] == true_labels[idx])
        identity_acc = identity_correct / identity_total * 100
        
        identity_metrics[identity] = {
            'total': identity_total,
            'correct': identity_correct,
            'accuracy': identity_acc,
            'true_class': true_labels[identity_indices[0]]
        }
        
        print(f"  {identity} ({identity_total} videos, Class {identity_metrics[identity]['true_class']}):")
        print(f"    Accuracy: {identity_acc:.2f}%")
    
    # Detailed per-video results
    detailed_results = []
    for i in range(total):
        true_pid = all_pids[i].item()
        pred_pid = all_preds[i].item()
        true_identity = all_identities[i]
        video_path = all_paths[i]
        video_name = os.path.basename(video_path)
        
        # Get predicted identity name
        pred_identity = test_loader.dataset.pid_list[pred_pid]
        
        is_correct = (pred_pid == true_pid)
        
        result_row = {
            'video_path': video_path,
            'video_name': video_name,
            'true_identity': true_identity,
            'true_class': true_pid,
            'pred_identity': pred_identity,
            'pred_class': pred_pid,
            'correct': is_correct,
            'class0_prob': all_probs[i, 0].item(),
            'class1_prob': all_probs[i, 1].item(),
            'true_class_prob': all_probs[i, true_pid].item(),
        }
        
        detailed_results.append(result_row)
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'confusion_matrix': cm,
        'precision_class0': precision_class0,
        'recall_class0': recall_class0,
        'f1_class0': f1_class0,
        'precision_class1': precision_class1,
        'recall_class1': recall_class1,
        'f1_class1': f1_class1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'identity_metrics': identity_metrics,
        'detailed_results': detailed_results,
        'class_mapping': {0: class0_name, 1: class1_name}
    }


def main():
    parser = argparse.ArgumentParser(description='Basketball Video Binary Classification Testing')
    parser.add_argument('--config-file', type=str, required=True,
                       help='path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='path to checkpoint')
    parser.add_argument('--output-dir', type=str, default='test_results_binary',
                       help='output directory for results')
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
    print("Binary Classification Testing Configuration:")
    print(f"  Config File  : {args.config_file}")
    print(f"  Checkpoint   : {args.checkpoint}")
    print(f"  Prefix       : {prefix}")
    print(f"  Video Type   : {cfg.DATA.VIDEO_TYPE}")
    print(f"  Shot Type    : {cfg.DATA.SHOT_TYPE}")
    print(f"  Output Dir   : {args.output_dir}")
    print(f"  Random Seed  : {seed}")
    print("=" * 80)

    # Build full test dataloader
    test_loader, num_classes = build_dataloader(cfg, is_train=False)
    
    # Verify binary classification
    if num_classes != 2:
        print(f"\n{'='*60}")
        print(f"WARNING: Expected 2 classes for binary classification, but got {num_classes}")
        print(f"This script is designed for binary classification only!")
        print(f"{'='*60}\n")
    
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

    # Test Binary Classification (on full test set)
    cls_results = test_binary_classification(cfg, model, test_loader, device)
    
    # Save results
    print(f"\n{'='*60}")
    print("Saving results...")
    print(f"{'='*60}\n")
    
    # 1. Detailed per-video classification results
    detailed_df = pd.DataFrame(cls_results['detailed_results'])
    detailed_csv = os.path.join(args.output_dir, f'{prefix}_detailed_results.csv')
    detailed_df.to_csv(detailed_csv, index=False)
    print(f"✓ Detailed per-video results saved to: {detailed_csv}")
    
    # 2. Per-identity accuracy
    identity_metrics_df = pd.DataFrame.from_dict(cls_results['identity_metrics'], orient='index')
    identity_metrics_df.index.name = 'identity'
    identity_csv = os.path.join(args.output_dir, f'{prefix}_per_identity_metrics.csv')
    identity_metrics_df.to_csv(identity_csv)
    print(f"✓ Per-identity metrics saved to: {identity_csv}")
    
    # 3. Overall summary
    summary_file = os.path.join(args.output_dir, f'{prefix}_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("BASKETBALL VIDEO REID - BINARY CLASSIFICATION TEST RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Config: {args.config_file}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Checkpoint Epoch: {checkpoint['epoch']}\n")
        f.write(f"Random Seed: {seed}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("BINARY CLASSIFICATION METRICS (Full Test Set)\n")
        f.write("="*60 + "\n")
        f.write(f"Total test videos: {cls_results['total']}\n")
        f.write(f"Correct predictions: {cls_results['correct']}\n")
        f.write(f"Overall Accuracy: {cls_results['accuracy']:.2f}%\n\n")
        
        f.write("Overall Metrics:\n")
        f.write(f"  Macro-Precision: {cls_results['macro_precision']:.4f}\n")
        f.write(f"  Macro-Recall: {cls_results['macro_recall']:.4f}\n")
        f.write(f"  Macro-F1: {cls_results['macro_f1']:.4f}\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write("               Predicted\n")
        f.write("             Class 0  Class 1\n")
        f.write("Actual\n")
        cm = cls_results['confusion_matrix']
        f.write(f"Class 0      {cm[0,0]:6d}   {cm[0,1]:6d}\n")
        f.write(f"Class 1      {cm[1,0]:6d}   {cm[1,1]:6d}\n\n")
        
        f.write("Per-Class Metrics:\n")
        f.write(f"  Class 0:\n")
        f.write(f"    Precision: {cls_results['precision_class0']:.4f}\n")
        f.write(f"    Recall: {cls_results['recall_class0']:.4f}\n")
        f.write(f"    F1-Score: {cls_results['f1_class0']:.4f}\n")
        f.write(f"  Class 1:\n")
        f.write(f"    Precision: {cls_results['precision_class1']:.4f}\n")
        f.write(f"    Recall: {cls_results['recall_class1']:.4f}\n")
        f.write(f"    F1-Score: {cls_results['f1_class1']:.4f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("DATASET INFO\n")
        f.write("="*60 + "\n")
        f.write(f"Number of classes: {num_classes}\n")
        f.write(f"Class 0: {cls_results['class_mapping'][0]}\n")
        f.write(f"Class 1: {cls_results['class_mapping'][1]}\n")
        f.write(f"Video type: {cfg.DATA.VIDEO_TYPE}\n")
        f.write(f"Shot type: {cfg.DATA.SHOT_TYPE}\n")
        f.write(f"Num frames: {cfg.DATA.NUM_FRAMES}\n")
    
    print(f"✓ Summary saved to: {summary_file}")
    
    print(f"\n{'='*60}")
    print("All results saved successfully!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
