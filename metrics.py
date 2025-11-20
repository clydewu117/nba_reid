#!/usr/bin/env python
"""
Evaluation metrics for Basketball ReID (Rank-1, mAP) and Classification (Accuracy, F1)
Simplified version without camera IDs
"""

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def evaluate_rank(distmat, q_pids, g_pids, max_rank=50):
    """
    Evaluate ranking metrics for ReID.
    
    Args:
        distmat: Distance matrix (num_query x num_gallery)
        q_pids: Query person IDs
        g_pids: Gallery person IDs
        max_rank: Maximum rank to evaluate
    
    Returns:
        cmc: Cumulative Matching Characteristics curve
        mAP: mean Average Precision
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
    
    # Sort gallery by distance for each query
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis])

    all_cmc, all_AP = [], []
    num_valid_q = 0

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        
        # Get matches for this query
        orig_cmc = matches[q_idx]
        
        # Skip if no matches found in gallery
        if not np.any(orig_cmc):
            continue

        # Compute CMC curve
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1

        # Compute Average Precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        precision = tmp_cmc / (np.arange(len(tmp_cmc)) + 1)
        AP = (precision * orig_cmc).sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    
    # Average over all valid queries
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    
    return all_cmc, mAP


class R1_mAP_eval(object):
    """
    Evaluator for Rank-1, Rank-5, Rank-10, and mAP metrics.
    Simplified version without camera IDs.
    """
    def __init__(self, num_query, max_rank=50, feat_norm=True):
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reset()

    def reset(self):
        """Reset internal storage."""
        self.feats, self.pids = [], []

    def update(self, data):
        """
        Update with new batch of features.
        
        Args:
            data: Tuple of (features, person_ids)
                  Note: camid is no longer required
        """
        # Support both (feat, pid) and (feat, pid, camid) for backward compatibility
        if len(data) == 3:
            feat, pid, _ = data  # Ignore camid if provided
        else:
            feat, pid = data
            
        if self.feat_norm:
            feat = torch.nn.functional.normalize(feat, dim=1, p=2)
        self.feats.append(feat.cpu())
        self.pids.extend(pid.cpu().numpy())

    def compute(self):
        """
        Compute evaluation metrics.
        
        Returns:
            cmc: CMC curve (numpy array of length max_rank)
            mAP: mean Average Precision (float)
        """
        feats = torch.cat(self.feats, dim=0)
        qf = feats[:self.num_query]
        gf = feats[self.num_query:]
        q_pids = np.asarray(self.pids[:self.num_query])
        g_pids = np.asarray(self.pids[self.num_query:])

        distmat = cdist(qf, gf, metric='euclidean')
        cmc, mAP = evaluate_rank(distmat, q_pids, g_pids, self.max_rank)
        return cmc, mAP


class ClassificationEvaluator(object):
    """
    Evaluator for binary classification task (e.g., freethrow vs 3pt).
    Computes Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
    """
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset internal storage."""
        self.preds, self.labels = [], []

    def update(self, pred, label):
        """
        Update with new batch of predictions.
        
        Args:
            pred: Predicted logits or probabilities (batch_size, num_classes)
            label: Ground truth labels (batch_size,)
        """
        # Convert logits to predictions
        if pred.dim() == 2:
            pred = torch.argmax(pred, dim=1)
        
        self.preds.extend(pred.cpu().numpy())
        self.labels.extend(label.cpu().numpy())

    def compute(self):
        """
        Compute classification metrics.
        
        Returns:
            metrics: Dictionary containing:
                - accuracy: Overall accuracy
                - precision: Per-class precision (macro-averaged)
                - recall: Per-class recall (macro-averaged)
                - f1: Per-class F1-score (macro-averaged)
                - confusion_matrix: Confusion matrix
        """
        preds = np.asarray(self.preds)
        labels = np.asarray(self.labels)
        
        # Compute metrics
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='macro', zero_division=0
        )
        cm = confusion_matrix(labels, preds)
        
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }