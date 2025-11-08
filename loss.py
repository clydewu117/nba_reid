#!/usr/bin/env python
"""
Loss functions for Basketball ReID (CrossEntropy + Triplet)
Fixed: Handle uneven sample distribution per class in hard_example_mining
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Label Smoothing Cross Entropy
# -----------------------------
class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# -----------------------------
# Helper Functions
# -----------------------------
def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


def hard_example_mining(dist_mat, labels):
    """
    For each anchor, find the hardest positive and negative sample.
    Fixed to handle cases where each class has different number of samples.
    
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # 逐个样本处理，而不是用view
    # 因为每个class的样本数可能不同
    dist_ap = []
    dist_an = []
    
    for i in range(N):
        # 对每个anchor，找最难的正负样本
        pos_dists = dist_mat[i][is_pos[i]]  # 该样本的所有正样本距离
        neg_dists = dist_mat[i][is_neg[i]]  # 该样本的所有负样本距离
        
        # 最远正样本（最难）
        if pos_dists.numel() > 0:
            dist_ap.append(pos_dists.max().unsqueeze(0))
        else:
            # 没有正样本，用一个很大的值
            dist_ap.append(torch.tensor([1e6], device=dist_mat.device))
        
        # 最近负样本（最难）
        if neg_dists.numel() > 0:
            dist_an.append(neg_dists.min().unsqueeze(0))
        else:
            # 没有负样本，用一个很小的值
            dist_an.append(torch.tensor([0.0], device=dist_mat.device))
    
    dist_ap = torch.cat(dist_ap)
    dist_an = torch.cat(dist_an)

    return dist_ap, dist_an


# -----------------------------
# Triplet Loss (BatchHard)
# -----------------------------
class TripletLoss(nn.Module):
    """
    Triplet loss using HARD example mining
    """
    def __init__(self, margin=0.3, hard_factor=0.0):
        super().__init__()
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = F.normalize(global_feat, p=2, dim=-1)
        
        global_feat = global_feat.float()
        
        # Compute pairwise distance
        dist_mat = euclidean_dist(global_feat, global_feat)
        
        # Hard example mining
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)
        
        # Optional: adjust distances with hard factor
        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)
        
        # 过滤掉无效的样本
        valid_mask = (dist_ap < 1e6) & (dist_an > 0)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=global_feat.device, requires_grad=True)
        
        dist_ap = dist_ap[valid_mask]
        dist_an = dist_an[valid_mask]
        
        # Compute ranking loss
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        
        return loss


# -----------------------------
# Combined Loss Function
# -----------------------------
class CombinedLoss(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        
        self.id_loss = LabelSmoothingCrossEntropy(
            smoothing=cfg.LOSS.LABEL_SMOOTH_EPSILON
        )
        
        self.use_triplet = cfg.LOSS.USE_TRIPLET
        if self.use_triplet:
            self.triplet_loss = TripletLoss(margin=cfg.LOSS.TRIPLET_MARGIN)
        
        self.id_weight = cfg.LOSS.ID_WEIGHT
        self.triplet_weight = cfg.LOSS.TRIPLET_WEIGHT

    def forward(self, cls_score, global_feat, feat, labels):
        id_loss_val = self.id_loss(cls_score, labels)
        loss = self.id_weight * id_loss_val
        loss_dict = {"id_loss": id_loss_val}

        if self.use_triplet:
            triplet_loss_val = self.triplet_loss(global_feat, labels, normalize_feature=True)
            loss += self.triplet_weight * triplet_loss_val
            loss_dict["triplet_loss"] = triplet_loss_val

        return loss, loss_dict


def make_loss(cfg, num_classes):
    return CombinedLoss(cfg, num_classes)