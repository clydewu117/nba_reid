#!/usr/bin/env python
"""
MViT-specific lightweight components to avoid importing Uniformer stack.

Contains:
- ReIDHead: BNNeck + Linear projection + classifier, identical behavior to
  the head used in Uniformerv2ReID but defined locally to remove dependency
  on uniformerv2 modules (and thus timm).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReIDHead(nn.Module):
    """
    ReID Head with BNNeck and feature projection to a target embedding size (default 512).

    Inputs:
    - features: [B, in_dim]
    - label (optional): [B]

    Training returns: (cls_score, global_feat, normalized_feat)
    Eval returns: normalized_feat
    """

    def __init__(self, in_dim: int, num_classes: int, embed_dim: int = 512, neck_feat: str = "after"):
        super().__init__()
        self.num_classes = num_classes
        self.neck_feat = neck_feat
        self.embed_dim = embed_dim

        # Feature projection: in_dim -> embed_dim
        self.feat_proj = nn.Linear(in_dim, embed_dim)

        # BNNeck on projected features
        self.bottleneck = nn.BatchNorm1d(embed_dim)
        self.bottleneck.bias.requires_grad_(False)

        # Classifier
        self.classifier = nn.Linear(embed_dim, num_classes, bias=False)

        self._init_params()

    def _init_params(self):
        nn.init.kaiming_normal_(self.feat_proj.weight, mode="fan_out")
        nn.init.constant_(self.feat_proj.bias, 0)
        nn.init.constant_(self.bottleneck.weight, 1)
        nn.init.constant_(self.bottleneck.bias, 0)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, features: torch.Tensor, label=None):
        # Project to embed_dim
        feat_proj = self.feat_proj(features)  # [B, in_dim] -> [B, embed_dim]

        # BNNeck
        bn_feat = self.bottleneck(feat_proj)

        # Normalized feature for retrieval
        if self.neck_feat == "after":
            feat = F.normalize(bn_feat, p=2, dim=1)
        else:
            feat = F.normalize(feat_proj, p=2, dim=1)

        # Always compute cls_score for GradCAM compatibility
        cls_score = self.classifier(bn_feat)

        if self.training:
            return cls_score, feat_proj, feat
        else:
            # Eval: return dict with both cls_score and feat
            return {"cls_score": cls_score, "feat": feat}
