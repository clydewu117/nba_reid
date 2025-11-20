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
    ReID Head with BNNeck (Single Branch)
    """
    def __init__(self, in_dim, num_classes, neck_feat='after', is_classification=False):
        super().__init__()
        self.num_classes = num_classes
        self.neck_feat = neck_feat
        self.is_classification = is_classification
        
        # BNNeck
        self.bottleneck = nn.BatchNorm1d(in_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        
        # Classifier
        self.classifier = nn.Linear(in_dim, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
    
    def forward(self, features, label=None):
        """
        Args:
            features: [B, 768] global features from backbone
        Returns:
            Training: (cls_score, bn_feat, feat)
            Testing (ReID): normalized features [B, 768]
            Testing (Classification): cls_score [B, num_classes]
        """
        # BNNeck
        bn_feat = self.bottleneck(features)  # [B, 768]
        
        if self.neck_feat == 'after':
            feat = F.normalize(bn_feat, p=2, dim=1)
        else:
            feat = F.normalize(features, p=2, dim=1)

        if self.training:
            cls_score = self.classifier(bn_feat)
            return cls_score, bn_feat, feat
        else:
            # Testing mode
            if self.is_classification:
                # Classification task: return logits for accuracy computation
                cls_score = self.classifier(bn_feat)
                return cls_score
            else:
                # ReID task: return normalized feature for distance computation
                if self.neck_feat == 'after':
                    return F.normalize(bn_feat, p=2, dim=1)
                else:
                    return F.normalize(features, p=2, dim=1)


# 添加初始化函数
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
