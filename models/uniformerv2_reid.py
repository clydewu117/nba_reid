#!/usr/bin/env python
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.uniformerv2_model as model
from .build import MODEL_REGISTRY

import utils.logging as logging

logger = logging.get_logger(__name__)


class ReIDHead(nn.Module):
    """
    ReID Head with BNNeck (Single Branch)
    """
    def __init__(self, in_dim, num_classes, neck_feat='after'):
        super().__init__()
        self.num_classes = num_classes
        self.neck_feat = neck_feat
        
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
            Training: dict with 'cls_score', 'bn_feat', 'feat'
            Testing: normalized features [B, 768]
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
            # Testing: return normalized feature
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


@MODEL_REGISTRY.register()
class Uniformerv2ReID(nn.Module):
    """
    UniFormerV2 for ReID
    """
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # -------------------------------
        # 1. 读取基础配置参数
        # -------------------------------
        use_checkpoint = cfg.MODEL.USE_CHECKPOINT
        checkpoint_num = cfg.MODEL.CHECKPOINT_NUM
        num_classes = cfg.MODEL.NUM_CLASSES 
        t_size = cfg.DATA.NUM_FRAMES

        backbone = cfg.UNIFORMERV2.BACKBONE
        n_layers = cfg.UNIFORMERV2.N_LAYERS
        n_dim = cfg.UNIFORMERV2.N_DIM
        n_head = cfg.UNIFORMERV2.N_HEAD
        mlp_factor = cfg.UNIFORMERV2.MLP_FACTOR
        backbone_drop_path_rate = cfg.UNIFORMERV2.BACKBONE_DROP_PATH_RATE
        drop_path_rate = cfg.UNIFORMERV2.DROP_PATH_RATE
        mlp_dropout = cfg.UNIFORMERV2.MLP_DROPOUT
        cls_dropout = cfg.UNIFORMERV2.CLS_DROPOUT
        return_list = cfg.UNIFORMERV2.RETURN_LIST

        temporal_downsample = cfg.UNIFORMERV2.TEMPORAL_DOWNSAMPLE
        dw_reduction = cfg.UNIFORMERV2.DW_REDUCTION
        no_lmhra = cfg.UNIFORMERV2.NO_LMHRA
        double_lmhra = cfg.UNIFORMERV2.DOUBLE_LMHRA

        frozen = cfg.UNIFORMERV2.FROZEN

        # -------------------------------
        # 2. 构建 Backbone
        #    关键：pretrained=False -> 禁止 uniformerv2_model 再次加载 CLIP .pt
        # -------------------------------
        self.backbone = model.__dict__[backbone](
            pretrained=False,                   
            use_checkpoint=use_checkpoint,
            checkpoint_num=checkpoint_num,
            t_size=t_size,
            dw_reduction=dw_reduction,
            backbone_drop_path_rate=backbone_drop_path_rate, 
            temporal_downsample=temporal_downsample,
            no_lmhra=no_lmhra,
            double_lmhra=double_lmhra,
            return_list=return_list, 
            n_layers=n_layers, 
            n_dim=n_dim, 
            n_head=n_head, 
            mlp_factor=mlp_factor, 
            drop_path_rate=drop_path_rate, 
            mlp_dropout=mlp_dropout, 
            cls_dropout=cls_dropout, 
            num_classes=num_classes,
            frozen=frozen,
        )

        self.feat_dim = n_dim

        # -------------------------------
        # 3. ReID Head (BNNeck + Linear)
        # -------------------------------
        self.neck_feat = cfg.REID.NECK_FEAT if hasattr(cfg, 'REID') else 'after'

        self.reid_head = ReIDHead(
            in_dim=self.feat_dim,
            num_classes=num_classes,
            neck_feat=self.neck_feat
        )

        # -------------------------------
        # 4. 加载预训练权重 (只加载一次)
        # -------------------------------
        if cfg.UNIFORMERV2.PRETRAIN != '':
            logger.info(f"[ReID] Load pretrained backbone from {cfg.UNIFORMERV2.PRETRAIN}")
            state_dict = torch.load(cfg.UNIFORMERV2.PRETRAIN, map_location='cpu')

            # 删除分类头层 (proj 层)
            keys_to_remove = [
                k for k in state_dict.keys()
                if 'transformer.proj.' in k or 'backbone.transformer.proj.' in k
            ]
            for k in keys_to_remove:
                logger.info(f"[ReID] Delete proj layer: {k}")
                del state_dict[k]

            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            logger.info(f"[ReID] Loaded pretrained weights: {len(missing)} missing / {len(unexpected)} unexpected")

        # -------------------------------
        # 5. 冻结 Backbone (可选)
        # -------------------------------
        if frozen:
            backbone_list = [
                'conv1', 'class_embedding', 'positional_embedding', 'ln_pre', 'transformer.resblocks'
            ]
            logger.info(f"[ReID] Freeze List: {backbone_list}")
            for name, p in self.backbone.named_parameters():
                p.requires_grad = not any(module in name for module in backbone_list)


    def forward(self, x, label=None):
        """
        Args:
            x: [B, C, T, H, W] video input
            label: [B] person IDs (optional, for training)
        
        Returns:
            Training: dict with 'cls_score', 'bn_feat', 'feat'
            Testing: normalized features [B, 512]
        """
        
        # Extract features from backbone (returns features not logits)
        features = self.backbone(x)  # [B, D]
        
        # Apply ReID head
        if self.training:
            cls_score, bn_feat, feat = self.reid_head(features, label)
            return {
                'cls_score': cls_score,    # [B, num_classes] for ID loss
                'bn_feat': bn_feat, # [B, 512] for triplet loss
                'feat': feat               # [B, 512] normalized for eval
            }
        else:
            return self.reid_head(features)  # [B, 512] normalized