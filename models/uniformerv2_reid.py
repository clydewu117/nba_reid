#!/usr/bin/env python
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import uniformerv2_model as model
from .build import MODEL_REGISTRY

import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)


class ReIDHead(nn.Module):
    """
    ReID Head with BNNeck and optional feature projection
    """
    def __init__(self, in_dim, num_classes, embed_dim=512, neck_feat='after'):
        super().__init__()
        self.num_classes = num_classes
        self.neck_feat = neck_feat
        self.embed_dim = embed_dim
        self.in_dim = in_dim

        # Feature projection: only if in_dim != embed_dim
        if in_dim != embed_dim:
            self.feat_proj = nn.Linear(in_dim, embed_dim)
        else:
            self.feat_proj = None

        # BNNeck on projected features
        self.bottleneck = nn.BatchNorm1d(embed_dim)
        self.bottleneck.bias.requires_grad_(False)

        # Classifier
        self.classifier = nn.Linear(embed_dim, num_classes, bias=False)

        self._init_params()

    def _init_params(self):
        if self.feat_proj is not None:
            nn.init.kaiming_normal_(self.feat_proj.weight, mode='fan_out')
            nn.init.constant_(self.feat_proj.bias, 0)
        nn.init.constant_(self.bottleneck.weight, 1)
        nn.init.constant_(self.bottleneck.bias, 0)
        nn.init.normal_(self.classifier.weight, std=0.001)
    
    def forward(self, features, label=None):
        """
        Args:
            features: [B, in_dim] global features from backbone
        Returns:
            Training: (cls_score, global_feat, normalized_feat)
            Eval: dict with cls_score and feat for GradCAM compatibility
        """
        # Project if needed
        if self.feat_proj is not None:
            feat_proj = self.feat_proj(features)
        else:
            feat_proj = features

        # BNNeck
        bn_feat = self.bottleneck(feat_proj)

        # Normalized feature for retrieval
        if self.neck_feat == 'after':
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
        self.embed_dim = cfg.REID.EMBED_DIM if hasattr(cfg, 'REID') and hasattr(cfg.REID, 'EMBED_DIM') else 512

        self.reid_head = ReIDHead(
            in_dim=self.feat_dim,
            num_classes=num_classes,
            embed_dim=self.embed_dim,
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
            Training: dict with 'cls_score', 'global_feat', 'feat'
            Testing: normalized features [B, 512]
        """
        
        # Extract features from backbone (returns features not logits)
        features = self.backbone(x)  # [B, D]
        
        # Apply ReID head
        if self.training:
            cls_score, global_feat, feat = self.reid_head(features, label)
            return {
                'cls_score': cls_score,    # [B, num_classes] for ID loss
                'global_feat': global_feat, # [B, 512] for triplet loss
                'feat': feat               # [B, 512] normalized for eval
            }
        else:
            return self.reid_head(features)  # [B, 512] normalized