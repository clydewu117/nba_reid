#!/usr/bin/env python
"""
VideoMAEv2 backbone wrapper with ReID head for the nba_reid training pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .build import MODEL_REGISTRY
from .videomaev2_model import (
    VisionTransformer,
    build_videomaev2_backbone,
    load_videomaev2_checkpoint,
)

import utils.logging as logging

logger = logging.get_logger(__name__)


class VideoMAEReIDHead(nn.Module):
    """
    BNNeck-style head with optional projection to a target embedding dimension.
    """

    def __init__(self, in_dim: int, num_classes: int, embed_dim: int = 512, neck_feat: str = "after") -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.neck_feat = neck_feat

        self.feat_proj = nn.Linear(in_dim, embed_dim)
        self.bottleneck = nn.BatchNorm1d(embed_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(embed_dim, num_classes, bias=False)

        self._init_params()

    def _init_params(self) -> None:
        nn.init.kaiming_normal_(self.feat_proj.weight, mode="fan_out")
        nn.init.constant_(self.feat_proj.bias, 0)
        nn.init.constant_(self.bottleneck.weight, 1)
        nn.init.constant_(self.bottleneck.bias, 0)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, features: torch.Tensor, label=None):
        feat_proj = self.feat_proj(features)
        bn_feat = self.bottleneck(feat_proj)

        if self.neck_feat == "after":
            feat = F.normalize(bn_feat, p=2, dim=1)
        else:
            feat = F.normalize(feat_proj, p=2, dim=1)

        # Always compute cls_score for GradCAM compatibility
        cls_score = self.classifier(bn_feat)

        if self.training:
            return cls_score, bn_feat, feat
        else:
            # Eval: return dict with both cls_score and feat
            return {"cls_score": cls_score, "feat": feat}


@MODEL_REGISTRY.register()
class VideoMAEv2ReID(nn.Module):
    """
    VideoMAEv2 backbone + ReID head wrapper compatible with nba_reid training.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        if not hasattr(cfg, "VIDEOMAEV2"):
            raise AttributeError("Config missing VIDEOMAEV2 node required for VideoMAEv2ReID.")
        vm_cfg = cfg.VIDEOMAEV2

        img_size = int(getattr(cfg.DATA, "HEIGHT", 224))
        width = int(getattr(cfg.DATA, "WIDTH", img_size))
        if img_size != width:
            raise ValueError("VideoMAE backbone expects square inputs: set DATA.HEIGHT == DATA.WIDTH.")
        num_frames = int(getattr(cfg.DATA, "NUM_FRAMES", 16))
        tubelet_size = int(getattr(vm_cfg, "TUBELET_SIZE", 2))

        arch = getattr(vm_cfg, "MODEL", "vit_base_patch16_224")
        drop_rate = float(getattr(vm_cfg, "DROP_RATE", 0.0))
        attn_drop_rate = float(getattr(vm_cfg, "ATTN_DROP_RATE", 0.0))
        drop_path_rate = float(getattr(vm_cfg, "DROP_PATH_RATE", 0.0))
        head_drop_rate = float(getattr(vm_cfg, "HEAD_DROP_RATE", 0.0))
        use_mean_pooling = bool(getattr(vm_cfg, "USE_MEAN_POOLING", True))
        init_scale = float(getattr(vm_cfg, "INIT_SCALE", 0.0))
        with_checkpoint = bool(getattr(vm_cfg, "WITH_CHECKPOINT", False))
        cos_attn = bool(getattr(vm_cfg, "COS_ATTENTION", False))

        # Build backbone
        self.backbone: VisionTransformer = build_videomaev2_backbone(
            arch=arch,
            img_size=img_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            head_drop_rate=head_drop_rate,
            use_mean_pooling=use_mean_pooling,
            init_scale=init_scale,
            with_checkpoint=with_checkpoint,
            cos_attn=cos_attn,
        )

        if hasattr(self.backbone, "reset_classifier"):
            self.backbone.reset_classifier(0)

        # Load pretrained weights (optional)
        pretrain_path = getattr(vm_cfg, "PRETRAIN", "")
        if pretrain_path:
            model_key = getattr(vm_cfg, "MODEL_KEY", "model|module|state_dict")
            missing, unexpected = load_videomaev2_checkpoint(
                self.backbone,
                pretrain_path,
                num_frames=num_frames,
                model_key=model_key,
                logger=logger,
            )
            logger.info(
                "[VideoMAE] Loaded pretrained weights from %s (missing=%d, unexpected=%d)",
                pretrain_path,
                len(missing),
                len(unexpected),
            )

        # Optionally freeze backbone
        if bool(getattr(vm_cfg, "FROZEN", False)):
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info("[VideoMAE] Backbone parameters are frozen.")

        in_dim = getattr(self.backbone, "num_features", getattr(self.backbone, "embed_dim", 768))
        num_classes = int(getattr(cfg.MODEL, "NUM_CLASSES", 0))
        neck_feat = getattr(cfg.REID, "NECK_FEAT", "after") if hasattr(cfg, "REID") else "after"
        embed_dim = getattr(cfg.REID, "EMBED_DIM", 512) if hasattr(cfg, "REID") else 512

        self.reid_head = VideoMAEReIDHead(
            in_dim=in_dim,
            num_classes=num_classes,
            embed_dim=embed_dim,
            neck_feat=neck_feat,
        )

    def forward(self, x: torch.Tensor, label=None):
        features = self.backbone.forward_features(x)
        if self.training:
            cls_score, bn_feat, feat = self.reid_head(features, label)
            return {
                "cls_score": cls_score,
                "bn_feat": bn_feat,
                "feat": feat,
            }
        else:
            # Eval: reid_head returns dict with cls_score and feat
            return self.reid_head(features)

    def no_weight_decay(self):
        if hasattr(self.backbone, "no_weight_decay"):
            return self.backbone.no_weight_decay()
        return set()

    def get_num_layers(self):
        if hasattr(self.backbone, "get_num_layers"):
            return self.backbone.get_num_layers()
        return 0
