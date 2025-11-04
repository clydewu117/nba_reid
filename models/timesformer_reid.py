#!/usr/bin/env python
"""
TimeSformer backbone wrapper for ReID within unireid framework.
- Uses TimeSformer VisionTransformer as video backbone to extract a 768-d CLS feature
- Applies BNNeck + Linear projection to 512-d and classifier head for ID loss
- Mirrors the Uniformerv2ReID interface so it can plug into existing training/eval

Requirements:
- Bundled official TimeSformer code under nba_reid/timesformer (used by default)
- Optional: TIMESFORMER.REPO_PATH to an external repo explicitly provided by user
- Dependencies: einops, fvcore (already used), etc.

Config namespace expected (add in config/defaults.py):
- MODEL.ARCH == 'timesformer' to indicate this backbone (optional)
- TIMESFORMER.REPO_PATH: optional absolute path to TimeSformer repo
- TIMESFORMER.PRETRAIN: optional pretrained checkpoint path (.pyth or .pth accepted by TimeSformer helpers)
- TIMESFORMER.ATTENTION_TYPE: 'divided_space_time' | 'space_only' | 'joint_space_time'
- TIMESFORMER.PATCH_SIZE: patch size, default 16
- TIMESFORMER.DROP_PATH_RATE: stochastic depth rate, default 0.1 (used by TS backbone)
- TIMESFORMER.EMBED_DIM: expected backbone embed dim (default 768)
- TIMESFORMER.FROZEN: whether to freeze the backbone parameters

Input shape: [B, C, T, H, W]
"""
import os
import sys
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from .build import MODEL_REGISTRY
from .uniformerv2_reid import ReIDHead  # reuse the BNNeck head
import utils.logging as logging

logger = logging.get_logger(__name__)


def _ensure_timesformer_on_path(repo_path: str = ""):
    """Make sure the top-level module name 'timesformer' can be imported.

    Behavior (no implicit external fallback):
    - If `repo_path` is provided and contains a 'timesformer' package, add its parent directory to sys.path.
    - Otherwise, expose the bundled copy at nba_reid/timesformer by adding the nba_reid directory to sys.path.

    This avoids searching arbitrary sibling '../TimeSformer' locations and keeps resolution deterministic.
    """
    # If user explicitly points to an external repo, honor it (but require it contains 'timesformer')
    if repo_path:
        repo_path = os.path.abspath(repo_path)
        parent = repo_path
        # If the path itself is a 'timesformer' folder, use its parent; else assume it contains that subfolder
        if os.path.basename(parent).lower() == "timesformer":
            parent = os.path.dirname(parent)
        if os.path.isdir(os.path.join(parent, "timesformer")):
            if parent not in sys.path:
                sys.path.insert(0, parent)
            return parent
        # If provided but invalid, fall through to bundled copy

    # Add nba_reid directory (so that nba_reid/timesformer is importable as top-level 'timesformer')
    this_dir = os.path.dirname(os.path.abspath(__file__))  # .../nba_reid/models
    nba_reid_dir = os.path.abspath(os.path.join(this_dir, ".."))  # .../nba_reid
    if os.path.isdir(os.path.join(nba_reid_dir, "timesformer")) and nba_reid_dir not in sys.path:
        sys.path.insert(0, nba_reid_dir)
        return nba_reid_dir
    return None


@MODEL_REGISTRY.register()
class TimeSformerReID(nn.Module):
    """
    TimeSformer backbone + ReID head wrapper.
    Uses TimeSformer VisionTransformer.forward_features to get CLS embedding (embed_dim)
    and then applies ReIDHead to produce cls_score and retrieval features.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Resolve TimeSformer import path
        _ = _ensure_timesformer_on_path(
            getattr(cfg, "TIMESFORMER", {}).get("REPO_PATH", "")
            if hasattr(cfg, "TIMESFORMER")
            else ""
        )

        # -------- Configure backbone (official TimeSformer only) --------
        img_size = cfg.DATA.HEIGHT  # assume square; align with transforms
        t_size = cfg.DATA.NUM_FRAMES
        patch_size = cfg.TIMESFORMER.PATCH_SIZE
        attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
        drop_path_rate = cfg.TIMESFORMER.DROP_PATH_RATE
        embed_dim = cfg.TIMESFORMER.EMBED_DIM
        num_classes = (
            cfg.MODEL.NUM_CLASSES
        )  # used for classifier only; we'll set 0 for backbone

        # Instantiate official VisionTransformer; set num_classes=0 so head is Identity
        try:
            from timesformer.models.vit import VisionTransformer as OfficialViT
        except Exception as e:
            raise ImportError(
                f"[ReID][TimeSformer] Failed to import official TimeSformer modules: {e}. "
                "Ensure 'nba_reid/timesformer' exists or set TIMESFORMER.REPO_PATH to a valid TimeSformer repo."
            )

        self.backbone = OfficialViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            num_classes=0,
            embed_dim=embed_dim,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=drop_path_rate,
            num_frames=t_size,
            attention_type=attention_type,
            dropout=0.0,
        )

        # Default cfg key for pretrained interpolation
        # No external default cfg required; local model handles positional/time embedding resize

        # Load pretrained if provided (official helper only)
        if cfg.TIMESFORMER.PRETRAIN:
            ckpt_path = cfg.TIMESFORMER.PRETRAIN
            try:
                from timesformer.models.helpers import load_pretrained as tsf_load_pretrained
                from timesformer.models.vit import default_cfgs as tsf_default_cfgs
                from timesformer.models.vit import _conv_filter as tsf_conv_filter
                num_patches = (img_size // patch_size) * (img_size // patch_size)
                tsf_load_pretrained(
                    self.backbone,
                    cfg=tsf_default_cfgs['vit_base_patch16_224'],
                    num_classes=0,
                    in_chans=3,
                    filter_fn=tsf_conv_filter,
                    img_size=img_size,
                    num_frames=t_size,
                    num_patches=num_patches,
                    attention_type=attention_type,
                    pretrained_model=ckpt_path,
                    strict=False,
                )
                logger.info(
                    f"[ReID][TimeSformer] Loaded pretrained weights from {ckpt_path}"
                )
            except Exception as e:
                raise RuntimeError(
                    f"[ReID][TimeSformer] Failed to load pretrained weights from {ckpt_path}: {e}"
                )

        # Optionally freeze backbone
        if cfg.TIMESFORMER.FROZEN:
            for name, p in self.backbone.named_parameters():
                p.requires_grad = False
            logger.info("[ReID][TimeSformer] Backbone parameters are frozen.")

        # ReID head
        self.feat_dim = embed_dim
        self.neck_feat = cfg.REID.NECK_FEAT if hasattr(cfg, "REID") else "after"
        self.embed_dim = (
            cfg.REID.EMBED_DIM
            if hasattr(cfg, "REID") and hasattr(cfg.REID, "EMBED_DIM")
            else 512
        )
        self.reid_head = ReIDHead(
            in_dim=self.feat_dim,
            num_classes=num_classes,
            embed_dim=self.embed_dim,
            neck_feat=self.neck_feat,
        )

    def forward(self, x, label=None):
        """
        Args:
            x: [B, C, T, H, W]
            label: [B] person IDs (optional)
        Returns (training): dict(cls_score, global_feat, feat)
                (eval): L2-normalized 512-d feature
        """
        # Extract CLS embedding from backbone
        features = self.backbone.forward_features(x)  # [B, embed_dim]

        if self.training:
            cls_score, global_feat, feat = self.reid_head(features, label)
            return {
                "cls_score": cls_score,
                "global_feat": global_feat,
                "feat": feat,
            }
        else:
            return self.reid_head(features)
