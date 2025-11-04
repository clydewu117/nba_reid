#!/usr/bin/env python
"""
TimeSformer backbone wrapper for ReID within unireid framework.
- Uses TimeSformer VisionTransformer as video backbone to extract a 768-d CLS feature
- Applies BNNeck + Linear projection to 512-d and classifier head for ID loss
- Mirrors the Uniformerv2ReID interface so it can plug into existing training/eval

Requirements:
- Local TimeSformer repository available (default: sibling folder ../TimeSformer)
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
from .timesformer_model import (
    VisionTransformer,
    load_pretrained_timesformer,
)  # use local lightweight implementation
import utils.logging as logging

logger = logging.get_logger(__name__)


def _ensure_timesformer_on_path(repo_path: str = ""):
    """Ensure the official TimeSformer repo is importable when requested.
    - If repo_path is provided and exists, append it to sys.path
    - Else, try sibling '../TimeSformer'
    """
    candidates = []
    if repo_path:
        candidates.append(repo_path)
    # try sibling path relative to this file
    this_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.abspath(os.path.join(this_dir, "..", "..", "TimeSformer")))
    candidates.append(os.path.abspath(os.path.join(this_dir, "..", "TimeSformer")))
    for p in candidates:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
            return p
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

        # -------- Configure backbone --------
        img_size = cfg.DATA.HEIGHT  # assume square; align with transforms
        t_size = cfg.DATA.NUM_FRAMES
        patch_size = cfg.TIMESFORMER.PATCH_SIZE
        attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
        drop_path_rate = cfg.TIMESFORMER.DROP_PATH_RATE
        embed_dim = cfg.TIMESFORMER.EMBED_DIM
        num_classes = (
            cfg.MODEL.NUM_CLASSES
        )  # used for classifier only; we'll set 0 for backbone

        # Instantiate backbone; set num_classes=0 so head is Identity
        impl = getattr(cfg.TIMESFORMER, "BACKBONE_IMPL", "lite") if hasattr(cfg, "TIMESFORMER") else "lite"
        if impl == "official":
            # Import official VisionTransformer
            try:
                from timesformer.models.vit import VisionTransformer as OfficialViT
            except Exception as e:
                logger.warning(f"[ReID][TimeSformer] Failed to import official TimeSformer ({e}), falling back to lite impl.")
                OfficialViT = None

            if OfficialViT is not None:
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
            else:
                # fallback to local lite implementation
                self.backbone = VisionTransformer(
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
        else:
            self.backbone = VisionTransformer(
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

        # Load pretrained if provided
        if cfg.TIMESFORMER.PRETRAIN:
            ckpt_path = cfg.TIMESFORMER.PRETRAIN
            try:
                if impl == "official" and 'OfficialViT' in locals() and isinstance(self.backbone, nn.Module) and self.backbone.__class__.__name__ == 'VisionTransformer':
                    # Use official helper to load and adapt weights
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
                            f"[ReID][TimeSformer] (official) Loaded pretrained weights from {ckpt_path}"
                        )
                    except Exception as ee:
                        logger.warning(f"[ReID][TimeSformer] Official load_pretrained failed: {ee}; trying lite loader...")
                        # minimal fallback: loose load state dict
                        try:
                            state = torch.load(ckpt_path, map_location="cpu")
                            if isinstance(state, dict) and "model_state" in state:
                                state = state["model_state"]
                            elif isinstance(state, dict) and "state_dict" in state:
                                state = state["state_dict"]
                            load_res = self.backbone.load_state_dict(state, strict=False)
                            logger.info(
                                f"[ReID][TimeSformer] (fallback) Loaded with strict=False | missing: {len(load_res.missing_keys)} unexpected: {len(load_res.unexpected_keys)}"
                            )
                        except Exception as e3:
                            logger.warning(f"[ReID][TimeSformer] Fallback load also failed: {e3}")
                else:
                    missing, unexpected = load_pretrained_timesformer(self.backbone, ckpt_path)  # type: ignore
                    logger.info(
                        f"[ReID][TimeSformer] Loaded pretrained weights from {ckpt_path} | missing: {len(missing)} unexpected: {len(unexpected)}"
                    )
            except Exception as e:
                logger.warning(
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
