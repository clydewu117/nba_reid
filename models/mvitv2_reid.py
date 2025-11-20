#!/usr/bin/env python
"""
MViTv2 backbone adapter for unireid (SlowFast implementation as feature extractor).

Goal:
- Instantiate SlowFast's MViT (v1/v2) backbone locally vendored under `unireid/slowfast`.
- Strip classification head and expose a ReID-friendly interface identical to Uniformerv2ReID/TimeSformerReID.
- Do NOT modify other framework logic; only add this backbone module and register it.

Inputs/Outputs contract:
- Input: x [B, C, T, H, W]
- Training output: dict with keys {"cls_score", "global_feat", "feat"}
- Eval output: L2-normalized feature [B, embed_dim]

Config expectations (add to unireid config if needed):
- cfg.MODEL.NAME: "MViTReID"
- cfg.MODEL.ARCH: "mvit" (for logging only)
- cfg.MVIT.PRETRAIN: optional path to SlowFast-style checkpoint
- cfg.MVIT.FROZEN: bool, freeze backbone weights
- cfg.MVIT.USE_MEAN_POOLING: bool, whether to use mean of patch tokens instead of CLS

We use the vendored SlowFast modules under `unireid/slowfast` so no external install is required.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .build import MODEL_REGISTRY


# Use vendored SlowFast
from slowfast.config.defaults import get_cfg as sf_get_cfg
from slowfast.models.video_model_builder import MViT as SF_MViT

import utils.logging as logging

logger = logging.get_logger(__name__)


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


def _build_slowfast_mvit_cfg(ureid_cfg):
	"""Create a minimal SlowFast cfg for building an MViT backbone.

	We keep most defaults and only override shapes and a few key options to match unireid input.
	"""
	cfg = sf_get_cfg()

	# Model basics
	cfg.MODEL.ARCH = "mvit"
	cfg.MODEL.MODEL_NAME = "MViT"
	# We don't use the classification head; set to a valid positive value to avoid construction issues.
	cfg.MODEL.NUM_CLASSES = max(1, int(getattr(ureid_cfg.MODEL, "NUM_CLASSES", 1)))

	# Data shapes to match unireid loader
	cfg.DATA.NUM_FRAMES = int(ureid_cfg.DATA.NUM_FRAMES)
	# Ensure square crop and align TRAIN/TEST sizes (MViT asserts equality)
	crop_size = int(getattr(ureid_cfg.DATA, "HEIGHT", 224))
	cfg.DATA.TRAIN_CROP_SIZE = crop_size
	cfg.DATA.TEST_CROP_SIZE = crop_size
	cfg.DATA.INPUT_CHANNEL_NUM = [3]

	# Enable CLS token outputs; use mean pooling optionally (maps to feature type)
	cfg.MVIT.CLS_EMBED_ON = True
	cfg.MVIT.USE_MEAN_POOLING = bool(getattr(ureid_cfg.MVIT, "USE_MEAN_POOLING", False))

	# Keep standard v2-ish defaults; allow light overrides if present
	for k in [
		"EMBED_DIM",
		"NUM_HEADS",
		"MLP_RATIO",
		"DROPPATH_RATE",
		"DEPTH",
		"PATCH_KERNEL",
		"PATCH_STRIDE",
		"PATCH_PADDING",
		"REL_POS_SPATIAL",
		"REL_POS_TEMPORAL",
		"USE_ABS_POS",
		"SEP_POS_EMBED",
		"USE_FIXED_SINCOS_POS",
		"DIM_MUL",
		"HEAD_MUL",
		"POOL_Q_STRIDE",
		"POOL_KV_STRIDE",
		"POOL_KV_STRIDE_ADAPTIVE",
		"POOL_KVQ_KERNEL",
		"DIM_MUL_IN_ATT",
		"SEPARATE_QKV",
		"NORM_STEM",
	]:
		if hasattr(ureid_cfg.MVIT, k):
			getattr(cfg.MVIT, k)
			cfg.MVIT[k] = ureid_cfg.MVIT[k]

	return cfg


def _load_pretrained_mvit(backbone: nn.Module, ckpt_path: str):
	"""Best-effort load of SlowFast checkpoints, ignoring head weights.

	Supports dict with keys: "model_state" or "state_dict" or raw state dict.
	"""
	if not ckpt_path or not os.path.isfile(ckpt_path):
		logger.warning(f"[ReID][MViT] Pretrained checkpoint not found: {ckpt_path}")
		return ([], [])
	state = torch.load(ckpt_path, map_location="cpu")
	if isinstance(state, dict) and "model_state" in state:
		state = state["model_state"]
	elif isinstance(state, dict) and "state_dict" in state:
		state = state["state_dict"]

	# Strip classification head weights
	drop_keys = [k for k in state.keys() if k.startswith("head.projection")] 
	for k in drop_keys:
		state.pop(k, None)

	missing, unexpected = backbone.load_state_dict(state, strict=False)
	logger.info(
		f"[ReID][MViT] Loaded pretrained: missing={len(missing)} unexpected={len(unexpected)} from {ckpt_path}"
	)
	return missing, unexpected


@MODEL_REGISTRY.register()
class MViTReID(nn.Module):
	"""
	SlowFast MViT(v2) backbone + ReID head wrapper.
	- Uses vendored SlowFast MViT to compute a single global feature per clip
	- Applies BNNeck + projection + classifier via ReIDHead
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg

		# 1) Build SlowFast MViT backbone (classification head will be ignored)
		sf_cfg = _build_slowfast_mvit_cfg(cfg)
		self.backbone = SF_MViT(sf_cfg)

		# 2) Optionally load pretrained weights
		pretrain_path = getattr(cfg.MVIT, "PRETRAIN", "") if hasattr(cfg, "MVIT") else ""
		if pretrain_path:
			try:
				_load_pretrained_mvit(self.backbone, pretrain_path)
			except Exception as e:
				logger.warning(f"[ReID][MViT] Failed to load pretrained weights: {e}")

		# 3) Optionally freeze backbone
		frozen = bool(getattr(cfg.MVIT, "FROZEN", False)) if hasattr(cfg, "MVIT") else False
		if frozen:
			for p in self.backbone.parameters():
				p.requires_grad = False
			logger.info("[ReID][MViT] Backbone parameters are frozen.")

		# 4) ReID head
		# Infer feature dim from head projection in-features (or norm shape)
		try:
			feat_dim = self.backbone.head.projection.in_features
		except Exception:
			feat_dim = (
				self.backbone.norm.normalized_shape[0]
				if hasattr(self.backbone, "norm")
				else 768
			)
		self.feat_dim = int(feat_dim)
		num_classes = int(getattr(cfg.MODEL, "NUM_CLASSES", 0))
		self.neck_feat = cfg.REID.NECK_FEAT if hasattr(cfg, "REID") else "after"
		self.embed_dim = (
			cfg.REID.EMBED_DIM if hasattr(cfg, "REID") and hasattr(cfg.REID, "EMBED_DIM") else 512
		)

		# Check if in classification mode
		is_classification = getattr(cfg.DATA, 'SHOT_CLASSIFICATION', False)
		
		self.reid_head = ReIDHead(
			in_dim=self.feat_dim,
			num_classes=num_classes,
			neck_feat=self.neck_feat,
			is_classification=is_classification,
		)

	@torch.no_grad()
	def _forward_features_eval(self, x):
		"""Feature extraction path (eval) mirroring SlowFast MViT forward up to pre-head."""
		m = self.backbone
		x, bcthw = m.patch_embed(x)
		bcthw = list(bcthw)
		if len(bcthw) == 4:
			bcthw.insert(2, torch.tensor(m.T))
		T, H, W = bcthw[-3], bcthw[-2], bcthw[-1]
		B, N, C = x.shape

		s = 1 if m.cls_embed_on else 0
		if getattr(m, "use_fixed_sincos_pos", False):
			x = x + m.pos_embed[:, s:, :]
		if m.cls_embed_on:
			cls_tokens = m.cls_token.expand(B, -1, -1)
			if getattr(m, "use_fixed_sincos_pos", False):
				cls_tokens = cls_tokens + m.pos_embed[:, :s, :]
			x = torch.cat((cls_tokens, x), dim=1)

		if m.use_abs_pos:
			if m.sep_pos_embed:
				pos_embed = m.pos_embed_spatial.repeat(1, m.patch_dims[0], 1) + torch.repeat_interleave(
					m.pos_embed_temporal, m.patch_dims[1] * m.patch_dims[2], dim=1
				)
				if m.cls_embed_on:
					pos_embed = torch.cat([m.pos_embed_class, pos_embed], 1)
				x = x + m._get_pos_embed(pos_embed, bcthw)
			else:
				x = x + m._get_pos_embed(m.pos_embed, bcthw)

		if m.drop_rate:
			x = m.pos_drop(x)
		if m.norm_stem is not None:
			x = m.norm_stem(x)

		thw = [T, H, W]
		for blk in m.blocks:
			x, thw = blk(x, thw)

		if m.use_mean_pooling:
			if m.cls_embed_on:
				x = x[:, 1:]
			x = x.mean(1)
			x = m.norm(x)
		elif m.cls_embed_on:
			x = m.norm(x)
			x = x[:, 0]
		else:
			x = m.norm(x)
			x = x.mean(1)

		return x

	def _forward_features_train(self, x):
		"""Training path: identical to eval but with autograd enabled."""
		m = self.backbone
		x, bcthw = m.patch_embed(x)
		bcthw = list(bcthw)
		if len(bcthw) == 4:
			bcthw.insert(2, torch.tensor(m.T))
		T, H, W = bcthw[-3], bcthw[-2], bcthw[-1]
		B, N, C = x.shape

		s = 1 if m.cls_embed_on else 0
		if getattr(m, "use_fixed_sincos_pos", False):
			x = x + m.pos_embed[:, s:, :]
		if m.cls_embed_on:
			cls_tokens = m.cls_token.expand(B, -1, -1)
			if getattr(m, "use_fixed_sincos_pos", False):
				cls_tokens = cls_tokens + m.pos_embed[:, :s, :]
			x = torch.cat((cls_tokens, x), dim=1)

		if m.use_abs_pos:
			if m.sep_pos_embed:
				pos_embed = m.pos_embed_spatial.repeat(1, m.patch_dims[0], 1) + torch.repeat_interleave(
					m.pos_embed_temporal, m.patch_dims[1] * m.patch_dims[2], dim=1
				)
				if m.cls_embed_on:
					pos_embed = torch.cat([m.pos_embed_class, pos_embed], 1)
				x = x + m._get_pos_embed(pos_embed, bcthw)
			else:
				x = x + m._get_pos_embed(m.pos_embed, bcthw)

		if m.drop_rate:
			x = m.pos_drop(x)
		if m.norm_stem is not None:
			x = m.norm_stem(x)

		thw = [T, H, W]
		for blk in m.blocks:
			x, thw = blk(x, thw)

		if m.use_mean_pooling:
			if m.cls_embed_on:
				x = x[:, 1:]
			x = x.mean(1)
			x = m.norm(x)
		elif m.cls_embed_on:
			x = m.norm(x)
			x = x[:, 0]
		else:
			x = m.norm(x)
			x = x.mean(1)

		return x

	def forward(self, x, label=None):
		"""
		Args:
			x: [B, C, T, H, W]
			label: Optional [B] IDs for training loss
		Returns:
			Train: dict(cls_score, global_feat, feat)
			Eval: normalized features [B, 512]
		"""
		# SlowFast MViT expects list of pathways; single-pathway input
		if isinstance(x, (list, tuple)):
			x = x[0]

		# Extract features from backbone
		if self.training:
			feats = self._forward_features_train(x)
			cls_score, bn_feat, feat = self.reid_head(feats, label)
			return {"cls_score": cls_score, "bn_feat": bn_feat, "feat": feat}
		else:
			feats = self._forward_features_eval(x)
			return self.reid_head(feats)
