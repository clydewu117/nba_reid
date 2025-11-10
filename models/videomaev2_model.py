#!/usr/bin/env python
"""
VideoMAEv2 backbone components ported from the official repository for use
inside the nba_reid training framework.

This module exposes:
    - VisionTransformer: backbone implementation with 3D patch embedding.
    - Factory helpers to instantiate common backbone variants (small/base/large...).
    - Utility to load finetuning checkpoints with positional embedding interpolation.
"""

from __future__ import annotations

import os
from functools import partial
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

Tensor = torch.Tensor


def _cfg(url: str = "", **kwargs) -> Dict:
    return {
        "url": url,
        "num_classes": 400,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        **kwargs,
    }


class DropPath(nn.Module):
    """Stochastic depth per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CosAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_head_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = attn_head_dim or (dim // num_heads)
        all_head_dim = head_dim * self.num_heads

        if qk_scale is None:
            self.scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        else:
            self.scale = qk_scale

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias)
            )
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(self.scale, max=4.6052).exp()
        attn = attn * logit_scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_head_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = attn_head_dim or (dim // num_heads)
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias)
            )
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        attn_head_dim: Optional[int] = None,
        cos_attn: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        if cos_attn:
            self.attn = CosAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                attn_head_dim=attn_head_dim,
            )
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                attn_head_dim=attn_head_dim,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        else:
            self.gamma_1 = None
            self.gamma_2 = None

    def forward(self, x: Tensor) -> Tensor:
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Video to patch embedding with (tubelet_size, patch_size, patch_size) kernel."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        num_frames: int = 16,
        tubelet_size: int = 2,
    ) -> None:
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_spatial_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        num_patches = num_spatial_patches * (num_frames // tubelet_size)

        self.img_size = img_size
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
            stride=(self.tubelet_size, patch_size[0], patch_size[1]),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, C, T, H, W = x.shape
        if H != self.img_size[0] or W != self.img_size[1]:
            raise AssertionError(
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            )
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


def get_sinusoid_encoding_table(n_position: int, d_hid: int) -> Tensor:
    """Sin-cos position encoding table."""

    def get_position_angle_vec(position: int) -> Iterable[float]:
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    table = torch.tensor(sinusoid_table, dtype=torch.float32).unsqueeze(0)
    return table


class VisionTransformer(nn.Module):
    """Vision Transformer for video inputs with tubelet embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        head_drop_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        init_values: float = 0.0,
        use_learnable_pos_emb: bool = False,
        init_scale: float = 0.0,
        all_frames: int = 16,
        tubelet_size: int = 2,
        use_mean_pooling: bool = True,
        with_cp: bool = False,
        cos_attn: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_frames=all_frames,
            tubelet_size=tubelet_size,
        )
        num_patches = self.patch_embed.num_patches
        self.with_cp = with_cp

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            self.register_buffer(
                "pos_embed",
                get_sinusoid_encoding_table(num_patches, embed_dim),
                persistent=False,
            )

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    cos_attn=cos_attn,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head_dropout = nn.Dropout(head_drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if isinstance(self.pos_embed, nn.Parameter):
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)
        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self) -> int:
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self) -> set:
        return {"pos_embed", "cls_token"}

    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: str = "") -> None:
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x: Tensor) -> Tensor:
        B = x.size(0)
        x = self.patch_embed(x)

        if self.pos_embed is not None:
            pos_embed = self.pos_embed.expand(B, -1, -1).to(x.dtype).to(x.device)
            x = x + pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            if self.with_cp:
                x = cp.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.fc_norm is not None:
            return self.fc_norm(x.mean(1))
        else:
            return self.norm(x[:, 0])

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.head_dropout(x)
        x = self.head(x)
        return x


def vit_small_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


def vit_base_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


def vit_large_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


def vit_huge_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


def vit_giant_patch14_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model = VisionTransformer(
        patch_size=14,
        embed_dim=1408,
        depth=40,
        num_heads=16,
        mlp_ratio=48 / 11,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


MODEL_FACTORY: Dict[str, callable] = {
    "vit_small_patch16_224": vit_small_patch16_224,
    "vit_base_patch16_224": vit_base_patch16_224,
    "vit_large_patch16_224": vit_large_patch16_224,
    "vit_huge_patch16_224": vit_huge_patch16_224,
    "vit_giant_patch14_224": vit_giant_patch14_224,
}


def build_videomaev2_backbone(
    arch: str,
    img_size: int,
    num_frames: int,
    tubelet_size: int = 2,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    head_drop_rate: float = 0.0,
    use_mean_pooling: bool = True,
    init_scale: float = 0.0,
    with_checkpoint: bool = False,
    cos_attn: bool = False,
    **extra_kwargs,
) -> VisionTransformer:
    if arch not in MODEL_FACTORY:
        raise ValueError(f"Unsupported VideoMAE architecture '{arch}'. "
                         f"Available: {list(MODEL_FACTORY.keys())}")

    builder = MODEL_FACTORY[arch]
    model = builder(
        img_size=img_size,
        num_classes=extra_kwargs.pop("num_classes", 0),
        all_frames=num_frames,
        tubelet_size=tubelet_size,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        head_drop_rate=head_drop_rate,
        use_mean_pooling=use_mean_pooling,
        init_scale=init_scale,
        with_cp=with_checkpoint,
        cos_attn=cos_attn,
        **extra_kwargs,
    )
    return model


def _resize_pos_embed(
    pos_embed_checkpoint: Tensor,
    model: VisionTransformer,
    target_num_frames: int,
) -> Tensor:
    """Resize positional embeddings to match current model configuration."""
    if not isinstance(pos_embed_checkpoint, torch.Tensor):
        pos_embed_checkpoint = torch.tensor(pos_embed_checkpoint)

    pos_embed_checkpoint = pos_embed_checkpoint.float()
    target_pos_embed = model.pos_embed

    if pos_embed_checkpoint.shape == target_pos_embed.shape:
        return pos_embed_checkpoint

    tubelet_size = getattr(model.patch_embed, "tubelet_size", 1)
    target_tokens = target_pos_embed.shape[-2]
    target_T = max(target_num_frames // tubelet_size, 1)

    num_extra_tokens = max(target_pos_embed.shape[-2] - model.patch_embed.num_patches, 0)
    embedding_size = pos_embed_checkpoint.shape[-1]

    orig_tokens = pos_embed_checkpoint.shape[1] - num_extra_tokens
    if orig_tokens <= 0 or target_tokens <= 0:
        return pos_embed_checkpoint

    orig_size = int(round((orig_tokens / target_T) ** 0.5))
    new_size = int(round((target_tokens / target_T) ** 0.5))
    if orig_size <= 0 or new_size <= 0:
        return pos_embed_checkpoint

    if num_extra_tokens > 0:
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    else:
        extra_tokens = pos_embed_checkpoint[:, :0]
        pos_tokens = pos_embed_checkpoint

    pos_tokens = pos_tokens.reshape(-1, target_T, orig_size, orig_size, embedding_size)
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = F.interpolate(pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, target_T, new_size, new_size, embedding_size)
    pos_tokens = pos_tokens.flatten(1, 3)

    if num_extra_tokens > 0:
        pos_tokens = torch.cat((extra_tokens, pos_tokens), dim=1)
    return pos_tokens


def load_videomaev2_checkpoint(
    model: VisionTransformer,
    ckpt_path: str,
    *,
    num_frames: int,
    model_key: str = "model|module|state_dict",
    logger: Optional[object] = None,
) -> Tuple[Iterable[str], Iterable[str]]:
    """Load a pretrained checkpoint with best-effort key alignment."""
    if not ckpt_path:
        return [], []
    if not os.path.isfile(ckpt_path):
        msg = f"[VideoMAE] Checkpoint not found: {ckpt_path}"
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return [], []

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    checkpoint_model = None
    for key in (model_key or "").split("|"):
        key = key.strip()
        if key and key in checkpoint:
            checkpoint_model = checkpoint[key]
            if logger:
                logger.info(f"[VideoMAE] Load state_dict by model_key = {key}")
            break
    if checkpoint_model is None and "state_dict" in checkpoint:
        checkpoint_model = checkpoint["state_dict"]
    if checkpoint_model is None:
        checkpoint_model = checkpoint

    new_dict = {}
    for k, v in checkpoint_model.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod.") :]
        if k.startswith("backbone."):
            new_dict[k[len("backbone.") :]] = v
        elif k.startswith("encoder."):
            new_dict[k[len("encoder.") :]] = v
        else:
            new_dict[k] = v
    checkpoint_model = new_dict

    state_dict = model.state_dict()
    for head_key in ["head.weight", "head.bias"]:
        if head_key in checkpoint_model and head_key in state_dict:
            if checkpoint_model[head_key].shape != state_dict[head_key].shape:
                checkpoint_model.pop(head_key)

    if "pos_embed" in checkpoint_model:
        checkpoint_model["pos_embed"] = _resize_pos_embed(checkpoint_model["pos_embed"], model, num_frames)

    missing, unexpected = model.load_state_dict(checkpoint_model, strict=False)
    return missing, unexpected


__all__ = [
    "VisionTransformer",
    "build_videomaev2_backbone",
    "load_videomaev2_checkpoint",
    "MODEL_FACTORY",
]

