#!/usr/bin/env python
"""
A lightweight, self-contained TimeSformer-style video ViT backbone.
- Supports attention_type in {'divided_space_time', 'space_only', 'joint_space_time'}
- Input: [B, C, T, H, W]
- Output: CLS feature [B, embed_dim]

Notes:
- This is an original minimal implementation inspired by the idea of TimeSformer but not copied
  from the reference implementation. It focuses on compatibility with this repo's ReID head.
- Positional embedding is 2D for spatial tokens and a separate learnable temporal embedding.
- Patch embedding is Conv2d with kernel=stride=patch_size applied per-frame.

Dependencies: PyTorch. (einops is optional; we avoid it here.)
"""
from typing import Tuple
import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F


def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)


class DropPath(nn.Module):
    """Stochastic depth per sample.
    Reference idea: https://arxiv.org/abs/1603.09382
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        output = x.div(keep_prob) * random_tensor
        return output


class LayerNorm1e6(nn.LayerNorm):
    """LayerNorm with eps=1e-6 to match official TimeSformer defaults."""

    def __init__(self, normalized_shape):
        super().__init__(normalized_shape, eps=1e-6)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=LayerNorm1e6,
        attention_type="divided_space_time",
    ):
        super().__init__()
        assert attention_type in {
            "divided_space_time",
            "space_only",
            "joint_space_time",
        }
        self.attention_type = attention_type

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        if self.attention_type == "divided_space_time":
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
            self.temporal_fc = nn.Linear(dim, dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, B, T, H, W):
        # x: [B, 1 + T*H*W, C]
        if self.attention_type in {"space_only", "joint_space_time"}:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

        # divided_space_time
        # Temporal attention (exclude CLS), compute per spatial position across time
        cls_token = x[:, :1, :]
        xt = x[:, 1:, :]  # [B, T*H*W, C]
        xt = xt.view(B, T, H * W, -1).transpose(1, 2).contiguous()  # [B, H*W, T, C]
        xt = xt.view(B * H * W, T, -1)  # [(B*H*W), T, C]
        xt = xt + self.drop_path(
            self.temporal_attn(self.temporal_norm1(xt))
        )  # temporal attn
        xt = self.temporal_fc(xt)
        xt = (
            xt.view(B, H * W, T, -1).transpose(1, 2).contiguous().view(B, T * H * W, -1)
        )

        # Spatial branch (official behavior): build residual via attention output, average per-frame CLS
        init_cls_token = cls_token  # [B,1,C]
        cls_per_t = init_cls_token.expand(B, T, -1).contiguous().view(B * T, 1, -1)
        xs = xt.view(B, T, H * W, -1).contiguous().view(B * T, H * W, -1)
        xs_with_cls = torch.cat([cls_per_t, xs], dim=1)  # [B*T, 1+H*W, C]

        res_spatial = self.drop_path(self.attn(self.norm1(xs_with_cls)))

        # Take care of CLS token: average per-frame
        cls_tok = (
            res_spatial[:, 0, :].view(B, T, -1).mean(dim=1, keepdim=True)
        )  # [B,1,C]
        res = (
            res_spatial[:, 1:, :]
            .view(B, T, H * W, -1)
            .contiguous()
            .view(B, T * H * W, -1)
        )

        # Residual sum over concatenated tokens
        x = torch.cat([init_cls_token, xt], dim=1) + torch.cat([cls_tok, res], dim=1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to patch embedding applied per-frame.
    Input: [B, C, T, H, W] -> Output: [B*T, N_patches, C_emb]
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.num_patches = num_patches
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)  # [B*T,C,H,W]
        x = self.proj(x)  # [B*T, E, H/P, W/P]
        Hp, Wp = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)  # [B*T, N, E]
        return x, T, Hp, Wp


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=LayerNorm1e6,
        num_frames=8,
        attention_type="divided_space_time",
        dropout=0.0,
    ):
        super().__init__()
        self.attention_type = attention_type
        self.depth = depth
        self.embed_dim = embed_dim
        self.num_frames = num_frames

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.attention_type != "space_only":
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
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
                    attention_type=attention_type,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        # Initialize temporal_fc to zero for blocks i > 0 (match official behavior)
        if self.attention_type == "divided_space_time":
            for i, m in enumerate(self.blocks):
                if hasattr(m, "temporal_fc") and i > 0:
                    nn.init.constant_(m.temporal_fc.weight, 0)
                    nn.init.constant_(m.temporal_fc.bias, 0)

    def no_weight_decay(self):
        return {"pos_embed", "cls_token"} | (
            {"time_embed"} if self.attention_type != "space_only" else set()
        )

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def _resize_pos_embed(self, x, Hp, Wp):
        # x: [B*T, N+1, C] after adding cls
        pos_embed = self.pos_embed  # [1, 1+N, C]
        if x.size(1) == pos_embed.size(1):
            return x + pos_embed
        # interpolate spatial part
        cls_pos = pos_embed[:, :1]
        spatial_pos = pos_embed[:, 1:]  # [1, N, C]
        P = int(math.sqrt(spatial_pos.size(1)))
        spatial_pos = spatial_pos.reshape(1, P, P, self.embed_dim).permute(
            0, 3, 1, 2
        )  # [1,C,P,P]
        new_pos = F.interpolate(spatial_pos, size=(Hp, Wp), mode="nearest")
        new_pos = new_pos.permute(0, 2, 3, 1).reshape(1, Hp * Wp, self.embed_dim)
        new_pos = torch.cat([cls_pos, new_pos], dim=1)
        return x + new_pos

    def forward_features(self, x):
        B = x.size(0)
        x, T, Hp, Wp = self.patch_embed(x)  # [B*T, N, C]
        # add cls token per sequence (B*T)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # [B*T,1,C]
        x = torch.cat([cls_tokens, x], dim=1)  # [B*T, 1+N, C]
        x = self._resize_pos_embed(x, Hp, Wp)
        x = self.pos_drop(x)

        if self.attention_type != "space_only":
            # Follow official: separate cls, add temporal embedding to spatial tokens only
            x_bt = x.view(B, T, -1, self.embed_dim)  # [B, T, 1+N, C]
            cls_tokens = x_bt[:, 0:1, 0:1, :].squeeze(
                2
            )  # [B,1,C] take cls from first frame
            spatial = x_bt[:, :, 1:, :]  # [B, T, N, C]
            # reshape to per-spatial sequences: [B*N, T, C]
            BN, N = B * (Hp * Wp), (Hp * Wp)
            spatial_seq = (
                spatial.permute(0, 2, 1, 3).contiguous().view(BN, T, self.embed_dim)
            )
            # interpolate temporal embedding if needed (nearest)
            if T != self.time_embed.size(1):
                te = self.time_embed.transpose(1, 2)  # [1,C,T0]
                te = F.interpolate(te, size=T, mode="nearest")
                te = te.transpose(1, 2)  # [1,T,C]
            else:
                te = self.time_embed
            spatial_seq = spatial_seq + te  # broadcast [1,T,C]
            spatial_seq = self.time_drop(spatial_seq)
            # reshape back to [B, T*N, C]
            spatial_seq = (
                spatial_seq.view(B, N, T, self.embed_dim)
                .permute(0, 2, 1, 3)
                .contiguous()
                .view(B, T * N, self.embed_dim)
            )
            # concat single cls back
            x = torch.cat([cls_tokens, spatial_seq], dim=1)  # [B, 1 + T*N, C]

        # pass blocks with knowledge of grid
        for blk in self.blocks:
            x = blk(x, B, T, Hp, Wp)

        if self.attention_type == "space_only":
            # reshape back to [B, T, N+1, C] and average over frames
            x = x.view(B, T, -1, self.embed_dim).mean(dim=1)

        x = self.norm(x)
        return x[:, 0]  # CLS

    def forward(self, x):
        feats = self.forward_features(x)
        logits = self.head(feats)
        return logits


def _conv_filter_linear_to_conv(state_dict, patch_size=16, in_chans=3):
    """Convert patch embedding weight from flattened linear to conv weight if needed.
    Accepts keys like 'patch_embed.proj.weight' with shape [embed_dim, in_chans*P*P] and reshapes to
    [embed_dim, in_chans, P, P]. If already 4D, returns as-is.
    """
    out = {}
    for k, v in state_dict.items():
        if k.endswith("patch_embed.proj.weight") and v.ndim == 2:
            P2C = v.shape[1]
            p = int(round((P2C / in_chans) ** 0.5))
            v = v.view(v.shape[0], in_chans, p, p)
            out[k] = v
        else:
            out[k] = v
    return out


def load_pretrained_timesformer(model: VisionTransformer, ckpt_path: str):
    """Load pretrained weights into local VisionTransformer with best-effort compatibility.
    - Converts patch_embed weights if needed
    - Interpolates pos_embed/time_embed to current shapes when sizes mismatch
    - Loads with strict=False and reports missing/unexpected keys
    """
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    state = _conv_filter_linear_to_conv(state)

    # Handle pos_embed interpolation if present and mismatched
    if "pos_embed" in state and state["pos_embed"].shape != model.pos_embed.shape:
        with torch.no_grad():
            pos_embed = state["pos_embed"]  # [1, N1+1, C]
            C = pos_embed.shape[-1]
            cls_pos = pos_embed[:, :1]
            spatial = pos_embed[:, 1:]
            P = int(math.sqrt(spatial.shape[1]))
            spatial = spatial.reshape(1, P, P, C).permute(0, 3, 1, 2)
            Hp = model.patch_embed.num_patches
            # derive target Hp, Wp from model.pos_embed
            tgt_tokens = model.pos_embed.shape[1] - 1
            tgt_P = int(math.sqrt(tgt_tokens))
            new_spatial = F.interpolate(spatial, size=(tgt_P, tgt_P), mode="nearest")
            new_spatial = new_spatial.permute(0, 2, 3, 1).reshape(1, tgt_P * tgt_P, C)
            state["pos_embed"] = torch.cat([cls_pos, new_spatial], dim=1)

    # Handle time_embed interpolation if present and mismatched
    if (
        "time_embed" in state
        and state["time_embed"].shape
        != getattr(model, "time_embed", torch.empty(1, 0, 1)).shape
    ):
        with torch.no_grad():
            te = state["time_embed"]  # [1, T0, C]
            te = te.transpose(1, 2)  # [1, C, T0]
            T = model.num_frames
            te = F.interpolate(te, size=T, mode="nearest")
            state["time_embed"] = te.transpose(1, 2)

    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected
