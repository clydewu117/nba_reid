#!/usr/bin/env python
"""
Nearest Neighbor Visualization for NBA Video ReID

For each of 120 players, selects one query video and finds the top-5 nearest
neighbors using both an appearance model and a mask model.

Outputs per player (under <output-dir>/<player>/):
    results.txt   - query info + top-5 from each model
    app/          - appearance model neighbors (16 frames each + thumbnails)
    mask/         - mask model neighbors (app + mask frames each + thumbnails)
"""

import os
import random
import argparse
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import av

from config.defaults import get_cfg_defaults
from models.build import build_model


# ---------------------------------------------------------------------------
# Feature extraction helpers (from classification.py)
# ---------------------------------------------------------------------------

def _extract_backbone_features(model, videos):
    backbone = model.backbone
    if hasattr(backbone, "forward_features"):
        return backbone.forward_features(videos)
    return backbone(videos)


def _forward_reid_head_for_feat(model, features):
    head = model.reid_head
    if hasattr(head, "feat_proj") and head.feat_proj is not None:
        features = head.feat_proj(features)
    bn_feat = head.bottleneck(features)
    neck_feat = getattr(head, "neck_feat", "after")
    if neck_feat == "after":
        return F.normalize(bn_feat, p=2, dim=1)
    return F.normalize(features, p=2, dim=1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def decode_all_frames(video_path):
    """Decode every frame of a video file via PyAV. Returns list[np.ndarray]."""
    container = av.open(video_path)
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    return frames


def uniform_sample_indices(total_frames, num_frames):
    """Uniform segment sampling (test-mode): pick the first frame of each segment."""
    indices = np.arange(total_frames)
    num_pads = num_frames - (total_frames % num_frames)
    if num_pads != num_frames:
        indices = np.concatenate(
            [indices, np.full(num_pads, total_frames - 1, dtype=int)]
        )
    pools = np.array_split(indices, num_frames)
    return [int(seg[0]) for seg in pools]


def load_video_tensor(video_path, num_frames, transform):
    """Load & preprocess a single video → tensor [1, C, T, H, W]."""
    raw = decode_all_frames(video_path)
    if len(raw) == 0:
        raise RuntimeError(f"Video has 0 frames: {video_path}")
    idxs = uniform_sample_indices(len(raw), num_frames)
    frames = [transform(Image.fromarray(raw[i])) for i in idxs]
    video = torch.stack(frames, dim=0).permute(1, 0, 2, 3)   # [C,T,H,W]
    return video.unsqueeze(0)                                  # [1,C,T,H,W]


def extract_and_save_frames(video_path, save_dir, num_frames=16):
    """Save *num_frames* uniformly-sampled frames as JPGs. Returns first-frame path."""
    os.makedirs(save_dir, exist_ok=True)
    raw = decode_all_frames(video_path)
    if len(raw) == 0:
        raise RuntimeError(f"Video has 0 frames: {video_path}")
    idxs = uniform_sample_indices(len(raw), num_frames)
    first_path = None
    for i, idx in enumerate(idxs):
        img = Image.fromarray(raw[idx])
        p = os.path.join(save_dir, f"frame_{i:02d}.jpg")
        img.save(p)
        if i == 0:
            first_path = p
    return first_path


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------

def build_video_list(data_root, shot_type):
    """
    Scan data_root/{appearance,mask}/<player>/<shot_type>/ and return a
    unified list where each entry carries both the appearance and mask path.

    Returns
    -------
    videos : list[dict]  – keys: player, video_name, app_path, mask_path
    players : list[str]  – sorted player folder names
    """
    app_base = os.path.join(data_root, "appearance")
    mask_base = os.path.join(data_root, "mask")

    players = sorted(
        d for d in os.listdir(app_base)
        if os.path.isdir(os.path.join(app_base, d))
    )

    videos = []
    for player in players:
        app_shot = os.path.join(app_base, player, shot_type)
        mask_shot = os.path.join(mask_base, player, shot_type)
        if not os.path.isdir(app_shot):
            continue

        for vf in sorted(os.listdir(app_shot)):
            if not vf.lower().endswith((".mp4", ".avi", ".mov")):
                continue
            mp = os.path.join(mask_shot, vf)
            if not os.path.isfile(mp):
                continue
            videos.append({
                "player": player,
                "video_name": vf,
                "app_path": os.path.join(app_shot, vf),
                "mask_path": mp,
            })

    return videos, players


# ---------------------------------------------------------------------------
# Batch feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_all_features(model, video_paths, num_frames, transform, device,
                         batch_size=16):
    """
    Extract normalised ReID features for every video in *video_paths*.
    Processes videos one-by-one (variable-length decoding), but could be
    batched if all videos share the same frame count after sampling.
    """
    model.eval()
    all_feats = []
    for vp in tqdm(video_paths, desc="  features"):
        tensor = load_video_tensor(vp, num_frames, transform).to(device)
        backbone_feat = _extract_backbone_features(model, tensor)
        feat = _forward_reid_head_for_feat(model, backbone_feat)
        all_feats.append(feat.cpu())
    return torch.cat(all_feats, dim=0)                         # [N, D]


# ---------------------------------------------------------------------------
# Naming helper
# ---------------------------------------------------------------------------

def neighbor_name(rank, player, video_name):
    vid_id = os.path.splitext(video_name)[0]
    safe = player.replace(" ", "_")
    return f"rank{rank}_{safe}_{vid_id}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Nearest-Neighbor Visualization for NBA Video ReID"
    )
    parser.add_argument("--app-config",     type=str, required=True)
    parser.add_argument("--app-checkpoint", type=str, required=True)
    parser.add_argument("--mask-config",    type=str, required=True)
    parser.add_argument("--mask-checkpoint",type=str, required=True)
    parser.add_argument("--output-dir",     type=str,
                        default="/fs/scratch/PAS3184/v3_vis/nearest_neighbor")
    parser.add_argument("--shot-type",      type=str, default="freethrow")
    parser.add_argument("--query-index",    type=int, default=0,
                        help="index of the video to pick per player (0=first)")
    parser.add_argument("--top-k",          type=int, default=5)
    parser.add_argument("--num-frames",     type=int, default=None,
                        help="override NUM_FRAMES from config")
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ── Load appearance config ───────────────────────────────────────────────
    app_cfg = get_cfg_defaults()
    app_cfg.merge_from_file(args.app_config)
    data_root = app_cfg.DATA.ROOT

    # ── Build unified video list ─────────────────────────────────────────────
    videos, players = build_video_list(data_root, args.shot_type)
    num_classes = len(players)
    print(f"Found {len(videos)} videos across {num_classes} players "
          f"(shot_type={args.shot_type})")

    player_to_indices = {}
    for i, v in enumerate(videos):
        player_to_indices.setdefault(v["player"], []).append(i)

    num_frames = args.num_frames or app_cfg.DATA.NUM_FRAMES

    # ── Appearance model ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Loading appearance model …")
    app_cfg.MODEL.NUM_CLASSES = num_classes
    if not torch.cuda.is_available():
        app_cfg.NUM_GPUS = 0
    app_cfg.freeze()

    app_model = build_model(app_cfg)
    ckpt = torch.load(args.app_checkpoint, map_location=device, weights_only=False)
    app_model.load_state_dict(ckpt["model_state_dict"])
    app_model = app_model.to(device).eval()
    print(f"  Loaded epoch {ckpt.get('epoch', '?')}")

    app_transform = T.Compose([
        T.Resize((app_cfg.DATA.HEIGHT, app_cfg.DATA.WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"Extracting appearance features ({len(videos)} videos) …")
    app_feats = extract_all_features(
        app_model, [v["app_path"] for v in videos],
        num_frames, app_transform, device,
    )
    del app_model
    torch.cuda.empty_cache()

    # ── Mask model ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Loading mask model …")
    mask_cfg = get_cfg_defaults()
    mask_cfg.merge_from_file(args.mask_config)
    mask_cfg.MODEL.NUM_CLASSES = num_classes
    if not torch.cuda.is_available():
        mask_cfg.NUM_GPUS = 0
    mask_cfg.freeze()

    mask_model = build_model(mask_cfg)
    ckpt = torch.load(args.mask_checkpoint, map_location=device, weights_only=False)
    mask_model.load_state_dict(ckpt["model_state_dict"])
    mask_model = mask_model.to(device).eval()
    print(f"  Loaded epoch {ckpt.get('epoch', '?')}")

    mask_transform = T.Compose([
        T.Resize((mask_cfg.DATA.HEIGHT, mask_cfg.DATA.WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"Extracting mask features ({len(videos)} videos) …")
    mask_feats = extract_all_features(
        mask_model, [v["mask_path"] for v in videos],
        num_frames, mask_transform, device,
    )
    del mask_model
    torch.cuda.empty_cache()

    # ── Precompute numpy arrays ──────────────────────────────────────────────
    app_feats_np = app_feats.numpy()
    mask_feats_np = mask_feats.numpy()

    # ── Process each player ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Saving nearest-neighbor visualizations for {len(players)} players …")
    print(f"{'=' * 60}\n")

    for player in tqdm(players, desc="Players"):
        indices = player_to_indices.get(player, [])
        if not indices:
            continue

        qi = min(args.query_index, len(indices) - 1)
        query_idx = indices[qi]
        qv = videos[query_idx]

        player_dir = os.path.join(args.output_dir, player)
        os.makedirs(player_dir, exist_ok=True)

        # ── Appearance top-k ─────────────────────────────────────────────
        dists_app = np.linalg.norm(app_feats_np - app_feats_np[query_idx], axis=1)
        dists_app[query_idx] = np.inf
        app_nn = np.argsort(dists_app)[: args.top_k]

        # ── Mask top-k ───────────────────────────────────────────────────
        dists_mask = np.linalg.norm(mask_feats_np - mask_feats_np[query_idx], axis=1)
        dists_mask[query_idx] = np.inf
        mask_nn = np.argsort(dists_mask)[: args.top_k]

        # ── 1) results.txt ───────────────────────────────────────────────
        txt_path = os.path.join(player_dir, "results.txt")
        with open(txt_path, "w") as f:
            f.write(f"Query Player : {qv['player']}\n")
            f.write(f"Query Video  : {qv['video_name']}\n")
            f.write(f"App Path     : {qv['app_path']}\n")
            f.write(f"Mask Path    : {qv['mask_path']}\n")

            f.write(f"\n{'=' * 60}\n")
            f.write("Appearance Model – Top-5 Nearest Neighbors\n")
            f.write(f"{'=' * 60}\n")
            for rank, idx in enumerate(app_nn, 1):
                nb = videos[idx]
                f.write(f"  Rank {rank}: {nb['player']} / {nb['video_name']}  "
                        f"(dist={dists_app[idx]:.4f})\n")

            f.write(f"\n{'=' * 60}\n")
            f.write("Mask Model – Top-5 Nearest Neighbors\n")
            f.write(f"{'=' * 60}\n")
            for rank, idx in enumerate(mask_nn, 1):
                nb = videos[idx]
                f.write(f"  Rank {rank}: {nb['player']} / {nb['video_name']}  "
                        f"(dist={dists_mask[idx]:.4f})\n")

        # ── 2) app/ folder ───────────────────────────────────────────────
        app_dir = os.path.join(player_dir, "app")
        os.makedirs(app_dir, exist_ok=True)

        for rank, idx in enumerate(app_nn, 1):
            nb = videos[idx]
            name = neighbor_name(rank, nb["player"], nb["video_name"])

            frame_dir = os.path.join(app_dir, name)
            first_frame = extract_and_save_frames(
                nb["app_path"], frame_dir, num_frames
            )
            if first_frame:
                shutil.copy2(first_frame, os.path.join(app_dir, f"{name}.jpg"))

        # ── 3) mask/ folder ──────────────────────────────────────────────
        mask_out_dir = os.path.join(player_dir, "mask")
        os.makedirs(mask_out_dir, exist_ok=True)

        for rank, idx in enumerate(mask_nn, 1):
            nb = videos[idx]
            name = neighbor_name(rank, nb["player"], nb["video_name"])

            nb_dir = os.path.join(mask_out_dir, name)

            # mask sub-folder: frames from the mask video (the actual neighbor)
            extract_and_save_frames(
                nb["mask_path"], os.path.join(nb_dir, "mask"), num_frames
            )

            # app sub-folder: frames from the corresponding appearance video
            app_first = extract_and_save_frames(
                nb["app_path"], os.path.join(nb_dir, "app"), num_frames
            )

            # thumbnail in mask/ root: first frame from the appearance video
            if app_first:
                shutil.copy2(
                    app_first, os.path.join(mask_out_dir, f"{name}.jpg")
                )

    print(f"\nDone!  Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
