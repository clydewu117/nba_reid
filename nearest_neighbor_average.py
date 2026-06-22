#!/usr/bin/env python
"""
Nearest Neighbor Visualization for NBA Video ReID

For each of the first N players (default 20), splits videos into query /
gallery sets, averages each player's query-set features into a single
representative vector, and finds the top-5 nearest gallery videos using
both an appearance model and a mask model.

Outputs per player (under <output-dir>/<player>/):
    results.txt   - query video list + top-5 from each model
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

def build_video_list(data_root, shot_type, max_players=None):
    """
    Scan data_root/{appearance,mask}/<player>/<shot_type>/ and return a
    unified list where each entry carries both the appearance and mask path.

    Parameters
    ----------
    max_players : int or None
        If set, only include the first *max_players* players (alphabetical).

    Returns
    -------
    videos : list[dict]  – keys: player, video_name, app_path, mask_path
    players : list[str]  – sorted player folder names (after truncation)
    """
    app_base = os.path.join(data_root, "appearance")
    mask_base = os.path.join(data_root, "mask")

    players = sorted(
        d for d in os.listdir(app_base)
        if os.path.isdir(os.path.join(app_base, d))
    )
    if max_players is not None:
        players = players[:max_players]

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
# Query / Gallery split  (mirrors classification.py logic)
# ---------------------------------------------------------------------------

def split_query_gallery(player_to_indices, query_ratio=0.5, seed=42):
    """
    Per-player random split into query and gallery index sets.

    Returns
    -------
    query_indices  : list[int]  – indices into the global video list
    gallery_indices: list[int]
    """
    rng = np.random.RandomState(seed)
    query_indices, gallery_indices = [], []

    for _player, indices in sorted(player_to_indices.items()):
        shuffled = np.array(indices)
        rng.shuffle(shuffled)

        if len(shuffled) < 2:
            gallery_indices.extend(shuffled.tolist())
            continue

        split = max(1, int(len(shuffled) * query_ratio))
        split = min(split, len(shuffled) - 1)

        query_indices.extend(shuffled[:split].tolist())
        gallery_indices.extend(shuffled[split:].tolist())

    return query_indices, gallery_indices


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
    parser.add_argument("--num-players",    type=int, default=20,
                        help="only process the first N players (alphabetical)")
    parser.add_argument("--query-ratio",    type=float, default=0.5,
                        help="fraction of each player's videos used as query")
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
    # Scan ALL players first so num_classes matches the checkpoint (e.g. 120),
    # then keep only the first N for feature extraction and visualisation.
    all_videos, all_players = build_video_list(data_root, args.shot_type)
    num_classes = len(all_players)

    videos, players = build_video_list(
        data_root, args.shot_type, max_players=args.num_players,
    )
    print(f"Found {len(all_videos)} videos across {num_classes} players total; "
          f"using first {len(players)} players ({len(videos)} videos)")

    player_to_indices = {}
    for i, v in enumerate(videos):
        player_to_indices.setdefault(v["player"], []).append(i)

    # ── Query / Gallery split ────────────────────────────────────────────────
    query_indices, gallery_indices = split_query_gallery(
        player_to_indices, query_ratio=args.query_ratio, seed=args.seed,
    )
    gallery_set = set(gallery_indices)

    # Per-player query / gallery index lookup
    player_query_map = {}     # player → list of global indices (query)
    player_gallery_map = {}   # player → list of global indices (gallery)
    for idx in query_indices:
        p = videos[idx]["player"]
        player_query_map.setdefault(p, []).append(idx)
    for idx in gallery_indices:
        p = videos[idx]["player"]
        player_gallery_map.setdefault(p, []).append(idx)

    print(f"Query/Gallery split (ratio={args.query_ratio}):")
    print(f"  Query videos   : {len(query_indices)}")
    print(f"  Gallery videos : {len(gallery_indices)}")

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

    # Gallery feature matrices (rows aligned with gallery_indices)
    gallery_arr = np.array(gallery_indices)
    gallery_app_feats = app_feats_np[gallery_arr]     # [G, D]
    gallery_mask_feats = mask_feats_np[gallery_arr]   # [G, D]

    # ── Process each player ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Saving nearest-neighbor visualizations for {len(players)} players …")
    print(f"{'=' * 60}\n")

    for player in tqdm(players, desc="Players"):
        q_idxs = player_query_map.get(player, [])
        g_idxs = player_gallery_map.get(player, [])
        if not q_idxs:
            print(f"  {player}: skipped (no query videos)")
            continue

        player_dir = os.path.join(args.output_dir, player)
        os.makedirs(player_dir, exist_ok=True)

        # ── Average query features & re-normalise ────────────────────────
        app_q = app_feats_np[q_idxs].mean(axis=0)
        app_q /= np.linalg.norm(app_q) + 1e-12

        mask_q = mask_feats_np[q_idxs].mean(axis=0)
        mask_q /= np.linalg.norm(mask_q) + 1e-12

        # ── Appearance top-k in gallery ──────────────────────────────────
        dists_app = np.linalg.norm(gallery_app_feats - app_q, axis=1)
        app_topk_gidx = np.argsort(dists_app)[: args.top_k]

        # ── Mask top-k in gallery ────────────────────────────────────────
        dists_mask = np.linalg.norm(gallery_mask_feats - mask_q, axis=1)
        mask_topk_gidx = np.argsort(dists_mask)[: args.top_k]

        # ── 1) results.txt ───────────────────────────────────────────────
        txt_path = os.path.join(player_dir, "results.txt")
        with open(txt_path, "w") as f:
            f.write(f"Player       : {player}\n")
            f.write(f"Total videos : {len(q_idxs) + len(g_idxs)}\n")
            f.write(f"Query videos : {len(q_idxs)}\n")
            f.write(f"Gallery videos: {len(g_idxs)}\n")
            f.write(f"Query ratio  : {args.query_ratio}\n")
            f.write(f"Seed         : {args.seed}\n")

            f.write(f"\n{'=' * 60}\n")
            f.write("Query Video List\n")
            f.write(f"{'=' * 60}\n")
            for qi in sorted(q_idxs):
                v = videos[qi]
                f.write(f"  {v['video_name']}  "
                        f"(app: {v['app_path']})\n")

            f.write(f"\n{'=' * 60}\n")
            f.write("Appearance Model – Top-5 Nearest Neighbors (gallery)\n")
            f.write(f"{'=' * 60}\n")
            for rank, gi in enumerate(app_topk_gidx, 1):
                orig = gallery_arr[gi]
                nb = videos[orig]
                same = "SELF" if nb["player"] == player else ""
                f.write(f"  Rank {rank}: {nb['player']} / {nb['video_name']}  "
                        f"(dist={dists_app[gi]:.4f}) {same}\n")

            f.write(f"\n{'=' * 60}\n")
            f.write("Mask Model – Top-5 Nearest Neighbors (gallery)\n")
            f.write(f"{'=' * 60}\n")
            for rank, gi in enumerate(mask_topk_gidx, 1):
                orig = gallery_arr[gi]
                nb = videos[orig]
                same = "SELF" if nb["player"] == player else ""
                f.write(f"  Rank {rank}: {nb['player']} / {nb['video_name']}  "
                        f"(dist={dists_mask[gi]:.4f}) {same}\n")

        # ── 2) app/ folder ───────────────────────────────────────────────
        app_dir = os.path.join(player_dir, "app")
        os.makedirs(app_dir, exist_ok=True)

        for rank, gi in enumerate(app_topk_gidx, 1):
            orig = gallery_arr[gi]
            nb = videos[orig]
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

        for rank, gi in enumerate(mask_topk_gidx, 1):
            orig = gallery_arr[gi]
            nb = videos[orig]
            name = neighbor_name(rank, nb["player"], nb["video_name"])

            nb_dir = os.path.join(mask_out_dir, name)

            extract_and_save_frames(
                nb["mask_path"], os.path.join(nb_dir, "mask"), num_frames
            )

            app_first = extract_and_save_frames(
                nb["app_path"], os.path.join(nb_dir, "app"), num_frames
            )

            if app_first:
                shutil.copy2(
                    app_first, os.path.join(mask_out_dir, f"{name}.jpg")
                )

    print(f"\nDone!  Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
