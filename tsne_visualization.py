#!/usr/bin/env python
"""
t-SNE Visualization for NBA Video ReID

Extracts features for the first N players using both an appearance model and
a mask model, runs t-SNE on each feature set, and produces scatter plots
coloured by player identity.

On the Appearance plot, pairs of clusters that are close in appearance space
but far apart in mask space are connected with gradient-coloured lines to
highlight where the two models disagree.

Optional: --color-by-jersey adds jersey-colour plots.
"""

import os
import csv
import random
import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import av

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

_FONT_PATH = os.path.join(os.path.dirname(__file__), "DMSans-Regular.ttf")
if os.path.isfile(_FONT_PATH):
    fm.fontManager.addfont(_FONT_PATH)
    _font_name = fm.FontProperties(fname=_FONT_PATH).get_name()
    matplotlib.rcParams["font.family"] = _font_name

from config.defaults import get_cfg_defaults
from models.build import build_model


# ---------------------------------------------------------------------------
# Feature extraction helpers (shared with nearest_neighbor.py)
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
    container = av.open(video_path)
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    return frames


def uniform_sample_indices(total_frames, num_frames):
    indices = np.arange(total_frames)
    num_pads = num_frames - (total_frames % num_frames)
    if num_pads != num_frames:
        indices = np.concatenate(
            [indices, np.full(num_pads, total_frames - 1, dtype=int)]
        )
    pools = np.array_split(indices, num_frames)
    return [int(seg[0]) for seg in pools]


def load_video_tensor(video_path, num_frames, transform):
    raw = decode_all_frames(video_path)
    if len(raw) == 0:
        raise RuntimeError(f"Video has 0 frames: {video_path}")
    idxs = uniform_sample_indices(len(raw), num_frames)
    frames = [transform(Image.fromarray(raw[i])) for i in idxs]
    video = torch.stack(frames, dim=0).permute(1, 0, 2, 3)
    return video.unsqueeze(0)


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------

def build_video_list(data_root, shot_type, max_players=None):
    app_base = os.path.join(data_root, "appearance")
    mask_base = os.path.join(data_root, "mask")

    players = sorted(
        d for d in os.listdir(app_base)
        if os.path.isdir(os.path.join(app_base, d))
    )
    all_num = len(players)
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

    return videos, players, all_num


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_all_features(model, video_paths, num_frames, transform, device):
    model.eval()
    all_feats = []
    for vp in tqdm(video_paths, desc="  features"):
        tensor = load_video_tensor(vp, num_frames, transform).to(device)
        backbone_feat = _extract_backbone_features(model, tensor)
        feat = _forward_reid_head_for_feat(model, backbone_feat)
        all_feats.append(feat.cpu())
    return torch.cat(all_feats, dim=0)


# ---------------------------------------------------------------------------
# Jersey colour helpers
# ---------------------------------------------------------------------------

def hex_to_rgb(h):
    """'#1a2b3c' → (26, 43, 60)"""
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


REFERENCE_COLORS = {
    "White":   (240, 240, 240),
    "Black":   (20,  20,  20),
    "Gray":    (140, 140, 140),
    "Red":     (200, 30,  30),
    "Blue":    (30,  60,  180),
    "Navy":    (20,  30,  80),
    "Yellow":  (220, 200, 30),
    "Green":   (30,  140, 50),
    "Purple":  (110, 30,  140),
    "Orange":  (230, 130, 30),
    "Brown":   (120, 70,  40),
    "Pink":    (220, 150, 160),
    "Teal":    (30,  150, 150),
}


def nearest_color_name(rgb):
    """Return the name of the closest reference colour."""
    best_name, best_dist = "Unknown", float("inf")
    for name, ref in REFERENCE_COLORS.items():
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb, ref)))
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name


def load_jersey_colors(csv_path, shot_type):
    """
    Read colour CSV and return a dict mapping
    (identity_folder, video_name) → hex string.
    """
    lookup = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["shot_type"] != shot_type:
                continue
            key = (row["identity_folder"], row["video"])
            lookup[key] = row["jersey_hex"]
    return lookup


def cluster_colors(hex_list, n_clusters=10, seed=42):
    """
    KMeans-cluster a list of hex colours.

    Returns
    -------
    labels      : np.ndarray  – cluster id per sample
    centroids   : list[tuple] – RGB centroid per cluster (0-255)
    names       : list[str]   – human-readable name per cluster
    """
    rgbs = np.array([hex_to_rgb(h) for h in hex_list], dtype=float)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(rgbs)
    centroids = km.cluster_centers_.clip(0, 255).astype(int)

    names = []
    for c in centroids:
        names.append(nearest_color_name(tuple(c)))

    # De-duplicate names by appending a number if needed
    seen = {}
    unique_names = []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        unique_names.append(n if seen[n] == 1 else f"{n}-{seen[n]}")
    # Re-scan to fix the first occurrence if needed
    count = {}
    final_names = []
    for n in names:
        count[n] = count.get(n, 0) + 1
    seen2 = {}
    for n in names:
        seen2[n] = seen2.get(n, 0) + 1
        if count[n] > 1:
            final_names.append(f"{n}-{seen2[n]}")
        else:
            final_names.append(n)

    return labels, [tuple(c) for c in centroids], final_names


# ---------------------------------------------------------------------------
# Cluster-centroid & confused-pair helpers
# ---------------------------------------------------------------------------

def compute_robust_centroids(coords, player_labels, players):
    """Median-based centroids (robust to outliers) in t-SNE space."""
    centroids = {}
    for i, player in enumerate(players):
        pts = coords[np.array([p == player for p in player_labels])]
        centroids[i] = np.median(pts, axis=0)
    return centroids


def _knn_of_cluster(centroids, query_idx, k=5):
    """Return the *k* nearest cluster indices to *query_idx*."""
    dists = []
    for j, c in centroids.items():
        if j == query_idx:
            continue
        dists.append((j, np.linalg.norm(centroids[query_idx] - c)))
    dists.sort(key=lambda x: x[1])
    return [j for j, _ in dists[:k]]


def find_most_divergent_cluster(app_centroids, mask_centroids, n_players,
                                k=5):
    """
    For each cluster, find its *k* nearest neighbours in app-space and in
    mask-space.  Return the cluster whose two neighbour sets differ the most.

    Tie-breaking: among clusters with the same disagreement count, prefer the
    one whose summed rank-displacement is largest (i.e. the neighbours really
    shuffled, not just swapped between rank 5 and 6).

    Returns
    -------
    focus_idx      : int            – index of the chosen cluster
    app_neighbors  : list[int]      – k nearest cluster indices in app-space
    mask_neighbors : list[int]      – k nearest cluster indices in mask-space
    disagreement   : int            – |app_nn \\ mask_nn|  (0–k)
    """
    best_focus = 0
    best_diff = -1
    best_app_nn = []
    best_mask_nn = []

    for i in range(n_players):
        app_nn = _knn_of_cluster(app_centroids, i, k)
        mask_nn = _knn_of_cluster(mask_centroids, i, k)
        diff = len(set(app_nn) - set(mask_nn))
        if diff > best_diff:
            best_diff = diff
            best_focus = i
            best_app_nn = app_nn
            best_mask_nn = mask_nn

    return best_focus, best_app_nn, best_mask_nn, best_diff


def _draw_gradient_line(ax, p1, p2, c1, c2, n_seg=120, linewidth=2.5):
    """Draw a line from *p1* to *p2* with colour smoothly interpolated
    from *c1* (RGBA) at *p1* to *c2* (RGBA) at *p2*."""
    xs = np.linspace(p1[0], p2[0], n_seg + 1)
    ys = np.linspace(p1[1], p2[1], n_seg + 1)
    points = np.column_stack([xs, ys]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    colors = np.array([
        [c1[k] + (c2[k] - c1[k]) * t for k in range(len(c1))]
        for t in np.linspace(0, 1, n_seg)
    ])
    lc = LineCollection(segments, colors=colors, linewidths=linewidth)
    ax.add_collection(lc)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _draw_identity_panel(ax, coords, player_labels, players, title,
                         cmap, focus_idx=None, neighbor_indices=None,
                         centroids=None):
    """Draw a single t-SNE scatter panel on the given *ax*."""
    for i, player in enumerate(players):
        pmask = np.array([p == player for p in player_labels])
        ax.scatter(
            coords[pmask, 0], coords[pmask, 1],
            c=[cmap(i)], label=player.replace("_", " "),
            s=18, alpha=0.75, edgecolors="none",
        )

    if focus_idx is not None and neighbor_indices and centroids:
        c_focus = np.array(cmap(focus_idx))
        for nb in neighbor_indices:
            c_nb = np.array(cmap(nb))
            _draw_gradient_line(ax, centroids[focus_idx], centroids[nb],
                                c_focus, c_nb)

    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)


def plot_tsne_identity_pair(app_coords, mask_coords, player_labels, players,
                            save_path,
                            focus_idx=None, app_neighbors=None,
                            mask_neighbors=None,
                            app_centroids=None, mask_centroids=None):
    """Side-by-side Appearance / Mask t-SNE scatter saved as a single PDF."""
    fig, (ax_app, ax_mask) = plt.subplots(1, 2, figsize=(26, 10))
    cmap = plt.cm.get_cmap("tab20", len(players))

    _draw_identity_panel(ax_app, app_coords, player_labels, players,
                         "Appearance Features", cmap,
                         focus_idx=focus_idx,
                         neighbor_indices=app_neighbors,
                         centroids=app_centroids)

    _draw_identity_panel(ax_mask, mask_coords, player_labels, players,
                         "Mask Features", cmap,
                         focus_idx=focus_idx,
                         neighbor_indices=mask_neighbors,
                         centroids=mask_centroids)

    handles, labels = ax_app.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        fontsize=10, ncol=10,
        loc="upper center", bbox_to_anchor=(0.5, 0.06),
        frameon=True, markerscale=3.0, columnspacing=1.0,
        handletextpad=0.4,
    )

    fig.subplots_adjust(bottom=0.12, wspace=0.15)
    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_tsne_by_jersey(coords, cluster_labels, centroids, cluster_names,
                        title, save_path):
    """Scatter plot coloured by jersey-colour cluster (optional)."""
    fig, ax = plt.subplots(figsize=(13, 10))

    n_clusters = len(centroids)
    palette = plt.cm.get_cmap("tab10", max(n_clusters, 10))

    for cid in range(n_clusters):
        cmask = cluster_labels == cid
        if not cmask.any():
            continue
        plot_color = palette(cid)
        ax.scatter(
            coords[cmask, 0], coords[cmask, 1],
            c=[plot_color], s=18, alpha=0.75, edgecolors="none",
        )

    handles = []
    for cid in range(n_clusters):
        if not (cluster_labels == cid).any():
            continue
        plot_color = palette(cid)
        jersey_rgb01 = tuple(c / 255.0 for c in centroids[cid])
        cnt = int((cluster_labels == cid).sum())
        label = f"{cluster_names[cid]} (n={cnt})"
        dot = plt.Line2D([0], [0], marker="o", color="w",
                         markerfacecolor=plot_color, markersize=8)
        swatch = mpatches.Patch(facecolor=jersey_rgb01, edgecolor="k",
                                linewidth=0.5)
        handles.append((dot, swatch, label))

    leg1 = ax.legend(
        [h[0] for h in handles], [h[2] for h in handles],
        fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0),
        frameon=True, title="Cluster (plot colour)", title_fontsize=9,
    )
    ax.add_artist(leg1)
    ax.legend(
        [h[1] for h in handles], [h[2] for h in handles],
        fontsize=8, loc="lower left", bbox_to_anchor=(1.01, 0.0),
        frameon=True, title="Cluster (actual jersey RGB)", title_fontsize=9,
    )

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="t-SNE Visualization for NBA Video ReID"
    )
    parser.add_argument("--app-config",     type=str, required=True)
    parser.add_argument("--app-checkpoint", type=str, required=True)
    parser.add_argument("--mask-config",    type=str, required=True)
    parser.add_argument("--mask-checkpoint",type=str, required=True)
    parser.add_argument("--color-csv",      type=str,
                        default="/fs/scratch/PAS3184/v3/color_split_final.csv")
    parser.add_argument("--output-dir",     type=str,
                        default="/fs/scratch/PAS3184/v3_vis/t-SNE")
    parser.add_argument("--shot-type",      type=str, default="freethrow")
    parser.add_argument("--num-players",    type=int, default=20)
    parser.add_argument("--color-by-jersey", action="store_true",
                        help="also produce jersey-colour plots (off by default)")
    parser.add_argument("--n-color-clusters", type=int, default=10,
                        help="number of jersey-colour clusters for KMeans")
    parser.add_argument("--n-confused-pairs", type=int, default=5,
                        help="number of confused cluster pairs to highlight")
    parser.add_argument("--perplexity",     type=float, default=30.0)
    parser.add_argument("--num-frames",     type=int, default=None)
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Build video list ─────────────────────────────────────────────────────
    app_cfg = get_cfg_defaults()
    app_cfg.merge_from_file(args.app_config)
    data_root = app_cfg.DATA.ROOT

    videos, players, total_players = build_video_list(
        data_root, args.shot_type, max_players=args.num_players,
    )
    num_classes = total_players
    print(f"Total players in dataset: {total_players}")
    print(f"Using first {len(players)} players ({len(videos)} videos, "
          f"shot_type={args.shot_type})")

    num_frames = args.num_frames or app_cfg.DATA.NUM_FRAMES

    # ── Load jersey-colour mapping (optional) ────────────────────────────────
    cluster_labels = centroids = cluster_names = None
    if args.color_by_jersey:
        color_lookup = load_jersey_colors(args.color_csv, args.shot_type)
        hex_per_video = []
        missing_color = 0
        for v in videos:
            key = (v["player"], v["video_name"])
            h = color_lookup.get(key, None)
            if h is None:
                h = "#808080"
                missing_color += 1
            hex_per_video.append(h)

        if missing_color:
            print(f"  Warning: {missing_color} videos not found in colour CSV "
                  f"(defaulted to gray)")

        cluster_labels, centroids, cluster_names = cluster_colors(
            hex_per_video, n_clusters=args.n_color_clusters, seed=args.seed,
        )
        print(f"Jersey colours clustered into {args.n_color_clusters} groups:")
        for i, (name, cent) in enumerate(zip(cluster_names, centroids)):
            cnt = int((cluster_labels == i).sum())
            print(f"  Cluster {i}: {name:12s}  RGB={cent}  ({cnt} videos)")

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
    ).numpy()
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
    ).numpy()
    del mask_model
    torch.cuda.empty_cache()

    # ── t-SNE ────────────────────────────────────────────────────────────────
    print(f"\nRunning t-SNE (perplexity={args.perplexity}) …")

    tsne_app = TSNE(
        n_components=2, perplexity=args.perplexity,
        random_state=args.seed, init="pca", learning_rate="auto",
    ).fit_transform(app_feats)

    tsne_mask = TSNE(
        n_components=2, perplexity=args.perplexity,
        random_state=args.seed, init="pca", learning_rate="auto",
    ).fit_transform(mask_feats)

    print("  t-SNE done.")

    # ── Player labels ────────────────────────────────────────────────────────
    player_labels = [v["player"] for v in videos]

    # ── Most divergent cluster ───────────────────────────────────────────────
    app_centroids = compute_robust_centroids(tsne_app, player_labels, players)
    mask_centroids = compute_robust_centroids(tsne_mask, player_labels, players)

    focus_idx, app_nn, mask_nn, disagreement = find_most_divergent_cluster(
        app_centroids, mask_centroids, len(players),
        k=args.n_confused_pairs,
    )

    print(f"\nMost divergent cluster: {players[focus_idx]}  "
          f"(disagreement={disagreement}/{args.n_confused_pairs})")
    print(f"  App  top-{args.n_confused_pairs} neighbours: "
          + ", ".join(players[j] for j in app_nn))
    print(f"  Mask top-{args.n_confused_pairs} neighbours: "
          + ", ".join(players[j] for j in mask_nn))

    # ── Generate plots ───────────────────────────────────────────────────────
    print("\nGenerating plots …")

    plot_tsne_identity_pair(
        tsne_app, tsne_mask, player_labels, players,
        save_path=os.path.join(args.output_dir, "identity.pdf"),
        focus_idx=focus_idx,
        app_neighbors=app_nn,
        mask_neighbors=mask_nn,
        app_centroids=app_centroids,
        mask_centroids=mask_centroids,
    )

    if args.color_by_jersey and cluster_labels is not None:
        plot_tsne_by_jersey(
            tsne_app, cluster_labels, centroids, cluster_names,
            title="Appearance Features — by Jersey Colour",
            save_path=os.path.join(args.output_dir, "app_by_jersey_color.png"),
        )
        plot_tsne_by_jersey(
            tsne_mask, cluster_labels, centroids, cluster_names,
            title="Mask Features — by Jersey Colour",
            save_path=os.path.join(args.output_dir, "mask_by_jersey_color.png"),
        )

    # ── Summary TXT ──────────────────────────────────────────────────────────
    txt_path = os.path.join(args.output_dir, "tsne_info.txt")
    with open(txt_path, "w") as f:
        f.write("t-SNE Visualization Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Players          : {len(players)} (first {args.num_players} "
                f"of {total_players})\n")
        f.write(f"Shot type        : {args.shot_type}\n")
        f.write(f"Total videos     : {len(videos)}\n")
        f.write(f"Feature dim      : {app_feats.shape[1]}\n")
        f.write(f"t-SNE perplexity : {args.perplexity}\n")
        f.write(f"Seed             : {args.seed}\n")
        f.write(f"\nApp config       : {args.app_config}\n")
        f.write(f"App checkpoint   : {args.app_checkpoint}\n")
        f.write(f"Mask config      : {args.mask_config}\n")
        f.write(f"Mask checkpoint  : {args.mask_checkpoint}\n")

        f.write(f"\n{'=' * 60}\n")
        f.write("Players included:\n")
        f.write("=" * 60 + "\n")
        for p in players:
            cnt = sum(1 for v in videos if v["player"] == p)
            f.write(f"  {p:30s}  {cnt} videos\n")

        f.write(f"\n{'=' * 60}\n")
        f.write("Most Divergent Cluster:\n")
        f.write("=" * 60 + "\n")
        f.write(f"  Focus player   : {players[focus_idx]}\n")
        f.write(f"  Disagreement   : {disagreement}/{args.n_confused_pairs}\n")
        f.write(f"\n  App  top-{args.n_confused_pairs} neighbours:\n")
        for rank, j in enumerate(app_nn, 1):
            d = np.linalg.norm(app_centroids[focus_idx] - app_centroids[j])
            marker = "" if j in mask_nn else "  ← app only"
            f.write(f"    {rank}. {players[j]:30s}  dist={d:.4f}{marker}\n")
        f.write(f"\n  Mask top-{args.n_confused_pairs} neighbours:\n")
        for rank, j in enumerate(mask_nn, 1):
            d = np.linalg.norm(mask_centroids[focus_idx] - mask_centroids[j])
            marker = "" if j in app_nn else "  ← mask only"
            f.write(f"    {rank}. {players[j]:30s}  dist={d:.4f}{marker}\n")

        if args.color_by_jersey and cluster_labels is not None:
            f.write(f"\n{'=' * 60}\n")
            f.write("Jersey Colour Clusters:\n")
            f.write("=" * 60 + "\n")
            for ci, (name, cent) in enumerate(zip(cluster_names, centroids)):
                cnt = int((cluster_labels == ci).sum())
                f.write(f"  Cluster {ci:2d}: {name:12s}  "
                        f"RGB=({cent[0]:3d},{cent[1]:3d},{cent[2]:3d})  "
                        f"{cnt} videos\n")

    print(f"  Saved: {txt_path}")
    print(f"\nDone!  All outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
