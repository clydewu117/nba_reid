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
import pickle

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
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

_FONT_PATH = os.path.join(os.path.dirname(__file__), "DMSans-Regular.ttf")
if os.path.isfile(_FONT_PATH):
    fm.fontManager.addfont(_FONT_PATH)


def _setup_font(font_name="Times New Roman", bold=True):
    """Configure the matplotlib font globally.

    Parameters
    ----------
    font_name : str
        Any font name recognised by matplotlib (e.g. "Times New Roman",
        "Arial", "DMSans", or a path-registered font).
    bold : bool
        If *True*, set the default weight to bold for all text elements.
    """
    matplotlib.rcParams["font.family"] = font_name
    if bold:
        matplotlib.rcParams["font.weight"] = "bold"
        matplotlib.rcParams["axes.titleweight"] = "bold"
        matplotlib.rcParams["axes.labelweight"] = "bold"

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


_RANK_STYLES = [
    {"kind": "line", "linestyle": "-",   "lw": 2.4, "label": "Top-1 Neighbor"},
    {"kind": "line", "linestyle": "--",  "lw": 2.2, "label": "Top-2 Neighbor"},
    {"kind": "line", "linestyle": "-.",  "lw": 2.0, "label": "Top-3 Neighbor"},
    {"kind": "line", "linestyle": ":",   "lw": 1.8, "label": "Top-4 Neighbor"},
    {"kind": "star", "ms": 3.5, "markevery": 0.02, "label": "Top-5 Neighbor"},
]
_RANK_LINE_COLOR = "#2c2c2c"


def _distance_based_colors(pts, centroid, base_rgba, max_lighten=0.65):
    """Per-point colours graded deep→light by distance to *centroid*."""
    dists = np.linalg.norm(pts - centroid, axis=1)
    max_d = dists.max()
    normed = dists / max_d if max_d > 0 else np.zeros_like(dists)
    white = np.array([1.0, 1.0, 1.0])
    colors = np.empty((len(pts), 4))
    for i, t in enumerate(normed):
        colors[i, :3] = np.clip(
            base_rgba[:3] * (1 - t * max_lighten) + white * t * max_lighten,
            0.0, 1.0,
        )
        colors[i, 3] = base_rgba[3]
    return colors


def _draw_convex_hull_boundary(ax, points, color, linewidth=2.8, alpha=0.65,
                               pad_frac=0.04, smooth_k=3, n_eval=200):
    """Draw a smooth, padded dashed boundary around *points*.

    Uses the convex hull vertices, pads them outward from the centroid,
    then fits a periodic B-spline for a rounded appearance.
    """
    if len(points) < 3:
        return
    try:
        hull = ConvexHull(points)
    except Exception:
        return

    verts = points[hull.vertices]
    centroid = verts.mean(axis=0)

    # Pad outward from centroid
    dirs = verts - centroid
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    pad = pad_frac * norms.max()
    verts_padded = verts + dirs / norms * pad

    # Close the polygon for periodic spline
    verts_closed = np.vstack([verts_padded, verts_padded[0]])

    k = min(smooth_k, len(verts_padded) - 1)
    if k < 1:
        return
    try:
        tck, _ = splprep([verts_closed[:, 0], verts_closed[:, 1]],
                         s=0, per=True, k=k)
        t_new = np.linspace(0, 1, n_eval)
        xs, ys = splev(t_new, tck)
    except Exception:
        xs, ys = verts_closed[:, 0], verts_closed[:, 1]

    ax.plot(xs, ys, color=color, linestyle="--",
            linewidth=linewidth, alpha=alpha)


def _draw_ranked_line(ax, p1, p2, rank):
    """Draw a connection line whose style indicates *rank*."""
    if rank >= len(_RANK_STYLES):
        return
    style = _RANK_STYLES[rank]
    if style["kind"] == "line":
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color=_RANK_LINE_COLOR, linestyle=style["linestyle"],
                linewidth=style["lw"], alpha=0.75, solid_capstyle="round")
    else:
        dist = np.linalg.norm(np.array(p2) - np.array(p1))
        spacing = style["ms"] * 0.5
        n_pts = max(int(dist / spacing), 5)
        ts = np.linspace(0, 1, n_pts)
        xs = p1[0] + ts * (p2[0] - p1[0])
        ys = p1[1] + ts * (p2[1] - p1[1])
        ax.plot(xs, ys, marker="*", linestyle="none",
                markersize=style["ms"], color=_RANK_LINE_COLOR, alpha=0.75)


def _save_fig(fig, save_path, dpi=400):
    """Save *fig* as PDF, PNG, and SVG (deriving paths from *save_path*)."""
    base, _ = os.path.splitext(save_path)
    for ext in (".pdf", ".png", ".svg"):
        out = base + ext
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _draw_identity_panel(ax, coords, player_labels, players, title,
                         cmap, focus_idx=None, neighbor_indices=None,
                         centroids=None):
    """Draw a single t-SNE scatter panel on the given *ax*."""
    for i, player in enumerate(players):
        pmask = np.array([p == player for p in player_labels])
        pts = coords[pmask]
        base_rgba = np.array(cmap(i))

        centroid = centroids[i] if centroids else np.median(pts, axis=0)

        point_colors = _distance_based_colors(pts, centroid, base_rgba)
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=point_colors, label=player.replace("_", " "),
                   s=18, alpha=0.85, edgecolors="none")

        _draw_convex_hull_boundary(ax, pts, base_rgba[:3])

    if focus_idx is not None and neighbor_indices and centroids:
        for rank, nb in enumerate(neighbor_indices):
            _draw_ranked_line(ax, centroids[focus_idx], centroids[nb], rank)

    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)


def plot_tsne_identity_pair(app_coords, mask_coords, player_labels, players,
                            save_path,
                            focus_idx=None, app_neighbors=None,
                            mask_neighbors=None,
                            app_centroids=None, mask_centroids=None,
                            app_sample_videos=None, mask_sample_videos=None):
    """Side-by-side Appearance / Mask t-SNE scatter saved as a single PDF.

    When *app_sample_videos* / *mask_sample_videos* are provided (dicts
    mapping player index → video dict), representative frame thumbnails
    are drawn in the margins of each panel with connecting lines to the
    corresponding class centroids.
    """
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

    if app_sample_videos and focus_idx is not None and app_neighbors:
        _add_sample_insets(ax_app, app_centroids, focus_idx, app_neighbors,
                           app_sample_videos, players, cmap, side="left")
    if mask_sample_videos and focus_idx is not None and mask_neighbors:
        _add_sample_insets(ax_mask, mask_centroids, focus_idx, mask_neighbors,
                           mask_sample_videos, players, cmap, side="right")

    player_handles, player_labels_txt = ax_app.get_legend_handles_labels()

    rank_handles = []
    for style in _RANK_STYLES:
        if style["kind"] == "line":
            h = plt.Line2D(
                [0], [0], color=_RANK_LINE_COLOR,
                linestyle=style["linestyle"], linewidth=style["lw"],
                label=style["label"],
            )
        else:
            h = plt.Line2D(
                [0], [0], color=_RANK_LINE_COLOR,
                marker="*", linestyle="none", markersize=style["ms"],
                label=style["label"],
            )
        rank_handles.append(h)
    ax_app.legend(handles=rank_handles, loc="upper right", fontsize=9,
                  frameon=True, framealpha=0.9, title="Neighbor Rank",
                  title_fontsize=10, numpoints=6, handlelength=5.0)
    ax_mask.legend(handles=rank_handles, loc="upper right", fontsize=9,
                   frameon=True, framealpha=0.9, title="Neighbor Rank",
                   title_fontsize=10, numpoints=6, handlelength=5.0)

    fig.legend(
        player_handles, player_labels_txt,
        fontsize=10, ncol=10,
        loc="upper center", bbox_to_anchor=(0.5, 0.04),
        frameon=True, markerscale=3.0, columnspacing=1.0,
        handletextpad=0.4,
    )

    fig.subplots_adjust(bottom=0.12, wspace=0.15)
    _save_fig(fig, save_path)
    plt.close(fig)


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
    _save_fig(fig, save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sample image helpers
# ---------------------------------------------------------------------------

def _find_centroid_video(videos, player_labels, tsne_coords, centroid, player):
    """Return the video dict whose t-SNE point is nearest the *centroid*."""
    pmask = np.array([p == player for p in player_labels])
    indices = np.where(pmask)[0]
    dists = np.linalg.norm(tsne_coords[indices] - centroid, axis=1)
    return videos[indices[np.argmin(dists)]]


def _extract_middle_frame(video_path, size=(224, 224)):
    """Extract and resize the middle frame of a video file."""
    try:
        frames = decode_all_frames(video_path)
    except Exception:
        return Image.new("RGB", size, (200, 200, 200))
    if not frames:
        return Image.new("RGB", size, (200, 200, 200))
    img = Image.fromarray(frames[len(frames) // 2])
    return img.resize(size, Image.LANCZOS)


def _add_sample_insets(ax, centroids, focus_idx, neighbor_indices,
                       sample_videos, players, cmap, side="left",
                       img_size=200, zoom=0.45):
    """Place sample thumbnails in a compact 2-col × 3-row grid on one side.

    *side* controls placement: ``'left'`` for the appearance panel (images
    appear to the left of the axes) and ``'right'`` for the mask panel
    (images appear to the right).  Connecting lines run from each class
    centroid to its thumbnail.
    """
    all_indices = [focus_idx] + list(neighbor_indices)
    n = len(all_indices)

    grid_cols, grid_rows = 2, (n + 1) // 2
    col_step = 0.16
    row_step = 0.22
    x_base = -0.34 if side == "left" else 1.12
    y_top = 0.8

    positions = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            if len(positions) >= n:
                break
            positions.append((x_base + c * col_step,
                              y_top - r * row_step))

    for k, (idx, (bx, by)) in enumerate(zip(all_indices, positions)):
        if idx not in sample_videos:
            continue
        video = sample_videos[idx]
        img = _extract_middle_frame(video["app_path"],
                                    size=(img_size, img_size))
        img_arr = np.array(img)
        imagebox = OffsetImage(img_arr, zoom=zoom)

        is_focus = (k == 0)
        rgba = cmap(idx)
        edge_color = "red" if is_focus else rgba[:3]
        lw = 2.8 if is_focus else 1.8

        ab = AnnotationBbox(
            imagebox, xy=tuple(centroids[idx]),
            xybox=(bx, by),
            xycoords="data", boxcoords="axes fraction",
            arrowprops=dict(arrowstyle="-", color=edge_color, lw=1.5),
            frameon=True,
            bboxprops=dict(edgecolor=edge_color, linewidth=lw),
            pad=0.3,
        )
        ab.set_clip_on(False)
        ax.add_artist(ab)

        name = players[idx].replace("_", " ")
        label = f"Focus: {name}" if is_focus else f"Top-{k}: {name}"
        txt = ax.text(bx, by - 0.10, label, transform=ax.transAxes,
                      fontsize=7, ha="center", va="top", fontweight="bold",
                      color="red" if is_focus else "k")
        txt.set_clip_on(False)


def plot_sample_images(focus_player, focus_video,
                       app_nb_players, app_nb_videos,
                       mask_nb_players, mask_nb_videos,
                       save_path):
    """Visualise representative frames for the focus player and its
    top-K nearest neighbours from both appearance and mask space."""
    k = len(app_nb_players)
    n_cols = 1 + k
    fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 7.5))

    for row, (nb_players, nb_videos, row_label) in enumerate([
        (app_nb_players, app_nb_videos, "App Neighbors"),
        (mask_nb_players, mask_nb_videos, "Mask Neighbors"),
    ]):
        all_p = [focus_player] + nb_players
        all_v = [focus_video] + nb_videos
        for col, (player, video) in enumerate(zip(all_p, all_v)):
            ax = axes[row, col]
            img = _extract_middle_frame(video["app_path"])
            ax.imshow(img)
            ax.axis("off")
            name = player.replace("_", " ")
            if col == 0:
                ax.set_title(f"Focus\n{name}", fontsize=9,
                             fontweight="bold", color="red")
            else:
                ax.set_title(f"Top-{col}\n{name}", fontsize=9)

    fig.text(0.01, 0.73, "App\nNeighbors", fontsize=11, fontweight="bold",
             va="center", ha="center", rotation=90)
    fig.text(0.01, 0.30, "Mask\nNeighbors", fontsize=11, fontweight="bold",
             va="center", ha="center", rotation=90)

    fig.suptitle("Most Divergent Individual & Nearest Neighbors",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0.03, 0, 1, 0.98])
    _save_fig(fig, save_path)
    plt.close(fig)


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
    parser.add_argument("--no-cache",       action="store_true",
                        help="force re-extraction even if cached features exist")
    parser.add_argument("--n-confused-pairs", type=int, default=5,
                        help="number of confused cluster pairs to highlight")
    parser.add_argument("--show-sample-insets", action="store_true",
                        help="embed sample-image thumbnails beside the "
                             "t-SNE scatter (off by default)")
    parser.add_argument("--perplexity",     type=float, default=30.0)
    parser.add_argument("--num-frames",     type=int, default=None)
    parser.add_argument("--font",           type=str,
                        default="Times New Roman",
                        help="font name for plots (default: Times New Roman)")
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    _setup_font(args.font, bold=True)
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Feature cache ────────────────────────────────────────────────────────
    cache_path = os.path.join(args.output_dir, "feature_cache.pkl")

    if not args.no_cache and os.path.isfile(cache_path):
        print(f"Loading cached features from {cache_path} …")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        app_feats = cache["app_feats"]
        mask_feats = cache["mask_feats"]
        videos = cache["videos"]
        players = cache["players"]
        total_players = cache["total_players"]
        print(f"  {len(players)} players, {len(videos)} videos, "
              f"feat dim={app_feats.shape[1]}")
    else:
        # ── Build video list ──────────────────────────────────────────────
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

        # ── Appearance model ──────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("Loading appearance model …")
        app_cfg.MODEL.NUM_CLASSES = num_classes
        if not torch.cuda.is_available():
            app_cfg.NUM_GPUS = 0
        app_cfg.freeze()

        app_model = build_model(app_cfg)
        ckpt = torch.load(args.app_checkpoint, map_location=device,
                          weights_only=False)
        app_model.load_state_dict(ckpt["model_state_dict"])
        app_model = app_model.to(device).eval()
        print(f"  Loaded epoch {ckpt.get('epoch', '?')}")

        app_transform = T.Compose([
            T.Resize((app_cfg.DATA.HEIGHT, app_cfg.DATA.WIDTH)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        print(f"Extracting appearance features ({len(videos)} videos) …")
        app_feats = extract_all_features(
            app_model, [v["app_path"] for v in videos],
            num_frames, app_transform, device,
        ).numpy()
        del app_model
        torch.cuda.empty_cache()

        # ── Mask model ────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("Loading mask model …")
        mask_cfg = get_cfg_defaults()
        mask_cfg.merge_from_file(args.mask_config)
        mask_cfg.MODEL.NUM_CLASSES = num_classes
        if not torch.cuda.is_available():
            mask_cfg.NUM_GPUS = 0
        mask_cfg.freeze()

        mask_model = build_model(mask_cfg)
        ckpt = torch.load(args.mask_checkpoint, map_location=device,
                          weights_only=False)
        mask_model.load_state_dict(ckpt["model_state_dict"])
        mask_model = mask_model.to(device).eval()
        print(f"  Loaded epoch {ckpt.get('epoch', '?')}")

        mask_transform = T.Compose([
            T.Resize((mask_cfg.DATA.HEIGHT, mask_cfg.DATA.WIDTH)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        print(f"Extracting mask features ({len(videos)} videos) …")
        mask_feats = extract_all_features(
            mask_model, [v["mask_path"] for v in videos],
            num_frames, mask_transform, device,
        ).numpy()
        del mask_model
        torch.cuda.empty_cache()

        # ── Save cache ────────────────────────────────────────────────────
        cache = {
            "app_feats": app_feats,
            "mask_feats": mask_feats,
            "videos": videos,
            "players": players,
            "total_players": total_players,
        }
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"  Cached features → {cache_path}")

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

    # ── Sample videos for the most divergent individual (optional) ──────────
    app_sample_videos = None
    mask_sample_videos = None
    if args.show_sample_insets:
        print("\nFinding representative frames for insets …")
        app_sample_videos = {}
        app_sample_videos[focus_idx] = _find_centroid_video(
            videos, player_labels, tsne_app,
            app_centroids[focus_idx], players[focus_idx],
        )
        for j in app_nn:
            app_sample_videos[j] = _find_centroid_video(
                videos, player_labels, tsne_app,
                app_centroids[j], players[j],
            )

        mask_sample_videos = {}
        mask_sample_videos[focus_idx] = _find_centroid_video(
            videos, player_labels, tsne_mask,
            mask_centroids[focus_idx], players[focus_idx],
        )
        for j in mask_nn:
            mask_sample_videos[j] = _find_centroid_video(
                videos, player_labels, tsne_mask,
                mask_centroids[j], players[j],
            )

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
        app_sample_videos=app_sample_videos,
        mask_sample_videos=mask_sample_videos,
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
