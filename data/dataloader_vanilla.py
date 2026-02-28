#!/usr/bin/env python
"""
Basketball Video ReID Dataset (PyAV version)
Directory structure:
  root/
    appearance/  (or mask/)
      Player_Name/
        3pt/
          video1.mp4 ...
        freethrow/
          video1.mp4 ...
"""

import os
import random
import numpy as np
from collections import defaultdict
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import av


def _derange(lst):
    """
    Return a derangement of lst: a permutation where NO element stays in
    its original position.  This guarantees that every frame selected for
    temporal shuffling is actually displaced (plain shuffle can leave
    elements in place, causing displaced count < k).

    Uses the Fisher-Yates-based rejection approach: shuffle until no fixed
    point remains.  Expected iterations ≈ e ≈ 2.718, O(n) in practice.
    Special case for length-1 lists (derangement impossible) returns as-is.
    """
    if len(lst) <= 1:
        return lst  # derangement impossible for length 1
    arr = lst.copy()
    while True:
        np.random.shuffle(arr)
        if all(arr[i] != lst[i] for i in range(len(lst))):
            return arr


class BasketballVideoDataset(Dataset):
    """Dataset for Basketball Video ReID using MP4 files"""

    def __init__(
        self,
        root,
        video_type="appearance",        # "appearance" or "mask"
        shot_type="both",               # "freethrow", "3pt", or "both"
        num_frames=16,
        is_train=True,
        train_ratio=0.7,
        transform=None,
        seed=42,
        sample_start="beginning",       # "beginning" or "middle"
        temporal_shuffle_ratio=0.0,     # 0.0=off, 0.25, 0.50, 1.0
    ):
        self.root = root
        self.video_type = video_type
        self.shot_type = shot_type
        self.num_frames = num_frames
        self.is_train = is_train
        self.train_ratio = train_ratio
        self.transform = transform
        self.sample_start = sample_start
        self.seed = seed
        self.temporal_shuffle_ratio = temporal_shuffle_ratio

        if not (0.0 <= temporal_shuffle_ratio <= 1.0):
            raise ValueError(f"temporal_shuffle_ratio must be in [0, 1], got {temporal_shuffle_ratio}")

        random.seed(seed)
        np.random.seed(seed)

        self.data, self.pid_list = self._load_data()
        self.num_classes = len(self.pid_list)

    def _load_data(self):
        """Load data and split into train/test by video-level ratio."""
        base_path = os.path.join(self.root, self.video_type)

        identity_folders = sorted(
            d for d in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, d))
        )

        pid_list = [folder.replace("_", " ") for folder in identity_folders]
        pid_to_label = {folder: label for label, folder in enumerate(identity_folders)}

        shot_types_to_load = ["freethrow", "3pt"] if self.shot_type == "both" else [self.shot_type]

        # Collect all videos per identity
        identity_videos = defaultdict(list)

        for identity_folder in identity_folders:
            identity_path = os.path.join(base_path, identity_folder)
            identity_display = identity_folder.replace("_", " ")
            pid = pid_to_label[identity_folder]

            for shot_type in shot_types_to_load:
                shot_path = os.path.join(identity_path, shot_type)
                if not os.path.exists(shot_path):
                    continue

                video_files = sorted(
                    v for v in os.listdir(shot_path)
                    if v.endswith((".mp4", ".MP4", ".avi", ".mov", ".MOV"))
                )

                for video_file in video_files:
                    identity_videos[identity_folder].append({
                        "video_path": os.path.join(shot_path, video_file),
                        "identity": identity_display,
                        "pid": pid,
                        "shot_type": shot_type,
                    })

        # Video-level train/test split
        data = []
        for identity_folder, videos in identity_videos.items():
            if not videos:
                continue

            if self.shot_type == "both":
                freethrow_videos = [v for v in videos if v["shot_type"] == "freethrow"]
                threept_videos   = [v for v in videos if v["shot_type"] == "3pt"]
                ft_train, ft_test   = self._split_videos(freethrow_videos)
                tp_train, tp_test   = self._split_videos(threept_videos)
                data.extend(ft_train + tp_train if self.is_train else ft_test + tp_test)
            else:
                train_v, test_v = self._split_videos(videos)
                data.extend(train_v if self.is_train else test_v)

        split = "Train" if self.is_train else "Test"
        shuffle_str = (
            f"temporal_shuffle={self.temporal_shuffle_ratio*100:.0f}%"
            if self.temporal_shuffle_ratio > 0.0
            else "temporal_shuffle=OFF"
        )
        print(f"\n{'='*60}")
        print(f"{split} Dataset: {self.video_type} | shot_type={self.shot_type}")
        print(f"Identities: {len(pid_list)} | Videos: {len(data)}")
        print(f"Ablation: {shuffle_str}")
        print(f"{'='*60}\n")

        return data, pid_list

    def _split_videos(self, videos):
        """Split video list into train/test by train_ratio."""
        if not videos:
            return [], []
        videos_copy = videos.copy()
        random.shuffle(videos_copy)
        split_idx = max(1, int(len(videos_copy) * self.train_ratio))
        if split_idx >= len(videos_copy) and len(videos_copy) > 1:
            split_idx = len(videos_copy) - 1
        return videos_copy[:split_idx], videos_copy[split_idx:]

    def __len__(self):
        return len(self.data)

    def _apply_temporal_shuffle(self, indices):
        """
        Partially or fully shuffle the sampled frame indices to ablate
        temporal kinematics while preserving static appearance.

        temporal_shuffle_ratio controls what fraction of the T frames are
        displaced from their original temporal position:
          - 0.0  → no shuffle  (baseline)
          - 0.25 → 25% of frames relocated
          - 0.50 → 50% of frames relocated
          - 1.0  → all frames fully shuffled

        The procedure is:
          1. Randomly select `k = round(ratio * T)` positions to shuffle.
          2. Permute only the frame indices sitting at those positions.
          3. Leave the remaining T-k positions untouched.
        This ensures the shuffled frames still come from the *same* video
        clip, so the only thing destroyed is their temporal ordering.
        """
        if self.temporal_shuffle_ratio == 0.0:
            return indices

        T = len(indices)
        k = max(1, round(self.temporal_shuffle_ratio * T))
        k = min(k, T)  # safety clamp

        # Pick which positions will be shuffled
        positions_to_shuffle = np.random.choice(T, size=k, replace=False)

        # Extract the frame indices at those positions and permute them.
        # Use a derangement so EVERY selected position is guaranteed to move
        # (plain np.random.shuffle can leave elements in place, causing the
        # actual displaced count to be less than k — e.g. 7 instead of 8
        # at ratio=0.5 with T=16).
        original_values = [indices[p] for p in positions_to_shuffle]
        permuted_values = _derange(original_values)

        # Write back
        indices = list(indices)
        for pos, new_val in zip(positions_to_shuffle, permuted_values):
            indices[pos] = new_val

        return indices

    def __getitem__(self, idx):
        item = self.data[idx]
        video_path = item["video_path"]

        try:
            container = av.open(video_path)
        except Exception as e:
            raise RuntimeError(f"Failed to open video: {video_path}, Error: {e}")

        all_frames = []
        for frame in container.decode(video=0):
            all_frames.append(frame.to_ndarray(format="rgb24"))
        container.close()

        total_frames = len(all_frames)
        if total_frames == 0:
            raise RuntimeError(f"Video has 0 frames: {video_path}")

        if self.is_train:
            indices = self._random_segment_sampling(total_frames, self.num_frames)
        else:
            indices = self._uniform_segment_sampling(total_frames, self.num_frames)

        # Apply temporal shuffling ablation (no-op when ratio == 0.0)
        indices = self._apply_temporal_shuffle(indices)

        frames = []
        for frame_idx in indices:
            img = Image.fromarray(all_frames[frame_idx])
            if self.transform is not None:
                img = self.transform(img)
            frames.append(img)

        # [T, C, H, W] → [C, T, H, W]
        video = torch.stack(frames, dim=0).permute(1, 0, 2, 3)

        return {
            "video": video,
            "pid": item["pid"],
            "video_path": video_path,
            "identity": item["identity"],
            "shot_type": item["shot_type"],
        }

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _get_sampling_range(self, total_frames):
        if self.sample_start == "middle":
            return total_frames // 2, total_frames
        else:  # "beginning"
            return 0, total_frames

    def _random_segment_sampling(self, total_frames, num_frames):
        """RRS: random segment sampling (TSN-style) for training."""
        start_idx, end_idx = self._get_sampling_range(total_frames)
        indices = np.arange(start_idx, end_idx)
        available = len(indices)
        num_pads = num_frames - (available % num_frames)
        if num_pads != num_frames:
            indices = np.concatenate([indices, np.full(num_pads, end_idx - 1, dtype=int)])
        segments = np.array_split(indices, num_frames)
        return [int(np.random.choice(seg)) for seg in segments]

    def _uniform_segment_sampling(self, total_frames, num_frames):
        """Uniform segment sampling (deterministic) for testing."""
        start_idx, end_idx = self._get_sampling_range(total_frames)
        indices = np.arange(start_idx, end_idx)
        available = len(indices)
        num_pads = num_frames - (available % num_frames)
        if num_pads != num_frames:
            indices = np.concatenate([indices, np.full(num_pads, end_idx - 1, dtype=int)])
        segments = np.array_split(indices, num_frames)
        return [int(seg[0]) for seg in segments]


# ------------------------------------------------------------------
# Transforms
# ------------------------------------------------------------------

def get_train_transforms(cfg):
    return T.Compose([
        T.Resize((cfg.DATA.HEIGHT, cfg.DATA.WIDTH)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3)),
    ])


def get_test_transforms(cfg):
    return T.Compose([
        T.Resize((cfg.DATA.HEIGHT, cfg.DATA.WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def collate_fn(batch):
    return {
        "video":      torch.stack([item["video"] for item in batch]),
        "pid":        torch.tensor([item["pid"] for item in batch], dtype=torch.long),
        "video_paths": [item["video_path"] for item in batch],
        "identities":  [item["identity"] for item in batch],
        "shot_types":  [item["shot_type"] for item in batch],
    }


# ------------------------------------------------------------------
# Builder
# ------------------------------------------------------------------

def build_dataloader(cfg, is_train=True):
    transform = get_train_transforms(cfg) if is_train else get_test_transforms(cfg)
    sample_start = getattr(cfg.DATA, "SAMPLE_START", "middle")
    temporal_shuffle_ratio = getattr(cfg.DATA, "TEMPORAL_SHUFFLE_RATIO", 0.0)

    dataset = BasketballVideoDataset(
        root=cfg.DATA.ROOT,
        video_type=cfg.DATA.VIDEO_TYPE,
        shot_type=cfg.DATA.SHOT_TYPE,
        num_frames=cfg.DATA.NUM_FRAMES,
        is_train=is_train,
        train_ratio=cfg.DATA.TRAIN_RATIO,
        transform=transform,
        seed=cfg.SEED,
        sample_start=sample_start,
        temporal_shuffle_ratio=temporal_shuffle_ratio,
    )

    if is_train and getattr(cfg.DATA, "USE_SAMPLER", False):
        from .sampler import RandomIdentitySampler
        sampler = RandomIdentitySampler(
            data_source=dataset.data,
            batch_size=cfg.DATA.BATCH_SIZE,
            num_instances=cfg.DATA.NUM_INSTANCES,
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=True,
            collate_fn=collate_fn,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.DATA.BATCH_SIZE if is_train else cfg.TEST.BATCH_SIZE,
            shuffle=is_train,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=True,
            drop_last=is_train,
            collate_fn=collate_fn,
        )

    return dataloader, dataset.num_classes


# ------------------------------------------------------------------
# Ablation runner utility
# ------------------------------------------------------------------

def run_temporal_shuffle_ablation(cfg, eval_fn, shuffle_ratios=(0.0, 0.25, 0.5, 1.0)):
    """
    Convenience wrapper to run the temporal shuffling ablation study.

    Parameters
    ----------
    cfg : config object
        Your experiment config. cfg.DATA.TEMPORAL_SHUFFLE_RATIO will be
        overwritten for each ablation run.
    eval_fn : callable
        A function with signature  eval_fn(dataloader, num_classes) -> dict
        that returns a metrics dict, e.g. {"rank1": 0.85, "mAP": 0.72}.
    shuffle_ratios : tuple
        The shuffle fractions to sweep.  Default: (0.0, 0.25, 0.5, 1.0).

    Returns
    -------
    results : dict  {ratio: metrics_dict}

    Example
    -------
    >>> results = run_temporal_shuffle_ablation(cfg, my_eval_fn)
    >>> for ratio, metrics in results.items():
    ...     print(f"Shuffle {ratio*100:.0f}%  Rank-1={metrics['rank1']:.3f}  mAP={metrics['mAP']:.3f}")
    """
    results = {}
    for ratio in shuffle_ratios:
        print(f"\n{'#'*60}")
        print(f"# Temporal Shuffle Ablation: {ratio*100:.0f}% frames shuffled")
        print(f"{'#'*60}")
        cfg.DATA.TEMPORAL_SHUFFLE_RATIO = ratio
        dataloader, num_classes = build_dataloader(cfg, is_train=False)
        metrics = eval_fn(dataloader, num_classes)
        results[ratio] = metrics
        print(f"  → {metrics}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Shuffle %':>10}  {'Rank-1':>8}  {'mAP':>8}")
    print(f"{'-'*30}")
    for ratio, metrics in results.items():
        r1  = metrics.get("rank1", float("nan"))
        mAP = metrics.get("mAP",   float("nan"))
        print(f"{ratio*100:>9.0f}%  {r1:>8.3f}  {mAP:>8.3f}")
    print(f"{'='*60}")

    return results