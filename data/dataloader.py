#!/usr/bin/env python
"""
Basketball Video ReID Dataset (OpenCV version)
"""

import os
import random
import numpy as np
from collections import defaultdict
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import cv2


class BasketballVideoDataset(Dataset):
    """Dataset for Basketball Video ReID using MP4 files"""

    def __init__(
        self,
        root,
        video_type="appearance",
        shot_type="both",
        num_frames=16,
        frame_stride=4,
        is_train=True,
        train_ratio=0.75,
        transform=None,
        seed=42,
    ):
        self.root = root
        self.video_type = video_type
        self.shot_type = shot_type
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.is_train = is_train
        self.train_ratio = train_ratio
        self.transform = transform

        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)

        # 加载数据
        self.data, self.pid_list = self._load_data()
        self.num_classes = len(self.pid_list)

    def _load_data(self):
        """加载数据并划分训练/测试集"""
        base_path = os.path.join(self.root, self.video_type)

        # 获取所有identity文件夹
        identity_folders = sorted(
            [
                d
                for d in os.listdir(base_path)
                if os.path.isdir(os.path.join(base_path, d))
            ]
        )

        # 创建identity到pid的映射
        pid_list = [identity.replace("_", " ") for identity in identity_folders]
        pid_to_label = {folder: label for label, folder in enumerate(identity_folders)}

        # 按identity收集所有视频路径
        identity_videos = defaultdict(list)

        for identity_folder in identity_folders:
            identity_path = os.path.join(base_path, identity_folder)
            identity_display = identity_folder.replace("_", " ")
            pid = pid_to_label[identity_folder]

            # 根据shot_type加载对应的视频
            shot_types_to_load = (
                ["freethrow", "3pt"] if self.shot_type == "both" else [self.shot_type]
            )

            for shot_type in shot_types_to_load:
                shot_path = os.path.join(identity_path, shot_type)

                if not os.path.exists(shot_path):
                    continue

                # 获取视频文件
                video_files = sorted(
                    [
                        v
                        for v in os.listdir(shot_path)
                        if v.endswith((".mp4", ".MP4", ".avi", ".mov", ".MOV"))
                    ]
                )

                for video_file in video_files:
                    video_path = os.path.join(shot_path, video_file)
                    identity_videos[identity_folder].append(
                        {
                            "video_path": video_path,
                            "pid": pid,
                            "identity": identity_display,
                            "shot_type": shot_type,
                        }
                    )

        # 划分训练/测试集
        data = []

        for identity_folder, videos in identity_videos.items():
            if len(videos) == 0:
                continue

            # 分别对freethrow和3pt进行划分
            if self.shot_type == "both":
                freethrow_videos = [v for v in videos if v["shot_type"] == "freethrow"]
                threept_videos = [v for v in videos if v["shot_type"] == "3pt"]

                freethrow_train, freethrow_test = self._split_videos(freethrow_videos)
                threept_train, threept_test = self._split_videos(threept_videos)

                if self.is_train:
                    data.extend(freethrow_train)
                    data.extend(threept_train)
                else:
                    data.extend(freethrow_test)
                    data.extend(threept_test)
            else:
                train_videos, test_videos = self._split_videos(videos)
                data.extend(train_videos if self.is_train else test_videos)

        # 打印基本信息
        split = "Train" if self.is_train else "Test"
        print(f"\n{'='*60}")
        print(f"{split} Dataset: {self.video_type} + {self.shot_type}")
        print(f"{'='*60}")
        print(f"Identities: {len(pid_list)}")
        print(f"Videos: {len(data)}")
        print(f"{'='*60}\n")

        return data, pid_list

    def _split_videos(self, videos):
        """将视频列表按train_ratio划分，确保无重复"""
        if len(videos) == 0:
            return [], []

        videos_copy = videos.copy()
        random.shuffle(videos_copy)

        split_idx = int(len(videos_copy) * self.train_ratio)

        # 确保至少有1个测试样本
        if split_idx >= len(videos_copy) and len(videos_copy) > 1:
            split_idx = len(videos_copy) - 1

        train_videos = videos_copy[:split_idx]
        test_videos = videos_copy[split_idx:]

        return train_videos, test_videos

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Returns: video [C, T, H, W], pid, video_path"""
        item = self.data[idx]
        video_path = item["video_path"]
        pid = item["pid"]

        # 使用OpenCV读取视频
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        # 获取视频总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            raise RuntimeError(f"Video has 0 frames: {video_path}")

        # 计算采样indices
        if total_frames < self.num_frames:
            indices = np.linspace(
                0, max(0, total_frames - 1), self.num_frames, dtype=int
            ).tolist()
        else:
            required_frames = self.num_frames * self.frame_stride

            if total_frames >= required_frames:
                if self.is_train:
                    start_idx = random.randint(0, total_frames - required_frames)
                else:
                    start_idx = (total_frames - required_frames) // 2
                indices = list(
                    range(start_idx, start_idx + required_frames, self.frame_stride)
                )
            else:
                indices = np.linspace(
                    0, total_frames - 1, self.num_frames, dtype=int
                ).tolist()

        # 读取帧
        frames = []
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                cap.release()
                raise RuntimeError(
                    f"Failed to read frame {frame_idx} from {video_path}"
                )

            # BGR → RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)

            if self.transform is not None:
                img = self.transform(img)
            frames.append(img)

        cap.release()

        # [T, C, H, W] → [C, T, H, W]
        video = torch.stack(frames, dim=0).permute(1, 0, 2, 3)

        return {
            "video": video,
            "pid": pid,
            "video_path": video_path,
            "identity": item["identity"],
            "shot_type": item["shot_type"],
        }


def get_train_transforms(cfg):
    """训练集数据增强"""
    # Note: Apply ToTensor before ColorJitter to use the tensor implementation,
    # which avoids a known PIL path overflow when hue_factor is negative.
    # See: OverflowError in torchvision._functional_pil.adjust_hue with uint8 casts.
    return T.Compose(
        [
            T.Resize((cfg.DATA.HEIGHT, cfg.DATA.WIDTH)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3)),
        ]
    )


def get_test_transforms(cfg):
    """测试集数据增强"""
    return T.Compose(
        [
            T.Resize((cfg.DATA.HEIGHT, cfg.DATA.WIDTH)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def collate_fn(batch):
    """自定义collate函数"""
    videos = torch.stack([item["video"] for item in batch])
    pids = torch.tensor([item["pid"] for item in batch], dtype=torch.long)

    return {
        "video": videos,
        "pid": pids,
        "video_paths": [item["video_path"] for item in batch],
        "identities": [item["identity"] for item in batch],
        "shot_types": [item["shot_type"] for item in batch],
    }


def build_dataloader(cfg, is_train=True):
    """构建DataLoader"""
    transform = get_train_transforms(cfg) if is_train else get_test_transforms(cfg)

    dataset = BasketballVideoDataset(
        root=cfg.DATA.ROOT,
        video_type=cfg.DATA.VIDEO_TYPE,
        shot_type=cfg.DATA.SHOT_TYPE,
        num_frames=cfg.DATA.NUM_FRAMES,
        frame_stride=cfg.DATA.FRAME_STRIDE,
        is_train=is_train,
        train_ratio=cfg.DATA.TRAIN_RATIO,
        transform=transform,
        seed=cfg.SEED,
    )

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
