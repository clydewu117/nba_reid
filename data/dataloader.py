#!/usr/bin/env python
"""
Basketball Video ReID Dataset (PyAV version)
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


class BasketballVideoDataset(Dataset):
    """Dataset for Basketball Video ReID using MP4 files"""
    
    def __init__(self, root, video_type='appearance', shot_type='both',
                 num_frames=16, frame_stride=4, is_train=True, 
                 train_ratio=0.7, transform=None, seed=42, 
                 sample_start='beginning', split_sampling=False, 
                 use_presplit=False, identity_split=False, 
                 train_identities=80, 
                 shot_classification=False, control_20=False):
        self.root = root
        self.video_type = video_type
        self.shot_type = shot_type
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.is_train = is_train
        self.train_ratio = train_ratio
        self.transform = transform
        self.sample_start = sample_start
        self.split_sampling = split_sampling
        self.use_presplit = use_presplit
        self.identity_split = identity_split
        self.train_identities = train_identities
        self.shot_classification = shot_classification
        self.control_20 = control_20
        self.seed = seed
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        
        # 加载数据
        self.data, self.pid_list = self._load_data()
        
        # 根据任务类型确定类别数
        if self.shot_classification:
            # Shot分类任务：2类 (freethrow=0, 3pt=1)
            self.num_classes = 2
            self.shot_to_label = {'freethrow': 0, '3pt': 1}
            
            # 保护机制：验证分类模式配置
            self._validate_classification_mode()
        else:
            # Identity ReID任务
            self.num_classes = len(self.pid_list)
    
    def _load_data(self):
        """加载数据并划分训练/测试集"""
        if self.use_presplit:
            # 使用预先分好的train/test文件夹
            base_path = os.path.join(self.root, self.video_type)
            
            # 获取所有identity文件夹
            identity_folders = sorted([d for d in os.listdir(base_path) 
                                      if os.path.isdir(os.path.join(base_path, d))])
            
            # 收集数据
            data = []
            split_folder = 'train' if self.is_train else 'test'
            identities_with_data = set()  # 记录有数据的identity
            
            for identity_folder in identity_folders:
                identity_path = os.path.join(base_path, identity_folder)
                identity_display = identity_folder.replace('_', ' ')
                
                # 根据shot_type加载对应的视频
                shot_types_to_load = ['freethrow', '3pt'] if self.shot_type == 'both' else [self.shot_type]
                
                # 如果启用control_20，先检查所有shot_type是否都满足>=20的条件
                if self.control_20:
                    skip_identity = False
                    for shot_type in shot_types_to_load:
                        split_path = os.path.join(identity_path, shot_type, split_folder)
                        if not os.path.exists(split_path):
                            skip_identity = True
                            break
                        video_files = sorted([v for v in os.listdir(split_path) 
                                            if v.endswith(('.mp4', '.MP4', '.avi', '.mov', '.MOV'))])
                        if len(video_files) < 20:
                            skip_identity = True
                            break
                    if skip_identity:
                        continue  # 跳过整个identity
                
                identity_has_data = False  # 标记当前identity是否有数据
                temp_videos = []  # 暂存当前identity的视频
                
                for shot_type in shot_types_to_load:
                    # 路径: root/video_type/identity/shot_type/train(或test)
                    split_path = os.path.join(identity_path, shot_type, split_folder)
                    
                    if not os.path.exists(split_path):
                        continue
                    
                    # 获取视频文件
                    video_files = sorted([v for v in os.listdir(split_path) 
                                         if v.endswith(('.mp4', '.MP4', '.avi', '.mov', '.MOV'))])
                    
                    # 如果启用control_20，随机抽取20个
                    if self.control_20:
                        random.shuffle(video_files)
                        video_files = video_files[:20]
                    
                    if len(video_files) > 0:
                        identity_has_data = True
                        identities_with_data.add(identity_folder)
                    
                    for video_file in video_files:
                        video_path = os.path.join(split_path, video_file)
                        temp_videos.append({
                            'video_path': video_path,
                            'identity': identity_display,
                            'identity_folder': identity_folder,  # 用于后续创建pid映射
                            'shot_type': shot_type
                        })
                
                # shot_classification模式下自动启用数据同步：只添加有数据的identity
                # 其他模式：添加所有视频（保持原行为）
                if not self.shot_classification or identity_has_data:
                    data.extend(temp_videos)
            
            # shot_classification模式下自动只包含有数据的identity
            if self.shot_classification:
                identity_folders = sorted(list(identities_with_data))
            
            # 创建identity到pid的映射
            pid_list = [identity.replace('_', ' ') for identity in identity_folders]
            pid_to_label = {folder: label for label, folder in enumerate(identity_folders)}
            
            # 更新data中的pid
            for item in data:
                item['pid'] = pid_to_label[item['identity_folder']]
                del item['identity_folder']  # 删除临时字段
            
            # 打印基本信息
            split = 'Train' if self.is_train else 'Test'
            print(f"\n{'='*60}")
            print(f"{split} Dataset (Pre-split): {self.video_type} + {self.shot_type}")
            if self.shot_classification:
                print(f"SHOT_CLASSIFICATION: True (auto-sync identities with both shot types)")
            print(f"CONTROL_20: {self.control_20}")
            if self.control_20:
                print(f"  → Only keep identities where ALL shot_types have >=20 videos")
            print(f"Sample Start: {self.sample_start}")
            print(f"Split Sampling: {self.split_sampling}")
            if self.split_sampling:
                print(f"  → First 50%: 4 frames, Last 50%: 12 frames")
            print(f"{'='*60}")
            print(f"Identities: {len(pid_list)}")
            print(f"Videos: {len(data)}")
            print(f"{'='*60}\n")
            
            return data, pid_list
        
        # 使用原来的随机划分方式
        base_path = os.path.join(self.root, self.video_type)
        
        # 获取所有identity文件夹
        identity_folders = sorted([d for d in os.listdir(base_path) 
                                  if os.path.isdir(os.path.join(base_path, d))])
        
        # Identity-level split: 将identities划分为训练集和测试集
        if self.identity_split:
            # 使用seed保证可复现
            identity_folders_copy = identity_folders.copy()
            random.shuffle(identity_folders_copy)
            
            # 前 train_identities 个作为训练集，其余作为测试集
            train_identity_folders = identity_folders_copy[:self.train_identities]
            test_identity_folders = identity_folders_copy[self.train_identities:]
            
            # 根据当前模式选择对应的identity
            if self.is_train:
                selected_identities = train_identity_folders
            else:
                selected_identities = test_identity_folders
            
            # 只保留选中的identities
            identity_folders = selected_identities
        
        # 按identity收集所有视频路径
        identity_videos = defaultdict(list)
        identities_with_data = set()  # 记录有数据的identity
        
        for identity_folder in identity_folders:
            identity_path = os.path.join(base_path, identity_folder)
            identity_display = identity_folder.replace('_', ' ')
            
            # 根据shot_type加载对应的视频
            shot_types_to_load = ['freethrow', '3pt'] if self.shot_type == 'both' else [self.shot_type]
            
            # 如果启用control_20，先检查所有shot_type是否都满足>=20的条件
            if self.control_20:
                skip_identity = False
                for shot_type in shot_types_to_load:
                    shot_path = os.path.join(identity_path, shot_type)
                    if not os.path.exists(shot_path):
                        skip_identity = True
                        break
                    video_files_check = sorted([v for v in os.listdir(shot_path) 
                                        if v.endswith(('.mp4', '.MP4', '.avi', '.mov', '.MOV'))])
                    if len(video_files_check) < 20:
                        skip_identity = True
                        break
                if skip_identity:
                    continue  # 跳过整个identity
            
            identity_has_data = False
            
            for shot_type in shot_types_to_load:
                shot_path = os.path.join(identity_path, shot_type)
                
                if not os.path.exists(shot_path):
                    continue
                
                # 获取视频文件
                video_files = sorted([v for v in os.listdir(shot_path) 
                                     if v.endswith(('.mp4', '.MP4', '.avi', '.mov', '.MOV'))])
                
                # 如果启用control_20，随机抽取20个
                if self.control_20:
                    random.shuffle(video_files)
                    video_files = video_files[:20]
                
                if len(video_files) > 0:
                    identity_has_data = True
                
                for video_file in video_files:
                    video_path = os.path.join(shot_path, video_file)
                    identity_videos[identity_folder].append({
                        'video_path': video_path,
                        'identity': identity_display,
                        'identity_folder': identity_folder,  # 临时保存
                        'shot_type': shot_type
                    })
            
            if identity_has_data:
                identities_with_data.add(identity_folder)
        
        # shot_classification模式下自动只保留有数据的identity
        if self.shot_classification:
            identity_folders = sorted(list(identities_with_data))
        
        # 创建pid映射
        pid_list = [identity.replace('_', ' ') for identity in identity_folders]
        pid_to_label = {folder: label for label, folder in enumerate(identity_folders)}
        
        # 划分训练/测试集
        data = []
        
        for identity_folder, videos in identity_videos.items():
            if len(videos) == 0:
                continue
            
            # shot_classification模式下：如果该identity不在有数据的列表中，跳过
            if self.shot_classification and identity_folder not in identities_with_data:
                continue
            
            # 检查identity_folder是否在pid_to_label中
            if identity_folder not in pid_to_label:
                continue
            
            pid = pid_to_label[identity_folder]
            
            # 更新videos中的pid
            for video in videos:
                video['pid'] = pid
                del video['identity_folder']  # 删除临时字段
            
            # Identity-level split: 直接使用该identity的所有视频
            if self.identity_split:
                # 不进行video-level划分，该identity的所有视频都用于当前集合
                data.extend(videos)
            # Video-level split: 分别对freethrow和3pt进行划分
            elif self.shot_type == 'both':
                freethrow_videos = [v for v in videos if v['shot_type'] == 'freethrow']
                threept_videos = [v for v in videos if v['shot_type'] == '3pt']
                
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
        split = 'Train' if self.is_train else 'Test'
        split_mode = 'Identity-level' if self.identity_split else 'Video-level'
        print(f"\n{'='*60}")
        print(f"{split} Dataset ({split_mode}): {self.video_type} + {self.shot_type}")
        if self.identity_split:
            print(f"Split Mode: {self.train_identities} train IDs / {120 - self.train_identities} test IDs")
        if self.shot_classification:
            print(f"SHOT_CLASSIFICATION: True (auto-sync identities with both shot types)")
        print(f"CONTROL_20: {self.control_20}")
        if self.control_20:
            print(f"  → Only keep identities where ALL shot_types have >=20 videos")
        print(f"Sample Start: {self.sample_start}")
        print(f"Split Sampling: {self.split_sampling}")  # 新增：显示分段采样状态
        if self.split_sampling:
            print(f"  → First 50%: 4 frames, Last 50%: 12 frames")
        print(f"{'='*60}")
        print(f"Identities: {len(pid_list)}")
        print(f"Videos: {len(data)}")
        if len(data) > 0:
            print(f"Avg videos per identity: {len(data) / len(pid_list):.1f}")
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
        """Returns: video [C, T, H, W], pid/shot_label, video_path"""
        item = self.data[idx]
        video_path = item['video_path']
        
        # 根据任务类型返回不同的标签
        if self.shot_classification:
            label = self.shot_to_label[item['shot_type']]
        else:
            label = item['pid']
        
        # 使用PyAV读取视频
        try:
            container = av.open(video_path)
        except Exception as e:
            raise RuntimeError(f"Failed to open video: {video_path}, Error: {e}")
        
        # 读取所有视频帧
        all_frames = []
        for frame in container.decode(video=0):
            # 转换为numpy array (RGB格式)
            img = frame.to_ndarray(format='rgb24')
            all_frames.append(img)
        
        container.close()
        
        total_frames = len(all_frames)
        
        if total_frames == 0:
            raise RuntimeError(f"Video has 0 frames: {video_path}")
        
        # 选择采样策略
        if self.split_sampling:
            # 使用前后分段采样：前50%采4帧，后50%采12帧
            if self.is_train:
                indices = self._split_random_sampling(total_frames)
            else:
                indices = self._split_uniform_sampling(total_frames)
        else:
            # 使用原始的RRS采样策略
            if self.is_train:
                indices = self._random_segment_sampling(total_frames, self.num_frames)
            else:
                indices = self._uniform_segment_sampling(total_frames, self.num_frames)
        
        # 从已读取的帧中采样
        frames = []
        for frame_idx in indices:
            frame = all_frames[frame_idx]
            
            # numpy array转PIL Image (已经是RGB格式)
            img = Image.fromarray(frame)
            
            if self.transform is not None:
                img = self.transform(img)
            frames.append(img)
        
        # [T, C, H, W] → [C, T, H, W]
        video = torch.stack(frames, dim=0).permute(1, 0, 2, 3)
        
        return {
            'video': video,
            'pid': label,  # 可能是identity pid或shot_type label
            'video_path': video_path,
            'identity': item['identity'],
            'shot_type': item['shot_type']
        }
    
    def _validate_classification_mode(self):
        """验证分类模式的配置和数据正确性"""
        # 1. 验证shot_type必须为'both'
        if self.shot_type != 'both':
            raise ValueError(
                f"[Classification Mode Error] SHOT_TYPE must be 'both' for classification task, "
                f"but got '{self.shot_type}'. Please set cfg.DATA.SHOT_TYPE = 'both'"
            )
        
        # 2. 验证数据集中确实包含两种shot类型
        shot_types_in_data = set(item['shot_type'] for item in self.data)
        expected_shot_types = {'freethrow', '3pt'}
        
        if shot_types_in_data != expected_shot_types:
            missing = expected_shot_types - shot_types_in_data
            extra = shot_types_in_data - expected_shot_types
            error_msg = "[Classification Mode Error] Data validation failed:\n"
            if missing:
                error_msg += f"  - Missing shot types: {missing}\n"
            if extra:
                error_msg += f"  - Unexpected shot types: {extra}\n"
            error_msg += f"  - Expected: {expected_shot_types}\n"
            error_msg += f"  - Found: {shot_types_in_data}"
            raise ValueError(error_msg)
        
        # 3. 验证每种shot类型都有足够的样本
        shot_type_counts = {'freethrow': 0, '3pt': 0}
        for item in self.data:
            shot_type_counts[item['shot_type']] += 1
        
        split = 'Train' if self.is_train else 'Test'
        print(f"\n{'='*60}")
        print(f"[Classification Mode Validation] {split} Set")
        print(f"  ✓ SHOT_TYPE = 'both'")
        print(f"  ✓ Found 2 shot classes: freethrow, 3pt")
        print(f"  ✓ Sample distribution:")
        print(f"      - freethrow: {shot_type_counts['freethrow']} samples")
        print(f"      - 3pt: {shot_type_counts['3pt']} samples")
        print(f"  ✓ Total samples: {len(self.data)}")
        print(f"{'='*60}\n")
    
    def _get_sampling_range(self, total_frames):
        """
        根据sample_start参数确定采样范围
        Returns: (start_idx, end_idx)
        """
        if self.sample_start == 'middle':
            # 从中间开始：取后50%的帧
            start_idx = total_frames // 2
            end_idx = total_frames
        else:  # 'beginning'
            # 从开头开始：取前50%的帧（如果帧数足够的话）
            # 或者使用全部帧
            start_idx = 0
            end_idx = total_frames
        
        return start_idx, end_idx
    
    def _split_random_sampling(self, total_frames):
        """
        前后分段随机采样（训练用）
        前50%采4帧，后50%采12帧
        """
        # 计算视频中点
        mid_point = total_frames // 2
        
        # 前半部分采4帧
        first_half_indices = self._random_segment_sampling_range(0, mid_point, num_frames=4)
        
        # 后半部分采12帧
        second_half_indices = self._random_segment_sampling_range(mid_point, total_frames, num_frames=12)
        
        # 合并索引（前4帧 + 后12帧 = 16帧）
        all_indices = first_half_indices + second_half_indices
        
        return all_indices
    
    def _split_uniform_sampling(self, total_frames):
        """
        前后分段均匀采样（测试用）
        前50%采4帧，后50%采12帧
        """
        # 计算视频中点
        mid_point = total_frames // 2
        
        # 前半部分采4帧
        first_half_indices = self._uniform_segment_sampling_range(0, mid_point, num_frames=4)
        
        # 后半部分采12帧
        second_half_indices = self._uniform_segment_sampling_range(mid_point, total_frames, num_frames=12)
        
        # 合并索引
        all_indices = first_half_indices + second_half_indices
        
        return all_indices
    
    def _random_segment_sampling_range(self, start_idx, end_idx, num_frames):
        """
        在指定范围内进行随机分段采样
        Args:
            start_idx: 起始帧索引
            end_idx: 结束帧索引
            num_frames: 要采样的帧数
        Returns:
            采样的帧索引列表
        """
        # 创建索引范围
        indices = np.arange(start_idx, end_idx)
        available_frames = len(indices)
        
        if available_frames == 0:
            # 如果范围为空，返回起始索引重复num_frames次
            return [start_idx] * num_frames
        
        # 填充到 num_frames 的倍数
        num_pads = num_frames - (available_frames % num_frames)
        if num_pads != num_frames:
            indices = np.concatenate([indices, np.ones(num_pads, dtype=int) * (end_idx - 1)])
        
        # 分成 num_frames 个均等的部分
        indices_pool = np.array_split(indices, num_frames)
        
        # 从每个segment中随机采样一帧
        sampled_indices = []
        for segment in indices_pool:
            sampled_idx = np.random.choice(segment)
            sampled_indices.append(int(sampled_idx))
        
        return sampled_indices
    
    def _uniform_segment_sampling_range(self, start_idx, end_idx, num_frames):
        """
        在指定范围内进行均匀分段采样
        Args:
            start_idx: 起始帧索引
            end_idx: 结束帧索引
            num_frames: 要采样的帧数
        Returns:
            采样的帧索引列表
        """
        # 创建索引范围
        indices = np.arange(start_idx, end_idx)
        available_frames = len(indices)
        
        if available_frames == 0:
            return [start_idx] * num_frames
        
        # 填充到 num_frames 的倍数
        num_pads = num_frames - (available_frames % num_frames)
        if num_pads != num_frames:
            indices = np.concatenate([indices, np.ones(num_pads, dtype=int) * (end_idx - 1)])
        
        # 分成 num_frames 个均等的部分
        indices_pool = np.array_split(indices, num_frames)
        
        # 从每个segment选择第一帧
        sampled_indices = []
        for segment in indices_pool:
            sampled_indices.append(int(segment[0]))
        
        return sampled_indices
    
    def _random_segment_sampling(self, total_frames, num_frames):
        """
        RRS: Random Segment Sampling for training (TSN-style)
        将视频分成num_frames段，从每段随机选择一帧
        
        优点：保证覆盖整个视频的时间范围
        """
        # Step 1: 确定采样范围
        start_idx, end_idx = self._get_sampling_range(total_frames)
        
        # Step 2: 创建采样范围内的索引
        indices = np.arange(start_idx, end_idx)
        available_frames = len(indices)
        
        # Step 3: 填充到 num_frames 的倍数
        num_pads = num_frames - (available_frames % num_frames)
        if num_pads != num_frames:  # 只在需要时填充
            # 用最后一帧填充
            indices = np.concatenate([indices, np.ones(num_pads, dtype=int) * (end_idx - 1)])
        
        # Step 4: 分成 num_frames 个均等的部分
        indices_pool = np.array_split(indices, num_frames)
        
        # Step 5: 从每个segment中随机采样一帧
        sampled_indices = []
        for segment in indices_pool:
            # 从当前segment随机选择一个索引
            sampled_idx = np.random.choice(segment)
            sampled_indices.append(int(sampled_idx))
        
        return sampled_indices
    
    def _uniform_segment_sampling(self, total_frames, num_frames):
        """
        Uniform Segment Sampling for testing
        将视频分成num_frames段，从每段选择第一帧（确定性）
        """
        # Step 1: 确定采样范围
        start_idx, end_idx = self._get_sampling_range(total_frames)
        
        # Step 2: 创建采样范围内的索引
        indices = np.arange(start_idx, end_idx)
        available_frames = len(indices)
        
        # Step 3: 填充到 num_frames 的倍数
        num_pads = num_frames - (available_frames % num_frames)
        if num_pads != num_frames:
            indices = np.concatenate([indices, np.ones(num_pads, dtype=int) * (end_idx - 1)])
        
        # Step 4: 分成 num_frames 个均等的部分
        indices_pool = np.array_split(indices, num_frames)
        
        # Step 5: 从每个segment选择第一帧（确定性）
        sampled_indices = []
        for segment in indices_pool:
            # 选择segment的第一帧
            sampled_indices.append(int(segment[0]))
        
        return sampled_indices


def get_train_transforms(cfg):
    """训练集数据增强"""
    return T.Compose([
        T.Resize((cfg.DATA.HEIGHT, cfg.DATA.WIDTH)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3))
    ])


def get_test_transforms(cfg):
    """测试集数据增强"""
    return T.Compose([
        T.Resize((cfg.DATA.HEIGHT, cfg.DATA.WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def collate_fn(batch):
    """自定义collate函数"""
    videos = torch.stack([item['video'] for item in batch])
    pids = torch.tensor([item['pid'] for item in batch], dtype=torch.long)
    
    return {
        'video': videos,
        'pid': pids,
        'video_paths': [item['video_path'] for item in batch],
        'identities': [item['identity'] for item in batch],
        'shot_types': [item['shot_type'] for item in batch]
    }


def build_dataloader(cfg, is_train=True):
    """构建DataLoader"""
    transform = get_train_transforms(cfg) if is_train else get_test_transforms(cfg)
    
    # 从config中获取sample_start参数，默认为'middle'
    sample_start = getattr(cfg.DATA, 'SAMPLE_START', 'middle')
    
    # 从config中获取split_sampling参数，默认为False
    split_sampling = getattr(cfg.DATA, 'SPLIT_SAMPLING', False)
    
    # 从config中获取use_presplit参数，默认为False
    use_presplit = getattr(cfg.DATA, 'USE_PRESPLIT', False)
    
    # 从config中获取identity_split参数，默认为False
    identity_split = getattr(cfg.DATA, 'IDENTITY_SPLIT', False)
    
    # 从config中获取train_identities参数，默认为80
    train_identities = getattr(cfg.DATA, 'TRAIN_IDENTITIES', 80)
    
    # 从config中获取shot_classification参数，默认为False
    shot_classification = getattr(cfg.DATA, 'SHOT_CLASSIFICATION', False)
    
    # 从config中获取control_20参数，默认为False
    control_20 = getattr(cfg.DATA, 'CONTROL_20', False)
    
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
        sample_start=sample_start,
        split_sampling=split_sampling,
        use_presplit=use_presplit,
        identity_split=identity_split,
        train_identities=train_identities,
        shot_classification=shot_classification,
        control_20=control_20
    )
    
    # 训练时使用RandomIdentitySampler (但分类模式不使用)
    if is_train and hasattr(cfg.DATA, 'USE_SAMPLER') and cfg.DATA.USE_SAMPLER and not shot_classification:
        from .sampler import RandomIdentitySampler
        
        sampler = RandomIdentitySampler(
            data_source=dataset.data,
            batch_size=cfg.DATA.BATCH_SIZE,
            num_instances=cfg.DATA.NUM_INSTANCES
        )
        
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,  # 注意：使用batch_sampler而不是sampler
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=True,
            collate_fn=collate_fn
        )
    else:
        # 测试时或不使用sampler时或分类模式时，使用正常的随机采样
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.DATA.BATCH_SIZE if is_train else cfg.TEST.BATCH_SIZE,
            shuffle=is_train,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=True,
            drop_last=is_train,
            collate_fn=collate_fn
        )
    
    # 分类模式额外验证：num_classes必须为2
    if shot_classification and dataset.num_classes != 2:
        raise ValueError(
            f"[Classification Mode Error] Expected num_classes=2, but got {dataset.num_classes}. "
            f"This indicates a data loading error."
        )
    
    return dataloader, dataset.num_classes