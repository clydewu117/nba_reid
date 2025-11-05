#!/usr/bin/env python
"""
NBA ReID Inference Script
支持多模型推理：TimeSformer, MViT, UniFormerV2
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

# 添加父目录到路径以便导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from config.defaults import get_cfg_defaults
from models.build import build_model
from data.dataloader import build_dataloader


class ReIDInference:
    """NBA ReID 推理类"""

    def __init__(
        self,
        model_name: str,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: str = "cuda:0",
    ):
        """
        初始化推理器

        Args:
            model_name: 模型名称 ('timesformer', 'mvit', 'uniformerv2')
            checkpoint_path: 模型权重文件路径
            config_path: 配置文件路径（可选）
            device: 计算设备
        """
        self.model_name = model_name.lower()
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 验证模型名称
        valid_models = ["timesformer", "mvit", "uniformerv2"]
        if self.model_name not in valid_models:
            raise ValueError(f"模型名称必须是以下之一: {valid_models}")

        # 加载配置
        self.cfg = self._load_config(config_path)

        # 构建并加载模型
        self.model = self._load_model()

        print(f"✓ 成功加载 {model_name} 模型")
        print(f"  设备: {self.device}")
        print(f"  权重: {checkpoint_path}")

    def _load_config(self, config_path: Optional[str] = None) -> object:
        """加载配置"""
        cfg = get_cfg_defaults()

        # 根据模型名称设置对应的配置
        model_config_map = {
            "timesformer": {
                "MODEL.NAME": "TimeSformerReID",
                "MODEL.MODEL_NAME": "TimeSformerReID",
                "MODEL.ARCH": "timesformer",
            },
            "mvit": {
                "MODEL.NAME": "MViTReID",
                "MODEL.MODEL_NAME": "MViTReID",
                "MODEL.ARCH": "mvit",
            },
            "uniformerv2": {
                "MODEL.NAME": "Uniformerv2ReID",
                "MODEL.MODEL_NAME": "Uniformerv2ReID",
                "MODEL.ARCH": "uniformerv2",
            },
        }

        # 应用模型特定配置
        if self.model_name in model_config_map:
            cfg.defrost()
            for key, value in model_config_map[self.model_name].items():
                keys = key.split(".")
                node = cfg
                for k in keys[:-1]:
                    node = getattr(node, k)
                setattr(node, keys[-1], value)
            cfg.freeze()

        # 如果提供了配置文件，加载并合并
        if config_path and os.path.exists(config_path):
            cfg.merge_from_file(config_path)

        return cfg

    def _load_model(self) -> nn.Module:
        """加载模型和权重"""
        # 加载权重以检查训练时的配置
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")

        # 处理不同的 checkpoint 格式
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            num_classes = checkpoint.get("num_classes", None)

            # 打印额外信息
            if "epoch" in checkpoint:
                print(f"  Epoch: {checkpoint['epoch']}")
            if "rank1" in checkpoint:
                print(f"  Rank-1: {checkpoint['rank1']:.2%}")
            if "mAP" in checkpoint:
                print(f"  mAP: {checkpoint['mAP']:.2%}")
        else:
            state_dict = checkpoint
            num_classes = None

        # 从state_dict推断num_frames（对于TimeSformer）
        num_frames_from_ckpt = None
        if "backbone.time_embed" in state_dict and self.model_name == "timesformer":
            num_frames_from_ckpt = state_dict["backbone.time_embed"].shape[1]
            print(f"  Detected num_frames from checkpoint: {num_frames_from_ckpt}")

        # 更新配置以匹配checkpoint
        self.cfg.defrost()

        # 设置NUM_CLASSES
        if num_classes is not None:
            self.cfg.MODEL.NUM_CLASSES = num_classes
        else:
            # 从classifier权重推断（如果存在）
            if "reid_head.classifier.weight" in state_dict:
                num_classes = state_dict["reid_head.classifier.weight"].shape[0]
                self.cfg.MODEL.NUM_CLASSES = num_classes
                print(f"  Detected num_classes from checkpoint: {num_classes}")
            else:
                self.cfg.MODEL.NUM_CLASSES = 1

        # 设置num_frames以匹配checkpoint
        if num_frames_from_ckpt is not None:
            self.cfg.DATA.NUM_FRAMES = num_frames_from_ckpt
            print(
                f"  Updated DATA.NUM_FRAMES to {num_frames_from_ckpt} to match checkpoint"
            )

        self.cfg.NUM_GPUS = 1 if torch.cuda.is_available() else 0
        self.cfg.freeze()

        # 构建模型（使用更新后的配置）
        model = build_model(self.cfg)

        # 加载 state_dict（允许不完全匹配）
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(
                f"  Warning: Missing keys in checkpoint: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}"
            )
        if unexpected_keys:
            print(
                f"  Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}"
            )

        # 转移到设备并设置为评估模式
        model = model.to(self.device)
        model.eval()

        return model

    @torch.no_grad()
    def extract_features(self, video_tensor: torch.Tensor) -> np.ndarray:
        """
        提取视频特征

        Args:
            video_tensor: 视频张量 [B, C, T, H, W] 或 [C, T, H, W]

        Returns:
            特征向量 [B, D] 或 [D]
        """
        # 确保是 4D 或 5D
        if video_tensor.dim() == 4:
            video_tensor = video_tensor.unsqueeze(0)

        # 转移到设备
        video_tensor = video_tensor.to(self.device)

        # 前向传播
        with autocast():
            features = self.model(video_tensor)

        # 返回 numpy 数组
        features = features.cpu().numpy()

        return features.squeeze() if features.shape[0] == 1 else features

    @torch.no_grad()
    def compute_similarity(
        self,
        query_features: np.ndarray,
        gallery_features: np.ndarray,
        metric: str = "cosine",
    ) -> np.ndarray:
        """
        计算查询和gallery之间的相似度

        Args:
            query_features: 查询特征 [N_q, D]
            gallery_features: gallery特征 [N_g, D]
            metric: 距离度量 ('cosine' 或 'euclidean')

        Returns:
            相似度矩阵 [N_q, N_g]
        """
        # 确保是 2D
        if query_features.ndim == 1:
            query_features = query_features[np.newaxis, :]
        if gallery_features.ndim == 1:
            gallery_features = gallery_features[np.newaxis, :]

        if metric == "cosine":
            # 归一化
            query_norm = query_features / (
                np.linalg.norm(query_features, axis=1, keepdims=True) + 1e-12
            )
            gallery_norm = gallery_features / (
                np.linalg.norm(gallery_features, axis=1, keepdims=True) + 1e-12
            )

            # 余弦相似度
            similarity = np.dot(query_norm, gallery_norm.T)

        elif metric == "euclidean":
            # 欧氏距离（转换为相似度：距离越小相似度越大）
            distances = np.sqrt(
                np.sum(query_features**2, axis=1, keepdims=True)
                + np.sum(gallery_features**2, axis=1, keepdims=True).T
                - 2 * np.dot(query_features, gallery_features.T)
            )
            similarity = 1 / (1 + distances)

        else:
            raise ValueError(f"不支持的度量方式: {metric}")

        return similarity

    @torch.no_grad()
    def rank_gallery(
        self,
        query_features: np.ndarray,
        gallery_features: np.ndarray,
        gallery_ids: Optional[List] = None,
        top_k: int = 10,
        metric: str = "cosine",
    ) -> List[Dict]:
        """
        对gallery进行排序

        Args:
            query_features: 查询特征 [D] 或 [1, D]
            gallery_features: gallery特征 [N, D]
            gallery_ids: gallery ID列表
            top_k: 返回前k个结果
            metric: 距离度量

        Returns:
            排序结果列表
        """
        # 计算相似度
        similarity = self.compute_similarity(query_features, gallery_features, metric)

        # 如果 query 是单个向量
        if similarity.ndim == 2 and similarity.shape[0] == 1:
            similarity = similarity[0]

        # 排序（从高到低）
        sorted_indices = np.argsort(similarity)[::-1][:top_k]

        # 构建结果
        results = []
        for rank, idx in enumerate(sorted_indices, 1):
            result = {
                "rank": rank,
                "index": int(idx),
                "similarity": float(similarity[idx]),
            }
            if gallery_ids is not None:
                result["id"] = gallery_ids[idx]
            results.append(result)

        return results

    def save_features(
        self, features: np.ndarray, save_path: str, metadata: Optional[Dict] = None
    ):
        """
        保存特征到文件

        Args:
            features: 特征数组
            save_path: 保存路径
            metadata: 元数据（可选）
        """
        data = {
            "features": features,
            "model_name": self.model_name,
            "checkpoint": self.checkpoint_path,
        }

        if metadata is not None:
            data["metadata"] = metadata

        # 根据扩展名选择保存格式
        ext = os.path.splitext(save_path)[1].lower()

        if ext == ".npy":
            np.save(save_path, features)
        elif ext == ".npz":
            np.savez(save_path, **data)
        else:
            raise ValueError(f"不支持的文件格式: {ext}，请使用 .npy 或 .npz")

        print(f"✓ 特征已保存到: {save_path}")

    def load_video(
        self, video_path: str, num_frames: Optional[int] = None
    ) -> torch.Tensor:
        """
        从视频文件加载并预处理为模型输入格式

        Args:
            video_path: 视频文件路径
            num_frames: 采样帧数（如果为None，使用配置中的值）

        Returns:
            视频张量 [1, C, T, H, W]
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        # 使用配置中的num_frames
        if num_frames is None:
            num_frames = self.cfg.DATA.NUM_FRAMES

        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"视频信息:")
        print(f"  路径: {video_path}")
        print(f"  总帧数: {total_frames}")
        print(f"  FPS: {fps:.2f}")
        print(f"  采样帧数: {num_frames}")

        if total_frames == 0:
            cap.release()
            raise RuntimeError(f"视频帧数为0: {video_path}")

        # 计算采样索引（均匀采样）
        if total_frames <= num_frames:
            # 如果视频帧数不足，重复采样
            indices = np.linspace(
                0, max(0, total_frames - 1), num_frames, dtype=int
            ).tolist()
        else:
            # 均匀采样
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

        # 读取帧
        frames = []
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                print(f"警告: 无法读取第 {frame_idx} 帧，使用最后一帧")
                if len(frames) > 0:
                    frames.append(frames[-1])
                continue

            # BGR → RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise RuntimeError(f"无法从视频中读取任何帧: {video_path}")

        # 确保有足够的帧
        while len(frames) < num_frames:
            frames.append(frames[-1])

        # 预处理变换（与测试集相同）
        transform = T.Compose(
            [
                T.Resize((self.cfg.DATA.HEIGHT, self.cfg.DATA.WIDTH)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # 应用变换
        processed_frames = []
        for frame in frames[:num_frames]:
            img = Image.fromarray(frame)
            img_tensor = transform(img)  # [C, H, W]
            processed_frames.append(img_tensor)

        # [T, C, H, W] → [C, T, H, W] → [1, C, T, H, W]
        video_tensor = (
            torch.stack(processed_frames, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
        )

        print(f"✓ 视频加载完成，形状: {video_tensor.shape}")

        return video_tensor


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="NBA ReID 推理工具")

    # 模型选择
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["timesformer", "mvit", "uniformerv2"],
        help="选择模型: timesformer, mvit, uniformerv2",
    )

    # 模型权重
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="模型权重文件路径"
    )

    # 配置文件（可选）
    parser.add_argument("--config", type=str, default=None, help="配置文件路径（可选）")

    # 输入数据
    parser.add_argument("--query", type=str, help="查询视频路径或特征文件")

    parser.add_argument("--gallery", type=str, help="gallery视频目录或特征文件")

    # 输出
    parser.add_argument(
        "--output", type=str, default="./inference_output", help="输出目录"
    )

    # 推理参数
    parser.add_argument("--top-k", type=int, default=10, help="返回前k个最相似的结果")

    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean"],
        help="距离度量方式",
    )

    parser.add_argument("--batch-size", type=int, default=8, help="批处理大小")

    # 设备
    parser.add_argument("--device", type=str, default="cuda:0", help="计算设备")

    # 模式
    parser.add_argument(
        "--mode",
        type=str,
        default="search",
        choices=["extract", "search", "dataset"],
        help="运行模式: extract(提取特征), search(查询检索), dataset(数据集评估)",
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    print("=" * 80)
    print("NBA ReID 推理工具")
    print("=" * 80)
    print(f"模型: {args.model}")
    print(f"权重: {args.checkpoint}")
    print(f"模式: {args.mode}")
    print("=" * 80)

    # 初始化推理器
    inferencer = ReIDInference(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
    )

    # 根据模式执行不同操作
    if args.mode == "extract":
        # 提取特征模式
        if not args.query:
            print("错误: extract 模式需要指定 --query 参数")
            return

        print(f"\n提取特征: {args.query}")

        # 加载视频
        try:
            video_tensor = inferencer.load_video(args.query)
        except Exception as e:
            print(f"错误: 无法加载视频 - {e}")
            return

        # 提取特征
        print("\n开始提取特征...")
        try:
            features = inferencer.extract_features(video_tensor)
            print(f"✓ 特征提取成功!")
            print(f"  特征形状: {features.shape}")
            print(f"  特征维度: {features.shape[-1] if features.ndim > 0 else 'N/A'}")

            # 保存特征
            video_name = Path(args.query).stem
            feature_path = os.path.join(args.output, f"{video_name}_features.npz")

            metadata = {
                "video_path": args.query,
                "video_name": video_name,
                "model": args.model,
                "num_frames": inferencer.cfg.DATA.NUM_FRAMES,
            }

            inferencer.save_features(features, feature_path, metadata)

        except Exception as e:
            print(f"错误: 特征提取失败 - {e}")
            import traceback

            traceback.print_exc()
            return

    elif args.mode == "search":
        # 检索模式
        if not args.query or not args.gallery:
            print("错误: search 模式需要指定 --query 和 --gallery 参数")
            return

        print(f"\n执行检索:")
        print(f"  Query: {args.query}")
        print(f"  Gallery: {args.gallery}")
        # TODO: 实现完整的检索流程
        print("提示: 完整的检索功能需要实现视频加载和相似度计算")

    elif args.mode == "dataset":
        # 数据集评估模式
        print("\n数据集评估模式")
        print("提示: 使用 train_test.py 进行完整的数据集评估")

    print("\n" + "=" * 80)
    print("推理完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
