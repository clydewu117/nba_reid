"""
NBA ReID 推理模块

提供统一的接口用于 TimeSformer、MViT 和 UniFormerV2 模型的推理。

主要类:
    ReIDInference - 推理器类，支持特征提取、相似度计算和检索排序

使用示例:
    >>> from inference import ReIDInference
    >>> import torch
    >>>
    >>> inferencer = ReIDInference(
    ...     model_name='timesformer',
    ...     checkpoint_path='outputs/timesformer_app_ft/timesformer_app_model.pth'
    ... )
    >>>
    >>> video = torch.randn(1, 3, 32, 224, 224)
    >>> features = inferencer.extract_features(video)
"""

from .inference import ReIDInference

__version__ = "1.0.0"
__all__ = ["ReIDInference"]
