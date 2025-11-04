#!/usr/bin/env python
"""
Configuration for Basketball Video ReID
"""

from fvcore.common.config import CfgNode

_C = CfgNode()

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()
_C.DATA.ROOT = "/fs/scratch/PAS3184/v2"
_C.DATA.VIDEO_TYPE = "mask"  # 'appearance' or 'mask'
_C.DATA.SHOT_TYPE = "both"  # 'freethrow', '3pt', or 'both'
_C.DATA.NUM_FRAMES = 32  # 每个视频采样的帧数
_C.DATA.FRAME_STRIDE = 4  # 帧采样间隔
_C.DATA.HEIGHT = 224
_C.DATA.WIDTH = 224
_C.DATA.BATCH_SIZE = 8
_C.DATA.NUM_WORKERS = 8
_C.DATA.TRAIN_RATIO = 0.75  # 训练集比例 75%

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()
_C.MODEL.NAME = "Uniformerv2ReID"
_C.MODEL.MODEL_NAME = _C.MODEL.NAME
_C.MODEL.NUM_CLASSES = 0  # 自动从数据集获取
_C.MODEL.USE_CHECKPOINT = True
_C.MODEL.CHECKPOINT_NUM = [0, 0, 8, 0]
_C.MODEL.ARCH = "uniformerv2"  # options: 'uniformerv2', 'timesformer', 'mvit'
# -----------------------------------------------------------------------------
# UniFormerV2
# -----------------------------------------------------------------------------
_C.UNIFORMERV2 = CfgNode()
_C.UNIFORMERV2.BACKBONE = "uniformerv2_b16"
_C.UNIFORMERV2.PRETRAIN = ""  # set your pretrained path explicitly if needed
_C.UNIFORMERV2.FROZEN = True
_C.UNIFORMERV2.N_LAYERS = 12
_C.UNIFORMERV2.N_DIM = 768
_C.UNIFORMERV2.N_HEAD = 12
_C.UNIFORMERV2.MLP_FACTOR = 4.0
_C.UNIFORMERV2.BACKBONE_DROP_PATH_RATE = 0.0
_C.UNIFORMERV2.DROP_PATH_RATE = 0.0
_C.UNIFORMERV2.MLP_DROPOUT = [0.5] * 12
_C.UNIFORMERV2.CLS_DROPOUT = 0.5
_C.UNIFORMERV2.RETURN_LIST = [8, 9, 10, 11]
_C.UNIFORMERV2.TEMPORAL_DOWNSAMPLE = True
_C.UNIFORMERV2.DW_REDUCTION = 1.5
_C.UNIFORMERV2.NO_LMHRA = False
_C.UNIFORMERV2.DOUBLE_LMHRA = True

# -----------------------------------------------------------------------------
# TimeSformer
# -----------------------------------------------------------------------------
_C.TIMESFORMER = CfgNode()
_C.TIMESFORMER.REPO_PATH = ""  # optional absolute path to local TimeSformer repo
_C.TIMESFORMER.PRETRAIN = "/users/PAS2099/clydewu117/nba_reid/checkpoints/timesformer/TimeSformer_divST_8x32_224_K400.pyth"  # optional pretrained checkpoint (.pyth / .pth)
_C.TIMESFORMER.ATTENTION_TYPE = (
    "divided_space_time"  # 'divided_space_time' | 'space_only' | 'joint_space_time'
)
_C.TIMESFORMER.PATCH_SIZE = 16
_C.TIMESFORMER.DROP_PATH_RATE = 0.1
_C.TIMESFORMER.EMBED_DIM = 768
_C.TIMESFORMER.FROZEN = False
_C.TIMESFORMER.BACKBONE_IMPL = "official"  # force official implementation; local lite impl is disabled

# -----------------------------------------------------------------------------
# MViT (SlowFast) - minimal knobs for ReID backbone
# -----------------------------------------------------------------------------
_C.MVIT = CfgNode()
_C.MVIT.PRETRAIN = "/users/PAS2099/clydewu117/nba_reid/checkpoints/mvitv2/MViTv2_S_16x4_k400_f302660347.pyth"  # optional SlowFast checkpoint path
_C.MVIT.FROZEN = False  # freeze backbone
_C.MVIT.USE_MEAN_POOLING = False  # use mean of patch tokens instead of CLS

# Default MViTv2-S (16x4, 224) backbone config to match SlowFast checkpoints and avoid OOM
# Reference: SlowFast configs/Kinetics/MVITv2_S_16x4.yaml
_C.MVIT.MODE = "conv"
_C.MVIT.CLS_EMBED_ON = True
_C.MVIT.PATCH_KERNEL = [3, 7, 7]
_C.MVIT.PATCH_STRIDE = [2, 4, 4]
_C.MVIT.PATCH_PADDING = [1, 3, 3]
_C.MVIT.EMBED_DIM = 96
_C.MVIT.NUM_HEADS = 1
_C.MVIT.MLP_RATIO = 4.0
_C.MVIT.QKV_BIAS = True
_C.MVIT.DROPPATH_RATE = 0.2
_C.MVIT.DEPTH = 16
_C.MVIT.NORM = "layernorm"
_C.MVIT.USE_ABS_POS = False
_C.MVIT.REL_POS_SPATIAL = True
_C.MVIT.REL_POS_TEMPORAL = True
_C.MVIT.SEP_POS_EMBED = False
_C.MVIT.USE_FIXED_SINCOS_POS = False
_C.MVIT.DIM_MUL_IN_ATT = True
_C.MVIT.RESIDUAL_POOLING = True

# Channel/head multipliers across depth to reach 768-dim at final blocks (match pretrained)
_C.MVIT.DIM_MUL = [[1, 2.0], [3, 2.0], [14, 2.0]]
_C.MVIT.HEAD_MUL = [[1, 2.0], [3, 2.0], [14, 2.0]]

# Token pooling schedule (critical to control memory and match checkpoint)
_C.MVIT.POOL_KVQ_KERNEL = [3, 3, 3]
_C.MVIT.POOL_KV_STRIDE_ADAPTIVE = [1, 8, 8]
_C.MVIT.POOL_Q_STRIDE = [
    [0, 1, 1, 1],
    [1, 1, 2, 2],
    [2, 1, 1, 1],
    [3, 1, 2, 2],
    [4, 1, 1, 1],
    [5, 1, 1, 1],
    [6, 1, 1, 1],
    [7, 1, 1, 1],
    [8, 1, 1, 1],
    [9, 1, 1, 1],
    [10, 1, 1, 1],
    [11, 1, 1, 1],
    [12, 1, 1, 1],
    [13, 1, 1, 1],
    [14, 1, 2, 2],
    [15, 1, 1, 1],
]

# -----------------------------------------------------------------------------
# ReID
# -----------------------------------------------------------------------------
_C.REID = CfgNode()
_C.REID.NECK_FEAT = "after"  # 'after' or 'before'
_C.REID.EMBED_DIM = 512  # Embedding dimension

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CfgNode()
_C.SOLVER.OPTIMIZER = "AdamW"
_C.SOLVER.BASE_LR = 0.0001
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.MAX_EPOCHS = 100
_C.SOLVER.WARMUP_EPOCHS = 0
_C.SOLVER.STEPS = [40, 70]
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.CHECKPOINT_PERIOD = 10
_C.SOLVER.EVAL_PERIOD = 10
_C.SOLVER.LOG_PERIOD = 1
_C.SOLVER.COSINE_END_LR = 1e-6
_C.SOLVER.COSINE_AFTER_WARMUP = True
# Gradient clipping
_C.SOLVER.CLIP_GRADIENTS = CfgNode()
_C.SOLVER.CLIP_GRADIENTS.ENABLED = True
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0

# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------
_C.LOSS = CfgNode()
_C.LOSS.USE_LABEL_SMOOTH = True
_C.LOSS.LABEL_SMOOTH_EPSILON = 0.1
_C.LOSS.USE_TRIPLET = True
_C.LOSS.TRIPLET_MARGIN = 0.3
_C.LOSS.TRIPLET_DISTANCE = "euclidean"
_C.LOSS.ID_WEIGHT = 1.0
_C.LOSS.TRIPLET_WEIGHT = 1.0

# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------
_C.TEST = CfgNode()
_C.TEST.BATCH_SIZE = 16
_C.TEST.WEIGHT = ""

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.OUTPUT_DIR = "./outputs/basketball_reid"
_C.SEED = 42
_C.GPU_IDS = [0]
_C.NUM_GPUS = 1


def get_cfg_defaults():
    """Get default config"""
    return _C.clone()


# -----------------------------------------------------------------------------
# 预定义配置
# -----------------------------------------------------------------------------


def get_appearance_freethrow_config():
    """Appearance + Freethrow only"""
    cfg = get_cfg_defaults()
    cfg.DATA.VIDEO_TYPE = "appearance"
    cfg.DATA.SHOT_TYPE = "freethrow"
    cfg.OUTPUT_DIR = "./outputs/appearance_freethrow"
    return cfg


def get_appearance_3pt_config():
    """Appearance + 3pt only"""
    cfg = get_cfg_defaults()
    cfg.DATA.VIDEO_TYPE = "appearance"
    cfg.DATA.SHOT_TYPE = "3pt"
    cfg.OUTPUT_DIR = "./outputs/appearance_3pt"
    return cfg


def get_appearance_both_config():
    """Appearance + Both shot types"""
    cfg = get_cfg_defaults()
    cfg.DATA.VIDEO_TYPE = "appearance"
    cfg.DATA.SHOT_TYPE = "both"
    cfg.OUTPUT_DIR = "./outputs/appearance_both"
    return cfg


def get_mask_freethrow_config():
    """Mask + Freethrow only"""
    cfg = get_cfg_defaults()
    cfg.DATA.VIDEO_TYPE = "mask"
    cfg.DATA.SHOT_TYPE = "freethrow"
    cfg.OUTPUT_DIR = "./outputs/mask_freethrow"
    return cfg


def get_mask_3pt_config():
    """Mask + 3pt only"""
    cfg = get_cfg_defaults()
    cfg.DATA.VIDEO_TYPE = "mask"
    cfg.DATA.SHOT_TYPE = "3pt"
    cfg.OUTPUT_DIR = "./outputs/mask_3pt"
    return cfg


def get_mask_both_config():
    """Mask + Both shot types"""
    cfg = get_cfg_defaults()
    cfg.DATA.VIDEO_TYPE = "mask"
    cfg.DATA.SHOT_TYPE = "both"
    cfg.OUTPUT_DIR = "./outputs/mask_both"
    return cfg


if __name__ == "__main__":
    # 测试不同配置
    configs = [
        ("Appearance + Freethrow", get_appearance_freethrow_config()),
        ("Appearance + 3pt", get_appearance_3pt_config()),
        ("Appearance + Both", get_appearance_both_config()),
        ("Mask + Freethrow", get_mask_freethrow_config()),
        ("Mask + 3pt", get_mask_3pt_config()),
        ("Mask + Both", get_mask_both_config()),
    ]

    for name, cfg in configs:
        print(f"\n{'='*60}")
        print(f"{name}")
        print(f"{'='*60}")
        print(f"Video Type: {cfg.DATA.VIDEO_TYPE}")
        print(f"Shot Type: {cfg.DATA.SHOT_TYPE}")
        print(f"Output Dir: {cfg.OUTPUT_DIR}")
