#!/usr/bin/env python
"""
打印 nba_reid 中三个模型的结构

使用方法:
1. 打印所有三个模型:
   python print_model.py --all

2. 打印特定模型:
   python print_model.py --model timesformer
   python print_model.py --model mvit
   python print_model.py --model uniformer

3. 显示详细参数信息:
   python print_model.py --model timesformer --verbose

4. 保存到文件:
   python print_model.py --all --output model_structures.txt
"""

import os
import sys
import argparse
import torch
import torch.nn as nn

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from config.defaults import get_cfg_defaults
from models.build import build_model
import utils.logging as logging

logger = logging.get_logger(__name__)


def count_parameters(model):
    """统计模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_model_info(model, model_name, verbose=True):
    """打印模型信息"""
    print("\n" + "=" * 80)
    print(f"模型名称: {model_name}")
    print("=" * 80)

    # 统计参数
    total_params, trainable_params = count_parameters(model)
    print(f"\n参数统计:")
    print(f"  总参数数: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"  冻结参数: {total_params - trainable_params:,}")

    # 打印模型结构
    print(f"\n模型结构:")
    print("-" * 80)
    if verbose:
        # 详细模式：打印所有层
        print(model)
    else:
        # 简洁模式：只打印主要模块
        for name, module in model.named_children():
            print(f"\n[{name}]")
            print(f"  类型: {module.__class__.__name__}")
            if hasattr(module, "__len__"):
                print(f"  子模块数量: {len(module)}")
            # 统计该模块的参数
            module_params = sum(p.numel() for p in module.parameters())
            module_trainable = sum(
                p.numel() for p in module.parameters() if p.requires_grad
            )
            print(f"  参数数: {module_params:,} ({module_params/1e6:.2f}M)")
            print(f"  可训练: {module_trainable:,}")

            # 如果有 named_children，打印一级子模块
            children = list(module.named_children())
            if children and len(children) <= 20:  # 限制打印数量
                for child_name, child_module in children[:10]:
                    print(f"    └─ {child_name}: {child_module.__class__.__name__}")
                if len(children) > 10:
                    print(f"    └─ ... (还有 {len(children)-10} 个子模块)")

    print("\n" + "=" * 80)


def get_timesformer_cfg():
    """创建 TimeSformer 配置"""
    cfg = get_cfg_defaults()
    cfg.MODEL.MODEL_NAME = "TimeSformerReID"
    cfg.MODEL.ARCH = "timesformer"
    cfg.MODEL.NUM_CLASSES = 13  # NBA ReID 示例

    # TimeSformer 特定配置
    cfg.TIMESFORMER.PATCH_SIZE = 16
    cfg.TIMESFORMER.ATTENTION_TYPE = "divided_space_time"
    cfg.TIMESFORMER.DROP_PATH_RATE = 0.1
    cfg.TIMESFORMER.EMBED_DIM = 768
    cfg.TIMESFORMER.PRETRAIN = ""  # 不加载预训练权重
    cfg.TIMESFORMER.FROZEN = False

    # 数据配置
    cfg.DATA.NUM_FRAMES = 8
    cfg.DATA.HEIGHT = 224
    cfg.DATA.WIDTH = 224

    # ReID Head 配置
    cfg.REID.NECK_FEAT = "after"
    cfg.REID.EMBED_DIM = 512

    cfg.NUM_GPUS = 0  # CPU 模式

    return cfg


def get_mvit_cfg():
    """创建 MViT 配置"""
    cfg = get_cfg_defaults()
    cfg.MODEL.MODEL_NAME = "MViTReID"
    cfg.MODEL.ARCH = "mvit"
    cfg.MODEL.NUM_CLASSES = 13

    # MViT 特定配置
    cfg.MVIT.PRETRAIN = ""
    cfg.MVIT.FROZEN = False
    cfg.MVIT.USE_MEAN_POOLING = False

    # 数据配置
    cfg.DATA.NUM_FRAMES = 16
    cfg.DATA.HEIGHT = 224
    cfg.DATA.WIDTH = 224

    # ReID Head 配置
    cfg.REID.NECK_FEAT = "after"
    cfg.REID.EMBED_DIM = 512

    cfg.NUM_GPUS = 0

    return cfg


def get_uniformer_cfg():
    """创建 UniFormer 配置"""
    cfg = get_cfg_defaults()
    cfg.MODEL.MODEL_NAME = "Uniformerv2ReID"
    cfg.MODEL.ARCH = "uniformer"
    cfg.MODEL.NUM_CLASSES = 13
    cfg.MODEL.USE_CHECKPOINT = False
    cfg.MODEL.CHECKPOINT_NUM = [0]

    # UniFormer 特定配置
    cfg.UNIFORMERV2.BACKBONE = "uniformerv2_b16"
    cfg.UNIFORMERV2.N_LAYERS = 12
    cfg.UNIFORMERV2.N_DIM = 768
    cfg.UNIFORMERV2.N_HEAD = 12
    cfg.UNIFORMERV2.MLP_FACTOR = 4.0
    cfg.UNIFORMERV2.BACKBONE_DROP_PATH_RATE = 0.0
    cfg.UNIFORMERV2.DROP_PATH_RATE = 0.0
    cfg.UNIFORMERV2.MLP_DROPOUT = [0.5, 0.5, 0.5, 0.5]
    cfg.UNIFORMERV2.CLS_DROPOUT = 0.5
    cfg.UNIFORMERV2.RETURN_LIST = [8, 9, 10, 11]
    cfg.UNIFORMERV2.TEMPORAL_DOWNSAMPLE = True
    cfg.UNIFORMERV2.DW_REDUCTION = 1.5
    cfg.UNIFORMERV2.NO_LMHRA = False
    cfg.UNIFORMERV2.DOUBLE_LMHRA = True
    cfg.UNIFORMERV2.FROZEN = False
    cfg.UNIFORMERV2.PRETRAIN = ""

    # 数据配置
    cfg.DATA.NUM_FRAMES = 8
    cfg.DATA.HEIGHT = 224
    cfg.DATA.WIDTH = 224

    # ReID Head 配置
    cfg.REID.NECK_FEAT = "after"
    cfg.REID.EMBED_DIM = 512

    cfg.NUM_GPUS = 0

    return cfg


def main():
    parser = argparse.ArgumentParser(description="打印 nba_reid 模型结构")
    parser.add_argument(
        "--model",
        type=str,
        choices=["timesformer", "mvit", "uniformer"],
        help="选择要打印的模型 (timesformer/mvit/uniformer)",
    )
    parser.add_argument("--all", action="store_true", help="打印所有三个模型")
    parser.add_argument(
        "--verbose", action="store_true", help="显示详细的模型结构（包括所有层）"
    )
    parser.add_argument("--output", type=str, default=None, help="保存输出到文件")

    args = parser.parse_args()

    if not args.all and not args.model:
        parser.print_help()
        print("\n错误: 请指定 --model 或 --all")
        return

    # 重定向输出到文件
    original_stdout = sys.stdout
    if args.output:
        output_file = open(args.output, "w", encoding="utf-8")
        sys.stdout = output_file

    try:
        models_to_print = []

        if args.all or args.model == "timesformer":
            models_to_print.append(("TimeSformer", get_timesformer_cfg))

        if args.all or args.model == "mvit":
            models_to_print.append(("MViT", get_mvit_cfg))

        if args.all or args.model == "uniformer":
            models_to_print.append(("UniFormerV2", get_uniformer_cfg))

        for model_name, cfg_func in models_to_print:
            print(f"\n正在构建 {model_name} 模型...")
            try:
                cfg = cfg_func()
                model = build_model(cfg, ema=False)  # 不使用 EMA，直接返回模型
                model.eval()  # 设置为评估模式
                print_model_info(model, model_name, verbose=args.verbose)
            except Exception as e:
                print(f"\n构建 {model_name} 失败: {e}")
                import traceback

                traceback.print_exc()

        print("\n完成!")

    finally:
        # 恢复标准输出
        sys.stdout = original_stdout
        if args.output:
            output_file.close()
            print(f"\n输出已保存到: {args.output}")


if __name__ == "__main__":
    main()
