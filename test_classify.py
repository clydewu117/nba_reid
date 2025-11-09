#!/usr/bin/env python
"""
Basketball Video ReID - Classification Performance Testing Script
测试模型checkpoint的分类表现
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import project modules
from config.defaults import get_cfg_defaults
from data.dataloader import build_dataloader
from models.build import build_model


class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def test_classification(cfg, model, test_loader, device, class_names=None):
    """
    测试模型的分类性能
    
    Args:
        cfg: 配置对象
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        class_names: 类别名称列表
    
    Returns:
        results: 包含各种指标的字典
    """
    # 设置为训练模式以获取分类输出，但禁用梯度和dropout
    model.train()
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            module.eval()
    
    # 初始化指标
    top1_acc = AverageMeter()
    top5_acc = AverageMeter()
    losses = AverageMeter()
    
    # 存储所有预测和真实标签
    all_preds = []
    all_labels = []
    all_probs = []
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n{'='*80}")
    print(f"开始测试分类性能...")
    print(f"测试样本数: {len(test_loader.dataset)}")
    print(f"类别数: {cfg.MODEL.NUM_CLASSES}")
    print(f"{'='*80}\n")
    
    # 遍历测试数据
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
        videos = batch["video"].to(device)
        labels = batch["pid"].to(device)
        
        with autocast():
            # 前向传播 - 训练模式会返回dict
            outputs = model(videos, label=labels)
            
            # 获取分类分数
            if isinstance(outputs, dict):
                logits = outputs["cls_score"]
            else:
                logits = outputs
            
            # 计算损失
            loss = criterion(logits, labels)
            
            # 计算概率
            probs = torch.softmax(logits, dim=1)
            
            # Top-1 准确率
            _, pred = logits.topk(1, 1, True, True)
            pred = pred.t()
            correct = pred.eq(labels.view(1, -1).expand_as(pred))
            top1_correct = correct[:1].reshape(-1).float().sum(0)
            top1_acc.update(top1_correct.item() * 100.0 / labels.size(0), labels.size(0))
            
            # Top-5 准确率
            k = min(5, logits.size(1))
            _, pred_top5 = logits.topk(k, 1, True, True)
            pred_top5 = pred_top5.t()
            correct_top5 = pred_top5.eq(labels.view(1, -1).expand_as(pred_top5))
            top5_correct = correct_top5[:k].reshape(-1).float().sum(0)
            top5_acc.update(top5_correct.item() * 100.0 / labels.size(0), labels.size(0))
            
            # 更新损失
            losses.update(loss.item(), labels.size(0))
            
            # 保存预测和标签
            all_preds.extend(pred.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 计算详细指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    # 计算每个类别的指标
    per_class_precision, per_class_recall, per_class_f1, per_class_support = \
        precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
    
    # 混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # 打印结果
    print(f"\n{'='*80}")
    print(f"测试结果:")
    print(f"{'='*80}")
    print(f"  Loss        : {losses.avg:.4f}")
    print(f"  Top-1 Acc   : {top1_acc.avg:.2f}%")
    print(f"  Top-5 Acc   : {top5_acc.avg:.2f}%")
    print(f"  Accuracy    : {accuracy*100:.2f}%")
    print(f"  Precision   : {precision*100:.2f}%")
    print(f"  Recall      : {recall*100:.2f}%")
    print(f"  F1-Score    : {f1*100:.2f}%")
    print(f"{'='*80}\n")
    
    # 不打印每个类别的详细指标（已保存到JSON文件中）
    # if class_names is not None and len(class_names) == cfg.MODEL.NUM_CLASSES:
    #     print(f"\n{'='*80}")
    #     print(f"每个类别的详细指标:")
    #     print(f"{'='*80}")
    #     print(f"{'Class':<30} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    #     print(f"{'-'*80}")
    #     
    #     for i in range(cfg.MODEL.NUM_CLASSES):
    #         class_name = class_names[i] if i < len(class_names) else f"Class_{i}"
    #         print(f"{class_name:<30} {per_class_precision[i]*100:>9.2f}% "
    #               f"{per_class_recall[i]*100:>9.2f}% {per_class_f1[i]*100:>9.2f}% "
    #               f"{per_class_support[i]:>10}")
    #     print(f"{'='*80}\n")
    
    # 构建结果字典
    results = {
        'loss': losses.avg,
        'top1_accuracy': top1_acc.avg,
        'top5_accuracy': top5_acc.avg,
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1 * 100,
        'confusion_matrix': conf_matrix.tolist(),
        'per_class_metrics': {
            'precision': per_class_precision.tolist(),
            'recall': per_class_recall.tolist(),
            'f1_score': per_class_f1.tolist(),
            'support': per_class_support.tolist()
        },
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist(),
        'probabilities': all_probs.tolist()
    }
    
    return results


def plot_confusion_matrix(conf_matrix, class_names, save_path):
    """
    绘制混淆矩阵
    
    Args:
        conf_matrix: 混淆矩阵
        class_names: 类别名称
        save_path: 保存路径
    """
    plt.figure(figsize=(max(10, len(class_names)//2), max(8, len(class_names)//2)))
    
    # 归一化混淆矩阵
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    # 只显示前50个类别（如果类别太多）
    if len(class_names) > 50:
        print(f"类别数量较多 ({len(class_names)})，只显示混淆矩阵的数值统计")
        return
    
    sns.heatmap(conf_matrix_norm, annot=False, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'})
    
    plt.title('Confusion Matrix (Normalized)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存到: {save_path}")
    plt.close()


def plot_per_class_metrics(per_class_metrics, class_names, save_path):
    """
    绘制每个类别的指标柱状图
    
    Args:
        per_class_metrics: 每个类别的指标字典
        class_names: 类别名称
        save_path: 保存路径
    """
    # 只显示前30个类别（如果类别太多）
    num_classes = len(class_names)
    if num_classes > 30:
        print(f"类别数量较多 ({num_classes})，只显示前30个类别的指标")
        num_classes = 30
        class_names = class_names[:30]
        per_class_metrics = {
            k: v[:30] for k, v in per_class_metrics.items()
        }
    
    fig, ax = plt.subplots(figsize=(max(12, num_classes//2), 6))
    
    x = np.arange(num_classes)
    width = 0.25
    
    precision = np.array(per_class_metrics['precision']) * 100
    recall = np.array(per_class_metrics['recall']) * 100
    f1 = np.array(per_class_metrics['f1_score']) * 100
    
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_ylabel('Score (%)')
    ax.set_title('Per-Class Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=90, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"类别指标图已保存到: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="测试模型checkpoint的分类表现")
    parser.add_argument(
        "--config-file",
        type=str,
        default="",
        help="配置文件路径"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型checkpoint路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_results",
        help="测试结果输出目录"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="批次大小 (默认使用配置文件中的值)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="是否绘制可视化图表"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="修改配置的其他选项"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    cfg = get_cfg_defaults()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    
    # 覆盖批次大小（如果指定）
    if args.batch_size is not None:
        cfg.defrost()
        cfg.TEST.BATCH_SIZE = args.batch_size
        cfg.freeze()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(f"cuda:{cfg.GPU_IDS[0]}" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.SEED)
    
    print(f"\n{'='*80}")
    print(f"配置信息:")
    print(f"{'='*80}")
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Output Dir  : {args.output_dir}")
    print(f"  Device      : {device}")
    print(f"  Batch Size  : {cfg.TEST.BATCH_SIZE}")
    print(f"  Model Arch  : {cfg.MODEL.ARCH}")
    print(f"  Video Type  : {cfg.DATA.VIDEO_TYPE}")
    print(f"  Shot Type   : {cfg.DATA.SHOT_TYPE}")
    print(f"  Num Frames  : {cfg.DATA.NUM_FRAMES}")
    print(f"{'='*80}\n")
    
    # 构建数据加载器
    print("加载测试数据...")
    test_loader, num_classes = build_dataloader(cfg, is_train=False)
    
    # 更新配置中的类别数
    cfg.defrost()
    cfg.MODEL.NUM_CLASSES = num_classes
    cfg.freeze()
    
    print(f"类别数: {num_classes}")
    print(f"测试样本数: {len(test_loader.dataset)}")
    
    # 获取类别名称
    class_names = test_loader.dataset.pid_list if hasattr(test_loader.dataset, 'pid_list') else None
    
    # 构建模型
    print("\n构建模型...")
    model = build_model(cfg).to(device)
    
    # 加载checkpoint
    print(f"加载checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # 处理不同的checkpoint格式
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Checkpoint信息:")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'rank1' in checkpoint:
            print(f"  Rank-1: {checkpoint['rank1']:.2%}")
        if 'mAP' in checkpoint:
            print(f"  mAP: {checkpoint['mAP']:.2%}")
    else:
        state_dict = checkpoint
    
    # 加载模型权重
    model.load_state_dict(state_dict, strict=False)
    print("模型权重加载完成")
    
    # 测试分类性能
    results = test_classification(cfg, model, test_loader, device, class_names)
    
    # 保存结果到JSON
    results_file = os.path.join(args.output_dir, "classification_results.json")
    with open(results_file, 'w') as f:
        # 只保存关键指标，移除混淆矩阵、预测结果等大数组
        results_to_save = {k: v for k, v in results.items() 
                          if k not in ['predictions', 'labels', 'probabilities', 'confusion_matrix']}
        json.dump(results_to_save, f, indent=2)
    print(f"\n结果已保存到: {results_file}")
    
    # 绘制可视化图表
    if args.plot and class_names is not None:
        print("\n生成可视化图表...")
        
        # 混淆矩阵
        conf_matrix_path = os.path.join(args.output_dir, "confusion_matrix.png")
        plot_confusion_matrix(
            np.array(results['confusion_matrix']),
            class_names,
            conf_matrix_path
        )
        
        # 每个类别的指标
        metrics_path = os.path.join(args.output_dir, "per_class_metrics.png")
        plot_per_class_metrics(
            results['per_class_metrics'],
            class_names,
            metrics_path
        )
    
    print(f"\n{'='*80}")
    print(f"测试完成！")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
