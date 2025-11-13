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
from contextlib import nullcontext
import numpy as np
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib
# Use a non-interactive backend for headless environments (e.g., SSH, servers)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Any, cast
from torch.utils.data import DataLoader as TorchDataLoader, Subset

# Prefer explicit CUDA autocast import when available; fallback to no-op
try:
    from torch.cuda.amp import autocast as cuda_autocast  # type: ignore
except Exception:
    cuda_autocast = nullcontext

# Import project modules
from config.defaults import get_cfg_defaults
from data.dataloader import build_dataloader, collate_fn
from models.build import build_model
from metrics import R1_mAP_eval


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
    # 在支持 cudnn 的情况下再设置，以避免 CPU-only 环境出错
    try:
        import torch.backends.cudnn as cudnn  # type: ignore
        cudnn.deterministic = True
        cudnn.benchmark = False
    except Exception:
        pass


def _load_training_label_map(map_path: str) -> List[str]:
    """
    读取训练时的类别顺序映射。
    支持三种格式：
      1) 纯列表 ["name1", "name2", ...]
      2) 对象 {"pid_list": [...]} 取其中的列表
      3) 字典 {"name": idx, ...} 将按 idx 排序得到列表
    返回：训练时的 pid_list（按训练顺序排列的类名列表）
    """
    with open(map_path, 'r') as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [str(x) for x in obj]
    if isinstance(obj, dict):
        if 'pid_list' in obj and isinstance(obj['pid_list'], list):
            return [str(x) for x in obj['pid_list']]
        # 视作 name->idx 字典
        try:
            items = sorted([(int(v), str(k)) for k, v in obj.items()], key=lambda x: x[0])
            return [name for _, name in items]
        except Exception as e:
            raise ValueError(f"Unrecognized label-map dict format: {e}")
    raise ValueError("Unsupported label-map format; must be list or dict")


def apply_label_map_to_dataset(dataset, training_pid_list: List[str], debug: bool = False) -> int:
    """
    将当前数据集的 pid 映射到训练时期的类别顺序。
    - dataset.data[*]['identity'] 是类名（形如 'First Last'）。
    - 根据 training_pid_list 构造 name->index 映射，重写每个样本的 pid。
    - 同时用训练顺序覆盖 dataset.pid_list。
    返回：映射成功的样本数。
    """
    name_to_idx = {name: i for i, name in enumerate(training_pid_list)}

    n_total = len(dataset.data)
    n_mapped = 0
    missing_names = set()
    for item in dataset.data:
        name = item.get('identity')
        if name in name_to_idx:
            item['pid'] = name_to_idx[name]
            n_mapped += 1
        else:
            missing_names.add(name)

    # 覆盖 pid_list 为训练顺序（便于打印/可视化）
    dataset.pid_list = training_pid_list

    if debug:
        print(f"[DEBUG] Label-map applied: mapped {n_mapped}/{n_total} samples.")
        if missing_names:
            show = list(missing_names)
            print(f"[DEBUG] Missing {len(missing_names)} identities not found in label-map (show up to 20): {show[:20]}")

    # 更新 dataset.num_classes
    try:
        dataset.num_classes = len(training_pid_list)
    except Exception:
        pass

    return n_mapped


def split_query_gallery(dataset, query_ratio: float = 0.5, seed: int = 42, min_samples: int = 2) -> Tuple[List[int], List[int]]:
    """
    将测试集拆分为 query / gallery（参考训练阶段的实现）。
    - 对每个 pid，随机按比例切分，确保每个 pid 至少各有 1 个样本进入 query/gallery。
    - 过滤掉样本数不足的 pid（< min_samples）。
    返回：query_indices, gallery_indices
    """
    np.random.seed(seed)

    pid_to_indices = {}
    for idx in range(len(dataset)):
        item = dataset.data[idx]  # 直接访问内部 data 列表
        pid = item['pid']
        pid_to_indices.setdefault(pid, []).append(idx)

    query_indices: List[int] = []
    gallery_indices: List[int] = []
    excluded_pids = []

    for pid, indices in pid_to_indices.items():
        if len(indices) < min_samples:
            excluded_pids.append(pid)
            continue

        np.random.shuffle(indices)
        split_point = max(1, int(len(indices) * query_ratio))
        split_point = min(split_point, len(indices) - 1)  # gallery 至少 1 个

        query_indices.extend(indices[:split_point])
        gallery_indices.extend(indices[split_point:])

    print(f"\n{'='*60}")
    print(f"Query/Gallery Split Summary (for ReID metrics):")
    print(f"  Query samples    : {len(query_indices)}")
    print(f"  Gallery samples  : {len(gallery_indices)}")
    print(f"  Valid PIDs       : {len(pid_to_indices) - len(excluded_pids)}")
    print(f"  Excluded PIDs    : {len(excluded_pids)} (< {min_samples} samples)")
    print(f"{'='*60}")

    # 验证没有重叠
    overlap = set(query_indices) & set(gallery_indices)
    assert len(overlap) == 0, f"Error: {len(overlap)} samples overlap between query and gallery!"

    return query_indices, gallery_indices


def find_presplit_query_gallery_indices(dataset) -> Tuple[List[int], List[int]]:
    """
    在已加载的数据集中，按视频路径中是否包含 'query' / 'gallery' 目录来提取索引。
    若找不到任何一侧，则返回空列表，表示不可用。
    """
    q_idx: List[int] = []
    g_idx: List[int] = []

    for idx in range(len(dataset)):
        item = dataset.data[idx]
        vpath = item.get('video_path', '')
        parts = set(os.path.normpath(vpath).split(os.sep))
        if 'query' in parts:
            q_idx.append(idx)
        elif 'gallery' in parts:
            g_idx.append(idx)

    return q_idx, g_idx


@torch.no_grad()
def compute_reid_metrics(cfg, model, dataset, device, rank_ks: List[int] = [1, 5, 10]):
    """
    计算 ReID 指标（Rank-k 和 mAP），参考训练阶段的流程：
    - 将测试集拆成 query / gallery
    - 提取特征，使用 R1_mAP_eval 计算 CMC 和 mAP
    返回：{'mAP': float, 'cmc': List[float], 'ranks': {k: float}}
    """
    # 优先：若启用 USE_PRESPLIT，尝试使用路径中自带的 query/gallery 划分
    q_idx: List[int] = []
    g_idx: List[int] = []
    used_mode = 'random-split'
    if getattr(cfg.DATA, 'USE_PRESPLIT', False):
        q_idx, g_idx = find_presplit_query_gallery_indices(dataset)
        if len(q_idx) > 0 and len(g_idx) > 0:
            used_mode = 'presplit'
        else:
            # Fallback: 无预分的 query/gallery，则走随机拆分
            q_idx, g_idx = split_query_gallery(dataset, query_ratio=0.5, seed=cfg.SEED)
            used_mode = 'fallback-random-split'
    else:
        # 默认使用随机拆分
        q_idx, g_idx = split_query_gallery(dataset, query_ratio=0.5, seed=cfg.SEED)
        used_mode = 'random-split'

    # 若无法形成有效的 query / gallery，直接返回不可用
    if len(q_idx) == 0 or len(g_idx) == 0:
        print("\n[ReID] Skip metrics: empty query/gallery after split."
              f" (num_query={len(q_idx)}, num_gallery={len(g_idx)})")
        return {
            'available': False,
            'mAP': 0.0,
            'cmc': [],
            'ranks': {},
            'num_query': len(q_idx),
            'num_gallery': len(g_idx),
        }

    # 构建对应的 DataLoader
    q_loader = TorchDataLoader(
        Subset(dataset, q_idx),
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATA.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    g_loader = TorchDataLoader(
        Subset(dataset, g_idx),
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATA.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    num_query = len(cast(Any, q_loader).dataset)
    evaluator = R1_mAP_eval(num_query, max_rank=max(rank_ks), feat_norm=True)

    # 提取特征（测试模式）
    model.eval()
    print(f"\nExtracting query features for ReID metrics ({num_query})... [{used_mode}]")
    num_q_feats = 0
    for batch in q_loader:
        videos = batch['video'].to(device)
        pids = batch['pid'].to(device)
        with (cuda_autocast() if device.type == 'cuda' else nullcontext()):
            feat = model(videos)  # 测试模式：返回 [B, D] 特征
        evaluator.update((feat, pids))
        num_q_feats += videos.size(0)

    num_gallery = len(cast(Any, g_loader).dataset)
    print(f"Extracting gallery features for ReID metrics ({num_gallery})...")
    num_g_feats = 0
    for batch in g_loader:
        videos = batch['video'].to(device)
        pids = batch['pid'].to(device)
        with (cuda_autocast() if device.type == 'cuda' else nullcontext()):
            feat = model(videos)
        evaluator.update((feat, pids))
        num_g_feats += videos.size(0)

    # 若未产生任何特征，返回不可用以避免 evaluator 内部拼接空张量
    if num_q_feats == 0 or num_g_feats == 0:
        print(f"[ReID] Skip metrics: no features extracted (q={num_q_feats}, g={num_g_feats}).")
        return {
            'available': False,
            'mAP': 0.0,
            'cmc': [],
            'ranks': {},
            'num_query': num_query,
            'num_gallery': num_gallery,
        }

    cmc, mAP = evaluator.compute()

    # 组装 rank-k
    ranks = {}
    for k in rank_ks:
        idx = min(k - 1, len(cmc) - 1)
        ranks[f'rank{k}'] = float(cmc[idx])

    print(f"\n{'='*60}")
    print("ReID Metrics:")
    print(f"  mAP     : {mAP:.2%}")
    for k in rank_ks:
        print(f"  Rank-{k} : {ranks[f'rank{k}']:.2%}")
    print(f"{'='*60}\n")

    return {
        'available': True,
        'mAP': float(mAP),
        'cmc': [float(x) for x in cmc],
        'ranks': ranks,
        'num_query': num_query,
        'num_gallery': num_gallery,
    }


@torch.no_grad()
def test_classification(cfg, model, test_loader, device, class_names=None, debug: bool = False):
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
        
        # Enable AMP on CUDA only; on CPU fall back to a no-op context
        with (cuda_autocast() if device.type == 'cuda' else nullcontext()):
            # 前向传播 - 训练模式会返回dict
            outputs = model(videos, label=labels)
            
            # 获取分类分数
            if isinstance(outputs, dict):
                # Be tolerant to different key names from various heads
                logits = outputs.get("cls_score", outputs.get("logits"))
                if logits is None:
                    raise KeyError("Model output dict must contain 'cls_score' or 'logits'.")
            else:
                logits = outputs
            
            # Debug: 打印输出结构与logits统计（仅首个batch）
            if debug and batch_idx == 0:
                if isinstance(outputs, dict):
                    print(f"[DEBUG] model(videos) returned keys: {list(outputs.keys())}")
                else:
                    print(f"[DEBUG] model(videos) returned type: {type(outputs)}")

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

            # Debug: 打印若干样本的 GT 类名 vs 预测类名（Top-1/Top-5）
            if debug and batch_idx == 0 and class_names is not None and len(class_names) == int(cfg.MODEL.NUM_CLASSES):
                try:
                    max_show = min(8, labels.size(0))
                    for i in range(max_show):
                        gt_idx = int(labels[i].item())
                        gt_name = class_names[gt_idx] if 0 <= gt_idx < len(class_names) else f"Class_{gt_idx}"

                        top1_idx = int(pred[:, i].item())
                        top1_name = class_names[top1_idx] if 0 <= top1_idx < len(class_names) else f"Class_{top1_idx}"
                        top1_prob = float(probs[i, top1_idx].item())

                        top5_idxs = [int(x) for x in pred_top5[:, i].cpu().numpy().tolist()]
                        top5_names = [class_names[j] if 0 <= j < len(class_names) else f"Class_{j}" for j in top5_idxs]
                        top5_probs = [float(probs[i, j].item()) for j in top5_idxs]

                        top5_str = ", ".join([f"{n}({j}, {p:.2f})" for n, j, p in zip(top5_names, top5_idxs, top5_probs)])
                        print(f"[DEBUG] Sample {i}: GT={gt_name}({gt_idx}) | Pred@1={top1_name}({top1_idx}, {top1_prob:.2f}) | Top5=[{top5_str}]")
                except Exception as e:
                    print(f"[DEBUG] Print GT vs Pred failed: {e}")
            
            # 更新损失
            losses.update(loss.item(), labels.size(0))
            
            # Debug: 打印logits/softmax统计（仅首个batch）
            if debug and batch_idx == 0:
                with torch.no_grad():
                    sm = probs
                    top1_val, _ = sm.max(dim=1)
                    entropy = -(sm * (sm.clamp_min(1e-12)).log()).sum(dim=1)
                    print(f"[DEBUG] logits.shape: {tuple(logits.shape)}")
                    print(f"[DEBUG] logits mean/std: {logits.mean().item():.4f} / {logits.std().item():.4f}")
                    print(f"[DEBUG] softmax top1 conf mean: {top1_val.mean().item():.4f}")
                    print(f"[DEBUG] softmax entropy mean: {entropy.mean().item():.4f}")

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
            'precision': np.asarray(per_class_precision).tolist(),
            'recall': np.asarray(per_class_recall).tolist(),
            'f1_score': np.asarray(per_class_f1).tolist(),
            'support': np.asarray(per_class_support).tolist()
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
    
    # 归一化混淆矩阵，避免除零
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    conf_matrix_norm = np.divide(conf_matrix.astype('float'), row_sums, where=row_sums!=0)
    conf_matrix_norm = np.nan_to_num(conf_matrix_norm)
    
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
    ax.set_ylim(0, 105)
    
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
        "--debug",
        action="store_true",
        help="打印调试信息（logits/softmax统计、缺失权重等，仅首个batch）"
    )
    parser.add_argument(
        "--rank-ks",
        type=str,
        default="1,5,10",
        help="要报告的Rank-k列表，逗号分隔，如: 1,5,10"
    )
    parser.add_argument(
        "--label-map",
        type=str,
        default=None,
        help="训练时的类别顺序映射文件（json）：list / {pid_list:[...]} / {name: idx}"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        choices=["mvit", "uniformerv2", "timesformer", "videomaev2"],
        help="显式指定要测试的骨干/架构；mvit 对应 MViTReID"
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

    # 显式选择架构，便于只评测 MViT
    if args.arch is not None:
        cfg.defrost()
        if args.arch == "mvit":
            cfg.MODEL.NAME = "MViTReID"
            cfg.MODEL.MODEL_NAME = "MViTReID"
            cfg.MODEL.ARCH = "mvit"
        elif args.arch == "uniformerv2":
            cfg.MODEL.NAME = "Uniformerv2ReID"
            cfg.MODEL.MODEL_NAME = "Uniformerv2ReID"
            cfg.MODEL.ARCH = "uniformerv2"
        elif args.arch == "timesformer":
            cfg.MODEL.NAME = "TimeSformerReID"
            cfg.MODEL.MODEL_NAME = "TimeSformerReID"
            cfg.MODEL.ARCH = "timesformer"
        elif args.arch == "videomaev2":
            cfg.MODEL.NAME = "VideoMAEv2ReID"
            cfg.MODEL.MODEL_NAME = "VideoMAEv2ReID"
            cfg.MODEL.ARCH = "videomaev2"
        cfg.freeze()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(f"cuda:{cfg.GPU_IDS[0]}" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        cfg.defrost()
        cfg.NUM_GPUS = 0
        cfg.freeze()
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

    # 如果指定 label-map，则将数据集标签顺序对齐到训练顺序
    if args.label_map is not None and os.path.isfile(args.label_map):
        try:
            training_pid_list = _load_training_label_map(args.label_map)
            mapped = apply_label_map_to_dataset(test_loader.dataset, training_pid_list, debug=args.debug)
            # 用训练类别数覆盖配置，确保分类头维度一致
            cfg.defrost()
            cfg.MODEL.NUM_CLASSES = len(training_pid_list)
            cfg.freeze()
            try:
                ds_len = len(test_loader.dataset)  # type: ignore[arg-type]
            except Exception:
                ds_len = mapped
            print(f"已根据 label-map 对齐标签顺序：classes={len(training_pid_list)}，映射样本 {mapped}/{ds_len}")
        except Exception as e:
            print(f"[Warning] Failed to apply label-map: {e}. Proceeding without remap.")
            # 正常用数据集类别数
            cfg.defrost()
            cfg.MODEL.NUM_CLASSES = num_classes
            cfg.freeze()
    else:
        # 正常用数据集类别数
        cfg.defrost()
        cfg.MODEL.NUM_CLASSES = num_classes
        cfg.freeze()
    
    print(f"类别数: {num_classes}")
    print(f"测试样本数: {len(test_loader.dataset)}")  # type: ignore[arg-type]
    
    # 获取类别名称
    class_names = getattr(test_loader.dataset, 'pid_list', None)
    
    # 构建模型
    print("\n构建模型...")
    gpu_id = cfg.GPU_IDS[0] if torch.cuda.is_available() and getattr(cfg, 'NUM_GPUS', 0) > 0 else None
    model = build_model(cfg, gpu_id=gpu_id)
    if isinstance(model, (tuple, list)):
        model = model[0]
    
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
    load_ret = model.load_state_dict(state_dict, strict=False)
    print("模型权重加载完成")
    # Debug: 打印缺失/多余权重键
    if args.debug:
        missing = getattr(load_ret, 'missing_keys', [])
        unexpected = getattr(load_ret, 'unexpected_keys', [])
        print(f"[DEBUG] load_state_dict.missing_keys: {len(missing)}")
        if missing:
            print("  - sample:", missing[:20])
        print(f"[DEBUG] load_state_dict.unexpected_keys: {len(unexpected)}")
        if unexpected:
            print("  - sample:", unexpected[:20])

        # 尝试定位可能的分类头（Linear，out_features=NUM_CLASSES）并打印范数
        try:
            cand = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and module.out_features == cfg.MODEL.NUM_CLASSES:
                    w_norm = module.weight.data.norm().item() if module.weight is not None else float('nan')
                    b_norm = module.bias.data.norm().item() if module.bias is not None else float('nan')
                    cand.append((name, module.in_features, module.out_features, w_norm, b_norm))
            if cand:
                print("[DEBUG] Candidate classification heads (Linear with out_features == NUM_CLASSES):")
                for name, in_f, out_f, wn, bn in cand[:5]:
                    print(f"  - {name}: in={in_f}, out={out_f}, |W|={wn:.4f}, |b|={bn:.4f}")
            else:
                print("[DEBUG] No Linear layer matches out_features == NUM_CLASSES; 分类头可能为自定义模块命名")
        except Exception as e:
            print(f"[DEBUG] Inspect classifier head failed: {e}")
    
    # 测试分类性能
    results = test_classification(cfg, model, test_loader, device, class_names, debug=getattr(args, 'debug', False))

    # 始终计算 ReID 指标（参考训练阶段）
    try:
        rank_ks = [int(x) for x in args.rank_ks.split(',') if x.strip()]
    except Exception:
        rank_ks = [1, 5, 10]
    reid_results = compute_reid_metrics(cfg, model, test_loader.dataset, device, rank_ks)
    
    # 保存结果到JSON
    results_file = os.path.join(args.output_dir, "classification_results.json")
    with open(results_file, 'w') as f:
        # 只保存关键指标，移除混淆矩阵、预测结果等大数组
        results_to_save = {k: v for k, v in results.items() 
                          if k not in ['predictions', 'labels', 'probabilities', 'confusion_matrix']}
        reid_save = {
            'available': reid_results.get('available', True),
            'num_query': reid_results.get('num_query', 0),
            'num_gallery': reid_results.get('num_gallery', 0),
        }
        if reid_save['available']:
            reid_save.update({
                'mAP': reid_results['mAP'] * 100.0,
                'ranks_percent': {k: v * 100.0 for k, v in reid_results['ranks'].items()},
            })
        results_to_save['reid_metrics'] = reid_save
        json.dump(results_to_save, f, indent=2)
    print(f"\n结果已保存到: {results_file}")
    
    # 测试结束时汇总打印（分类 + ReID）
    print(f"\n{'='*80}")
    print("测试结束汇总 (Classification + ReID):")
    print(f"  [Classification]")
    print(f"    Loss     : {results['loss']:.4f}")
    print(f"    Top-1    : {results['top1_accuracy']:.2f}%")
    print(f"    Top-5    : {results['top5_accuracy']:.2f}%")
    print(f"    Acc      : {results['accuracy']:.2f}%")
    print(f"    Precision: {results['precision']:.2f}%  Recall: {results['recall']:.2f}%  F1: {results['f1_score']:.2f}%")
    print(f"  [ReID]")
    if reid_results.get('available', True):
        print(f"    mAP      : {reid_results['mAP']*100.0:.2f}%")
        for rk in [1, 5, 10]:
            key = f'rank{rk}'
            if key in reid_results['ranks']:
                print(f"    Rank-{rk} : {reid_results['ranks'][key]*100.0:.2f}%")
    else:
        print(f"    Unavailable: no valid query/gallery (num_query={reid_results.get('num_query', 0)}, "
              f"num_gallery={reid_results.get('num_gallery', 0)})")
    print(f"{'='*80}")

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
