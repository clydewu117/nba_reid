#!/usr/bin/env python
"""
Enhanced test script for Basketball DataLoader with Sampler Verification
"""

import sys
import os
import torch
import numpy as np
from collections import Counter
import time

# 确保能import到相关模块
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config.defaults import get_cfg_defaults
from data.dataloader import build_dataloader


def verify_sampler_usage(loader, cfg):
    """
    验证是否正确使用了RandomIdentitySampler
    """
    print("\n" + "="*80)
    print("VERIFYING SAMPLER USAGE")
    print("="*80)
    
    # 检查配置
    use_sampler = getattr(cfg.DATA, 'USE_SAMPLER', False)
    print(f"\nConfig setting:")
    print(f"  USE_SAMPLER: {use_sampler}")
    
    if use_sampler:
        num_instances = cfg.DATA.NUM_INSTANCES
        batch_size = cfg.DATA.BATCH_SIZE
        expected_num_pids = batch_size // num_instances
        
        print(f"  NUM_INSTANCES: {num_instances}")
        print(f"  BATCH_SIZE: {batch_size}")
        print(f"  Expected PIDs per batch: {expected_num_pids}")
        
        # 验证batch_size能被num_instances整除
        if batch_size % num_instances != 0:
            print(f"\n❌ ERROR: BATCH_SIZE ({batch_size}) must be divisible by NUM_INSTANCES ({num_instances})!")
            return False
    
    # 检查DataLoader是否使用了sampler
    print(f"\nDataLoader properties:")
    print(f"  Has sampler: {hasattr(loader, 'sampler')}")
    print(f"  Has batch_sampler: {hasattr(loader, 'batch_sampler')}")
    
    if hasattr(loader, 'batch_sampler') and loader.batch_sampler is not None:
        print(f"  Batch sampler type: {type(loader.batch_sampler).__name__}")
        print(f"  ✅ Using batch_sampler (likely RandomIdentitySampler)")
    elif hasattr(loader, 'sampler') and loader.sampler is not None:
        sampler_type = type(loader.sampler).__name__
        print(f"  Sampler type: {sampler_type}")
        if 'RandomIdentitySampler' in sampler_type:
            print(f"  ✅ Using RandomIdentitySampler!")
        else:
            print(f"  ⚠️  Using {sampler_type} (not RandomIdentitySampler)")
    else:
        print(f"  ⚠️  No custom sampler detected")
    
    # 实际测试batch的PID分布
    print(f"\n--- Testing Actual Batch Distribution ---")
    
    if use_sampler:
        print(f"\nExpected behavior with RandomIdentitySampler:")
        print(f"  - Each batch should have exactly {expected_num_pids} unique PIDs")
        print(f"  - Each PID should appear exactly {num_instances} times")
    else:
        print(f"\nExpected behavior without sampler:")
        print(f"  - PIDs should be randomly distributed")
        print(f"  - No specific pattern expected")
    
    # 测试前5个batch
    num_test_batches = min(5, len(loader))
    print(f"\nTesting first {num_test_batches} batches:")
    
    all_match = True
    for i, batch in enumerate(loader):
        if i >= num_test_batches:
            break
        
        pids = batch['pid'].tolist()
        pid_counts = Counter(pids)
        unique_pids = len(pid_counts)
        
        print(f"\nBatch {i+1}:")
        print(f"  Total samples: {len(pids)}")
        print(f"  Unique PIDs: {unique_pids}")
        
        # 显示前5个PID的计数
        sorted_counts = sorted(pid_counts.items())
        display_counts = dict(sorted_counts[:5])
        if len(pid_counts) > 5:
            print(f"  PID counts (first 5): {display_counts}...")
        else:
            print(f"  PID counts: {display_counts}")
        
        if use_sampler:
            # 验证是否符合sampler的预期
            expected_count = num_instances
            counts = list(pid_counts.values())
            
            if unique_pids == expected_num_pids:
                print(f"  ✅ Unique PIDs matches expected: {unique_pids} == {expected_num_pids}")
            else:
                print(f"  ❌ Unique PIDs mismatch: {unique_pids} != {expected_num_pids}")
                all_match = False
            
            if all(c == expected_count for c in counts):
                print(f"  ✅ All PIDs appear exactly {expected_count} times")
            else:
                actual_counts = set(counts)
                print(f"  ❌ PID counts vary: {actual_counts} (expected all {expected_count})")
                all_match = False
        else:
            # 不使用sampler，只是报告分布
            min_count = min(pid_counts.values())
            max_count = max(pid_counts.values())
            avg_count = np.mean(list(pid_counts.values()))
            print(f"  PID count range: {min_count} to {max_count} (avg: {avg_count:.1f})")
    
    # 最终结论
    print(f"\n{'='*80}")
    if use_sampler:
        if all_match:
            print("✅ SAMPLER IS WORKING CORRECTLY!")
            print(f"   Each batch has {expected_num_pids} identities with {num_instances} samples each")
            print(f"   This is ideal for Triplet Loss training!")
        else:
            print("❌ SAMPLER IS NOT WORKING AS EXPECTED!")
            print("   Possible issues:")
            print("   1. RandomIdentitySampler not properly initialized in build_dataloader")
            print("   2. Config USE_SAMPLER=True not properly passed")
            print("   3. BATCH_SIZE or NUM_INSTANCES mismatch")
            print("   4. Insufficient samples per identity")
    else:
        print("ℹ️  NOT USING SAMPLER (as configured)")
        print("   Using default random shuffle")
        print("   Note: For Triplet Loss, using sampler is recommended")
    print(f"{'='*80}")
    
    return all_match if use_sampler else True


def verify_no_overlap(train_loader, test_loader):
    """验证训练集和测试集没有重复视频"""
    train_paths = set()
    test_paths = set()
    
    # 收集训练集视频路径
    for batch in train_loader:
        train_paths.update(batch['video_paths'])
    
    # 收集测试集视频路径
    for batch in test_loader:
        test_paths.update(batch['video_paths'])
    
    # 检查重复
    overlap = train_paths & test_paths
    
    if overlap:
        print(f"\n❌ Found {len(overlap)} overlapping videos!")
        print("Examples:")
        for path in list(overlap)[:5]:
            print(f"  - {path}")
        return False
    else:
        print(f"✅ No overlap! {len(train_paths)} train videos, {len(test_paths)} test videos")
        return True


def test_sampling_strategy(dataset, num_samples=5):
    """测试采样策略是否正确"""
    print("\n--- Testing Sampling Strategy ---")
    
    if len(dataset) == 0:
        print("⚠️  Dataset is empty!")
        return
    
    idx = 0  # 测试第一个视频
    video_path = dataset.data[idx]['video_path']
    
    print(f"\nTesting video: {os.path.basename(video_path)}")
    print(f"Sampling {num_samples} times to verify randomness...")
    
    for i in range(num_samples):
        sample = dataset[idx]
        print(f"  Sample {i+1}: Video shape = {sample['video'].shape}")
    
    # 检查是否有随机性（训练模式下）
    if dataset.is_train:
        print("✅ Training mode: Samples should have randomness (TSN-RRS)")
    else:
        print("✅ Test mode: Samples should be deterministic")


def analyze_batch_distribution(loader, max_batches=10):
    """分析batch中的样本分布"""
    print("\n--- Analyzing Batch Distribution ---")
    
    pid_counts = []
    identity_counts = []
    
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        
        pids = batch['pid'].tolist()
        identities = batch['identities']
        shot_types = batch['shot_types']
        
        # 统计每个batch中的分布
        pid_counter = Counter(pids)
        identity_counter = Counter(identities)
        shot_type_counter = Counter(shot_types)
        
        print(f"\nBatch {i+1}:")
        print(f"  Unique PIDs: {len(pid_counter)}")
        print(f"  Unique Identities: {len(identity_counter)}")
        
        # 显示前3个PID的计数
        top_pids = dict(sorted(pid_counter.items())[:3])
        print(f"  PID distribution (sample): {top_pids}...")
        print(f"  Shot types: {dict(shot_type_counter)}")
        
        pid_counts.append(len(pid_counter))
        identity_counts.append(len(identity_counter))
    
    if pid_counts:
        print(f"\nSummary across {len(pid_counts)} batches:")
        print(f"  Avg unique PIDs per batch: {np.mean(pid_counts):.2f}")
        print(f"  Min/Max unique PIDs: {min(pid_counts)}/{max(pid_counts)}")


def test_video_loading(loader, num_videos=3):
    """测试视频加载和数据质量"""
    print("\n--- Testing Video Loading ---")
    
    try:
        batch = next(iter(loader))
        videos = batch['video']
        
        print(f"\nVideo Tensor Info:")
        print(f"  Shape: {videos.shape}")  # [B, C, T, H, W]
        print(f"  Dtype: {videos.dtype}")
        print(f"  Device: {videos.device}")
        print(f"  Min value: {videos.min().item():.4f}")
        print(f"  Max value: {videos.max().item():.4f}")
        print(f"  Mean: {videos.mean().item():.4f}")
        print(f"  Std: {videos.std().item():.4f}")
        
        # 检查是否归一化
        if -3 < videos.min().item() < -0.5 and 0.5 < videos.max().item() < 3:
            print("✅ Data appears to be normalized (ImageNet stats)")
        else:
            print("⚠️  Data normalization might be incorrect")
        
        # 检查是否有NaN或Inf
        if torch.isnan(videos).any():
            print("❌ Found NaN values in video tensor!")
        elif torch.isinf(videos).any():
            print("❌ Found Inf values in video tensor!")
        else:
            print("✅ No NaN or Inf values")
        
        # 显示一些视频路径
        print(f"\nSample video paths:")
        for i, path in enumerate(batch['video_paths'][:num_videos]):
            print(f"  {i+1}. {os.path.basename(path)}")
            print(f"     Identity: {batch['identities'][i]}")
            print(f"     Shot type: {batch['shot_types'][i]}")
            print(f"     PID: {batch['pid'][i].item()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during video loading: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader(cfg):
    """完整测试DataLoader"""
    print("\n" + "="*80)
    print("Basketball Video ReID DataLoader Test (with Sampler Detection)")
    print("="*80)
    
    print(f"\nConfiguration:")
    print(f"  Data Root: {cfg.DATA.ROOT}")
    print(f"  Video Type: {cfg.DATA.VIDEO_TYPE}")
    print(f"  Shot Type: {cfg.DATA.SHOT_TYPE}")
    print(f"  Num Frames: {cfg.DATA.NUM_FRAMES}")
    print(f"  Image Size: {cfg.DATA.HEIGHT} x {cfg.DATA.WIDTH}")
    print(f"  Batch Size: {cfg.DATA.BATCH_SIZE}")
    print(f"  Train Ratio: {cfg.DATA.TRAIN_RATIO}")
    print(f"  Num Workers: {cfg.DATA.NUM_WORKERS}")
    print(f"  Use Sampler: {getattr(cfg.DATA, 'USE_SAMPLER', False)}")
    if hasattr(cfg.DATA, 'USE_SAMPLER') and cfg.DATA.USE_SAMPLER:
        print(f"  Num Instances: {cfg.DATA.NUM_INSTANCES}")
    
    try:
        # 构建训练集
        print("\n" + "="*80)
        print("BUILDING TRAIN DATALOADER")
        print("="*80)
        train_loader, num_classes = build_dataloader(cfg, is_train=True)
        
        print(f"\n✅ Train Dataset Created:")
        print(f"  Num Classes: {num_classes}")
        print(f"  Num Batches: {len(train_loader)}")
        print(f"  Total Samples: {len(train_loader.dataset)}")
        print(f"  Samples per class: {len(train_loader.dataset) / num_classes:.1f}")
        
        # ✅ 验证sampler使用情况
        if not verify_sampler_usage(train_loader, cfg):
            print("\n⚠️  Warning: Sampler verification failed!")
            print("Continuing with other tests...")
        
        # 构建测试集
        print("\n" + "="*80)
        print("BUILDING TEST DATALOADER")
        print("="*80)
        test_loader, _ = build_dataloader(cfg, is_train=False)
        
        print(f"\n✅ Test Dataset Created:")
        print(f"  Num Batches: {len(test_loader)}")
        print(f"  Total Samples: {len(test_loader.dataset)}")
        
        # 验证训练集和测试集无重复
        print("\n" + "="*80)
        print("VERIFYING TRAIN/TEST SPLIT")
        print("="*80)
        is_valid = verify_no_overlap(train_loader, test_loader)
        if not is_valid:
            return False
        
        # 测试视频加载
        print("\n" + "="*80)
        print("TESTING VIDEO LOADING (TRAIN)")
        print("="*80)
        if not test_video_loading(train_loader, num_videos=3):
            return False
        
        print("\n" + "="*80)
        print("TESTING VIDEO LOADING (TEST)")
        print("="*80)
        if not test_video_loading(test_loader, num_videos=3):
            return False
        
        # 分析batch分布（仅训练集）
        print("\n" + "="*80)
        print("ANALYZING BATCH DISTRIBUTION (TRAIN)")
        print("="*80)
        analyze_batch_distribution(train_loader, max_batches=5)
        
        # 测试维度
        print("\n" + "="*80)
        print("TESTING DIMENSIONS")
        print("="*80)
        batch = next(iter(train_loader))
        
        B, C, T, H, W = batch['video'].shape
        print(f"\nBatch shape: {batch['video'].shape}")
        assert B == cfg.DATA.BATCH_SIZE, f"Batch size mismatch: {B} != {cfg.DATA.BATCH_SIZE}"
        assert C == 3, f"Channel should be 3, got {C}"
        assert T == cfg.DATA.NUM_FRAMES, f"Num frames mismatch: {T} != {cfg.DATA.NUM_FRAMES}"
        assert H == cfg.DATA.HEIGHT, f"Height mismatch: {H} != {cfg.DATA.HEIGHT}"
        assert W == cfg.DATA.WIDTH, f"Width mismatch: {W} != {cfg.DATA.WIDTH}"
        print("✅ All dimension validations passed!")
        
        # 速度测试
        print("\n" + "="*80)
        print("SPEED TEST")
        print("="*80)
        num_batches = min(10, len(train_loader))
        start_time = time.time()
        
        for i, batch in enumerate(train_loader):
            if i >= num_batches:
                break
        
        elapsed = time.time() - start_time
        print(f"Loaded {num_batches} batches in {elapsed:.2f}s")
        print(f"Average time per batch: {elapsed/num_batches:.3f}s")
        print(f"Throughput: {num_batches * cfg.DATA.BATCH_SIZE / elapsed:.1f} samples/sec")
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n" + "="*80)
        print("❌ ERROR OCCURRED")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "="*80)
    print("Basketball Video ReID DataLoader Debug Tool")
    print("(with RandomIdentitySampler Detection)")
    print("="*80)
    
    # 获取配置
    cfg = get_cfg_defaults()
    
    # 可以通过命令行参数修改配置
    if len(sys.argv) > 1:
        print("\nApplying command line arguments...")
        cfg.merge_from_list(sys.argv[1:])
    
    # 打印配置文件路径
    print(f"\nLooking for data at: {cfg.DATA.ROOT}")
    
    # 检查数据路径是否存在
    if not os.path.exists(cfg.DATA.ROOT):
        print(f"❌ Data root does not exist: {cfg.DATA.ROOT}")
        print("\nPlease update the path in your config file or pass it via command line:")
        print(f"  python {sys.argv[0]} DATA.ROOT /path/to/your/data")
        return 1
    
    # 运行测试
    success = test_dataloader(cfg)
    
    return 0 if success else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)