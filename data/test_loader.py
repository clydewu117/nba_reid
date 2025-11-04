#!/usr/bin/env python
"""
Quick test script for Basketball DataLoader
"""

import sys
import os

# 确保能import到相关模块
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from config.defaults import get_cfg_defaults, get_appearance_both_config
from data.dataloader import build_dataloader, verify_no_overlap


def test_dataloader(cfg_func, cfg_name):
    """测试DataLoader"""
    print("\n" + "="*80)
    print(f"Testing: {cfg_name}")
    print("="*80)
    
    cfg = cfg_func()
    
    print(f"\nConfiguration:")
    print(f"  Data Root: {cfg.DATA.ROOT}")
    print(f"  Video Type: {cfg.DATA.VIDEO_TYPE}")
    print(f"  Shot Type: {cfg.DATA.SHOT_TYPE}")
    print(f"  Num Frames: {cfg.DATA.NUM_FRAMES}")
    print(f"  Frame Stride: {cfg.DATA.FRAME_STRIDE}")
    print(f"  Image Size: {cfg.DATA.HEIGHT} x {cfg.DATA.WIDTH}")
    print(f"  Batch Size: {cfg.DATA.BATCH_SIZE}")
    print(f"  Train Ratio: {cfg.DATA.TRAIN_RATIO}")
    
    try:
        # 构建训练集
        print("\n--- Building Train DataLoader ---")
        train_loader, num_classes = build_dataloader(cfg, is_train=True)
        
        print(f"\nTrain Dataset Info:")
        print(f"  Num Classes: {num_classes}")
        print(f"  Num Batches: {len(train_loader)}")
        print(f"  Total Samples: {len(train_loader.dataset)}")
        
        # 构建测试集
        print("\n--- Building Test DataLoader ---")
        test_loader, _ = build_dataloader(cfg, is_train=False)
        
        print(f"\nTest Dataset Info:")
        print(f"  Num Batches: {len(test_loader)}")
        print(f"  Total Samples: {len(test_loader.dataset)}")
        
        # ✅ 验证训练集和测试集无重复
        print("\n--- Verifying Train/Test Split ---")
        is_valid = verify_no_overlap(train_loader, test_loader)
        if not is_valid:
            print("❌ Train/Test overlap detected!")
            return False
        
        # 测试一个batch
        print("\n--- Testing First Batch ---")
        batch = next(iter(train_loader))
        
        print(f"\nBatch Contents:")
        print(f"  Video shape: {batch['video'].shape}")  # [B, 3, T, H, W]
        print(f"  Video dtype: {batch['video'].dtype}")
        print(f"  Video range: [{batch['video'].min():.3f}, {batch['video'].max():.3f}]")
        print(f"  PIDs: {batch['pid'].tolist()}")
        print(f"  PIDs shape: {batch['pid'].shape}")
        print(f"  Num identities: {len(set(batch['identities']))}")
        print(f"  Shot types: {set(batch['shot_types'])}")
        
        # 验证数据
        print("\n--- Validation ---")
        assert batch['video'].shape[0] == cfg.DATA.BATCH_SIZE, "Batch size mismatch"
        assert batch['video'].shape[1] == 3, "Channel should be 3"
        assert batch['video'].shape[2] == cfg.DATA.NUM_FRAMES, "Num frames mismatch"
        assert batch['video'].shape[3] == cfg.DATA.HEIGHT, "Height mismatch"
        assert batch['video'].shape[4] == cfg.DATA.WIDTH, "Width mismatch"
        assert len(batch['pid']) == cfg.DATA.BATCH_SIZE, "PID length mismatch"
        assert all(0 <= pid < num_classes for pid in batch['pid'].tolist()), "Invalid PID"
        
        print("✅ All validations passed!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("Basketball Video ReID DataLoader Test")
    print("="*80)
    
    # 测试不同配置
    test_configs = [
        (get_cfg_defaults, "Default Config"),
        # 取消注释以测试其他配置
        # (get_appearance_both_config, "Appearance + Both"),
        # (get_appearance_freethrow_config, "Appearance + Freethrow"),
        # (get_mask_both_config, "Mask + Both"),
    ]
    
    results = []
    for cfg_func, cfg_name in test_configs:
        success = test_dataloader(cfg_func, cfg_name)
        results.append((cfg_name, success))
    
    # 打印总结
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    for cfg_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {cfg_name}")
    
    # 返回退出码
    all_passed = all(success for _, success in results)
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)