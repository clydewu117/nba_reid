# NBA ReID 推理工具

支持三种视频重识别模型的统一推理接口：**TimeSformer**、**MViT**、**UniFormerV2**

## 📁 文件结构

```
inference/
├── README.md        # 本文件 - 使用说明
├── __init__.py      # Python 包初始化
└── inference.py     # 核心推理模块
```

## 🚀 快速开始

### Python API 使用

```python
from inference import ReIDInference
import torch

# 初始化推理器
inferencer = ReIDInference(
    model_name='timesformer',  # 'timesformer', 'mvit', 'uniformerv2'
    checkpoint_path='outputs/timesformer_app_ft/timesformer_app_model.pth',
    device='cuda:0'
)

# 提取特征
video = torch.randn(1, 3, 32, 224, 224)  # [B, C, T, H, W]
features = inferencer.extract_features(video)
print(f"特征形状: {features.shape}")
```

### 命令行使用

```bash
python -m inference.inference \
    --model timesformer \
    --checkpoint outputs/timesformer_app_ft/timesformer_app_model.pth \
    --mode extract
```

## 🎯 支持的模型

| 模型 | 特点 | 推荐场景 |
|------|------|----------|
| **TimeSformer** | 分割时空注意力，平衡性能 | 通用场景 |
| **MViT** | 多尺度特征，速度快 | 快速推理 |
| **UniFormerV2** | 局部+全局，准确率高 | 最佳性能 |

## � 主要功能

### 1. 特征提取
```python
features = inferencer.extract_features(video_tensor)
```

### 2. 相似度计算
```python
similarity = inferencer.compute_similarity(
    query_feat, gallery_feat, metric='cosine'
)
```

### 3. Gallery 排序
```python
results = inferencer.rank_gallery(
    query_feat, gallery_feat, gallery_ids, top_k=10
)
```

### 4. 特征保存
```python
inferencer.save_features(features, 'output.npz')
```

## 📖 完整示例

```python
from inference import ReIDInference
import torch
import numpy as np

# 初始化
inferencer = ReIDInference(
    model_name='timesformer',
    checkpoint_path='outputs/timesformer_app_ft/timesformer_app_model.pth'
)

# 准备数据
query_video = torch.randn(1, 3, 32, 224, 224)
gallery_videos = torch.randn(10, 3, 32, 224, 224)
gallery_ids = [f'person_{i}' for i in range(10)]

# 提取特征
query_feat = inferencer.extract_features(query_video)
gallery_feat = inferencer.extract_features(gallery_videos)

# 检索排序
results = inferencer.rank_gallery(
    query_feat, gallery_feat, gallery_ids, top_k=5
)

# 显示结果
for r in results:
    print(f"Rank {r['rank']}: {r['id']} (相似度: {r['similarity']:.4f})")
```

## � 使用技巧

### 批量处理
```python
batch_size = 8
all_features = []
for i in range(0, num_videos, batch_size):
    batch = videos[i:i+batch_size]
    features = inferencer.extract_features(batch)
    all_features.append(features)
all_features = np.vstack(all_features)
```

### 多模型集成
```python
models = ['timesformer', 'mvit']
ensemble_feat = []
for model in models:
    inf = ReIDInference(model, checkpoints[model])
    feat = inf.extract_features(video)
    ensemble_feat.append(feat)
final_feat = np.mean(ensemble_feat, axis=0)
```

## ⚙️ 环境要求

- Python >= 3.8
- PyTorch >= 1.10
- CUDA >= 11.0（GPU推理）
- 其他依赖见项目根目录的 `environment.yml`

## 🐛 常见问题

**Q: 显存不足？**  
A: 减小 batch_size 或使用 CPU
```python
inferencer = ReIDInference(..., device='cpu')
```

**Q: 模型加载失败？**  
A: 检查 checkpoint 路径是否正确（相对于项目根目录）

## 📞 API 参考

### ReIDInference 类

**初始化参数：**
- `model_name`: 模型名称 ('timesformer', 'mvit', 'uniformerv2')
- `checkpoint_path`: 模型权重文件路径
- `config_path`: 配置文件路径（可选）
- `device`: 计算设备（默认: 'cuda:0'）

**主要方法：**
- `extract_features(video_tensor)`: 提取特征
- `compute_similarity(query_feat, gallery_feat, metric)`: 计算相似度
- `rank_gallery(query_feat, gallery_feat, ids, top_k)`: 排序检索
- `save_features(features, path, metadata)`: 保存特征

**输入格式：**
- 视频张量：`[B, C, T, H, W]` = `[batch_size, 3, 32, 224, 224]`

---

更多信息请参考项目文档或源代码注释。
