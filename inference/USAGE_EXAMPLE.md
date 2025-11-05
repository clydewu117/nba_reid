# 单视频特征提取使用指南

## 快速开始

### 1. 对单个视频提取特征

```bash
python -m inference.inference \
    --model timesformer \
    --checkpoint outputs/timesformer_app_ft/timesformer_app_model.pth \
    --mode extract \
    --query /path/to/your/video.mp4 \
    --output ./inference_output \
    --device cuda:0
```

### 2. 参数说明

- `--model`: 模型类型 (`timesformer`, `mvit`, `uniformerv2`)
- `--checkpoint`: 模型权重文件路径
- `--mode`: 运行模式，使用 `extract` 提取单个视频特征
- `--query`: 视频文件路径（支持 .mp4, .avi, .mov 等格式）
- `--output`: 输出目录（默认 `./inference_output`）
- `--device`: 计算设备（默认 `cuda:0`）

### 3. 输出结果

特征将保存为 `.npz` 文件，包含：
- `features`: 特征向量（numpy数组）
- `model_name`: 使用的模型名称
- `checkpoint`: 使用的权重文件路径
- `metadata`: 元数据信息
  - `video_path`: 视频路径
  - `video_name`: 视频名称
  - `model`: 模型类型
  - `num_frames`: 采样帧数

### 4. 读取保存的特征

```python
import numpy as np

# 加载特征
data = np.load('inference_output/video_name_features.npz', allow_pickle=True)

features = data['features']  # 特征向量
metadata = data['metadata'].item()  # 元数据

print(f"特征形状: {features.shape}")
print(f"视频路径: {metadata['video_path']}")
print(f"采样帧数: {metadata['num_frames']}")
```

## 完整示例

### 示例1: 提取单个视频特征

```bash
python -m inference.inference \
    --model timesformer \
    --checkpoint outputs/timesformer_app_ft/timesformer_app_model.pth \
    --mode extract \
    --query /fs/scratch/PAS3184/v2/appearance/LeBron_James/freethrow/video_001.mp4 \
    --output ./my_features
```

### 示例2: 批量提取多个视频特征

创建一个脚本 `batch_extract.sh`:

```bash
#!/bin/bash

MODEL="timesformer"
CHECKPOINT="outputs/timesformer_app_ft/timesformer_app_model.pth"
OUTPUT_DIR="./batch_features"

# 视频列表
VIDEOS=(
    "/path/to/video1.mp4"
    "/path/to/video2.mp4"
    "/path/to/video3.mp4"
)

for VIDEO in "${VIDEOS[@]}"; do
    echo "处理: $VIDEO"
    python -m inference.inference \
        --model $MODEL \
        --checkpoint $CHECKPOINT \
        --mode extract \
        --query "$VIDEO" \
        --output "$OUTPUT_DIR"
done

echo "批量处理完成!"
```

### 示例3: Python API 使用

```python
from inference.inference import ReIDInference
import torch

# 初始化推理器
inferencer = ReIDInference(
    model_name='timesformer',
    checkpoint_path='outputs/timesformer_app_ft/timesformer_app_model.pth',
    device='cuda:0'
)

# 方法1: 从视频文件加载
video_tensor = inferencer.load_video('/path/to/video.mp4')
features = inferencer.extract_features(video_tensor)
print(f"特征形状: {features.shape}")

# 方法2: 使用自己的视频张量
# 形状必须是 [B, C, T, H, W] 或 [C, T, H, W]
# 例如: [1, 3, 16, 224, 224]
custom_video = torch.randn(1, 3, 16, 224, 224)
features = inferencer.extract_features(custom_video)

# 保存特征
inferencer.save_features(
    features, 
    'output.npz',
    metadata={'source': 'custom_video'}
)
```

## 视频要求

- **格式**: 支持常见视频格式（.mp4, .avi, .mov 等）
- **分辨率**: 任意（会自动resize到224x224）
- **帧数**: 任意（会自动采样到模型要求的帧数，通常是16帧）
- **时长**: 任意（均匀采样）

## 注意事项

1. **GPU内存**: 确保有足够的GPU内存，单个视频通常需要 < 2GB
2. **批处理**: 如果要处理多个视频，建议逐个处理以节省内存
3. **采样策略**: 默认使用均匀采样（从视频中均匀抽取帧）
4. **特征归一化**: 提取的特征已经L2归一化，可直接用于相似度计算

## 故障排除

### 问题1: 模型加载失败（形状不匹配）
**已解决** - 代码会自动从checkpoint检测正确的配置

### 问题2: 视频无法打开
检查：
- 视频文件是否存在
- 文件路径是否正确
- 视频格式是否支持
- OpenCV是否正确安装

### 问题3: GPU内存不足
解决方法：
- 使用CPU: `--device cpu`
- 减小batch size（当前默认为1）

## 高级用法

### 自定义帧数
模型会自动使用checkpoint训练时的帧数，无需手动指定。

### 提取中间层特征
如需提取中间层特征，需要修改模型的forward方法，这是高级用法。

### 特征后处理
```python
import numpy as np

# 加载特征
features = np.load('output.npz')['features']

# L2归一化（如果还未归一化）
features = features / (np.linalg.norm(features) + 1e-12)

# 计算两个特征的余弦相似度
feat1 = features  # 第一个视频特征
feat2 = ...  # 第二个视频特征
similarity = np.dot(feat1, feat2)
print(f"相似度: {similarity:.4f}")
```
