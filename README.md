# Video-based Person ReID with CAM Visualization

This repository provides tools for generating Class Activation Maps (CAM) on video-based person re-identification models. It supports two state-of-the-art video transformer architectures: **MViTv2** (Multi-scale Vision Transformer v2) and **UniFormerV2**.

## Overview

The workflow consists of:
1. **Model Loading**: Load pre-trained ReID models from checkpoints
2. **Video Processing**: Extract and preprocess frames from input videos
3. **CAM Generation**: Generate visual explanations using various CAM methods
4. **Visualization**: Create overlayed frames and output videos showing model attention

## Supported Architectures

### MViTv2 (Multi-scale Vision Transformer v2)
- Script: `run_gradcam_mvit.py`
- Configuration: Defined programmatically in script
- Features: Multi-scale architecture with hierarchical attention

### UniFormerV2
- Script: `run_gradcam_uniformer.py`
- Configuration: Requires external YAML config file
- Features: Unified transformer design with local-global attention

## CAM Methods

Five different CAM methods are supported, each with unique characteristics:

| Method | Type | Description | Best For |
|--------|------|-------------|----------|
| **OriginalCAM** | Gradient-free | Simple, fast, effective | Default choice, no gradient issues |
| **GradCAM** | Gradient-based | Classic method using gradients | General visualization |
| **GradCAM++** | Gradient-based | Second-order gradients | Better localization |
| **LayerCAM** | Gradient-based | Layer-wise gradients | Fine-grained analysis |
| **ScoreCAM** | Gradient-free | Perturbation-based scoring | Gradient-independent alternative |

### Method Selection Guide

- **Default**: Use `originalcam` - simple, fast, no gradient attenuation issues
- **For comparison**: Try `gradcam` or `scorecam`
- **For better localization**: Use `gradcam++` or `layercam`
- **When gradients are weak**: Use gradient-free methods (`originalcam`, `scorecam`)

**Important Note**: Based on extensive testing, gradient-dependent methods (GradCAM, GradCAM++, LayerCAM) suffer from gradient vanishing issues in deep transformer architectures. **We recommend using `gradcam` or `originalcam`** (OriginalCAM is used in the UniFormer paper) for more reliable and interpretable visualizations.


## Usage

### Basic Usage - MViTv2

```bash
python run_gradcam_mvit.py \
    --video /path/to/video.mp4 \
    --checkpoint /path/to/mvit_model.pth \
    --output ./cam_output \
    --method originalcam
```

### Basic Usage - UniFormerV2

```bash
python run_gradcam_uniformer.py \
    --video /path/to/video.mp4 \
    --checkpoint /path/to/uniformer_model.pth \
    --config /path/to/config.yaml \
    --output ./cam_uniformer_output \
    --method originalcam
```

### Advanced Options

```bash
# Specify target class (instead of predicted class)
--target_id 42

# Change number of sampled frames
--frames 32

# Use different CAM method
--method scorecam

# ScoreCAM batch size (memory vs speed tradeoff)
--scorecam_batch_size 64

# Enable FINER mode (target vs second-highest class)
--finer

# Use CPU instead of GPU
--device cpu
```

### 2. Video Processing

Videos are processed as follows:
1. **Frame Extraction**: Uniformly sample T frames (default: 16) across entire video
2. **Preprocessing**:
   - Resize to 224x224 for model input
   - Keep original resolution for high-quality visualization
   - Normalize using mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]
3. **Tensor Format**: [1, C, T, H, W] where C=3, T=16, H=W=224

### 3. CAM Generation

The CAM generation process:
1. **Target Layer Selection**: Use last transformer block
   - MViTv2: `model.backbone.blocks[-1]`
   - UniFormerV2: `model.backbone.transformer.resblocks[-1]`
2. **Forward Pass**: Process input through model
3. **Activation Extraction**: Capture intermediate feature maps
4. **CAM Computation**: Apply chosen CAM method
5. **Spatial Reshaping**: Convert flat tokens to [T, H, W] format
   - MViTv2: Typically 8×7×7 (temporal downsampling)
   - UniFormerV2: Typically 16×14×14 (full temporal resolution)

### 4. Visualization Output

Each run produces:

**Directory Structure**:
```
output_directory/
├── frames/                  # Individual overlayed frames
│   ├── frame_000.jpg
│   ├── frame_001.jpg
│   └── ...
├── cam_activations/         # Raw CAM arrays (NumPy)
│   ├── frame_000_cam.npy
│   ├── frame_001_cam.npy
│   └── ...
└── cam_video.mp4           # Final video with CAM overlay
```

**Visualization Details**:
- CAM is normalized per-frame to [0, 1]
- Applied using COLORMAP_JET (blue=low, red=high)
- 50/50 blend with original frame
- Output preserves original video resolution

