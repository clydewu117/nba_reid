#!/usr/bin/env python3
"""
UniFormerV2 CAM Visualization - Unified Multi-Method Version

Supports multiple CAM methods with eval mode inference:
- ✓ Original CAM (gradient-free, default)
- ✓ GradCAM / GradCAM++ / LayerCAM (gradient-based)
- ✓ ScoreCAM (gradient-free, perturbation-based)
- ✓ Accurate predictions (using running statistics)

Default: Original CAM (gradient-free, simple, effective)
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from models.uniformerv2_reid import Uniformerv2ReID
from yacs.config import CfgNode as CN
import utils.logging as logging
from cam_util import SimpleGradCAM, ScoreCAM, LayerCAM, GradCAMPlusPlus, OriginalCAM

logger = logging.get_logger(__name__)


def load_config(config_path, num_classes=None):
    """Load YACS config from YAML file with safe defaults."""
    yaml_cfg = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            yaml_cfg = yaml.safe_load(f) or {}

    cfg = CN()

    # Data config
    cfg.DATA = CN()
    data_cfg = yaml_cfg.get("DATA", {})
    cfg.DATA.BATCH_SIZE = data_cfg.get("BATCH_SIZE", 16)
    cfg.DATA.NUM_FRAMES = data_cfg.get("NUM_FRAMES", 16)
    cfg.DATA.HEIGHT = data_cfg.get("HEIGHT", 224)
    cfg.DATA.WIDTH = data_cfg.get("WIDTH", 224)
    cfg.DATA.NUM_WORKERS = data_cfg.get("NUM_WORKERS", 4)
    cfg.DATA.FRAME_STRIDE = data_cfg.get("FRAME_STRIDE", 4)
    cfg.DATA.NUM_INSTANCES = data_cfg.get("NUM_INSTANCES", 4)
    cfg.DATA.ROOT = data_cfg.get("ROOT", "")
    cfg.DATA.SHOT_TYPE = data_cfg.get("SHOT_TYPE", "")
    cfg.DATA.TRAIN_RATIO = data_cfg.get("TRAIN_RATIO", 0.7)
    cfg.DATA.USE_SAMPLER = data_cfg.get("USE_SAMPLER", True)
    cfg.DATA.VIDEO_TYPE = data_cfg.get("VIDEO_TYPE", "")
    cfg.DATA.SAMPLE_START = data_cfg.get("SAMPLE_START", "beginning")
    cfg.DATA.SPLIT_SAMPLING = data_cfg.get("SPLIT_SAMPLING", False)
    cfg.DATA.USE_PRESPLIT = data_cfg.get("USE_PRESPLIT", False)

    # Model config
    cfg.MODEL = CN()
    model_cfg = yaml_cfg.get("MODEL", {})
    cfg.MODEL.ARCH = model_cfg.get("ARCH", "uniformerv2")
    cfg.MODEL.MODEL_NAME = model_cfg.get("MODEL_NAME", "Uniformerv2ReID")
    cfg.MODEL.NAME = model_cfg.get("NAME", "Uniformerv2ReID")
    cfg.MODEL.NUM_CLASSES = (
        num_classes if num_classes is not None else model_cfg.get("NUM_CLASSES", 0)
    )
    cfg.MODEL.USE_CHECKPOINT = model_cfg.get("USE_CHECKPOINT", True)
    cfg.MODEL.CHECKPOINT_NUM = model_cfg.get("CHECKPOINT_NUM", [0])

    # UniFormerV2 config (12-layer UniFormerV2-B/16)
    cfg.UNIFORMERV2 = CN()
    uniformer_cfg = yaml_cfg.get("UNIFORMERV2", {})
    cfg.UNIFORMERV2.BACKBONE = uniformer_cfg.get("BACKBONE", "uniformerv2_b16")
    cfg.UNIFORMERV2.N_LAYERS = uniformer_cfg.get("N_LAYERS", 12)
    cfg.UNIFORMERV2.N_DIM = uniformer_cfg.get("N_DIM", 768)
    cfg.UNIFORMERV2.N_HEAD = uniformer_cfg.get("N_HEAD", 12)
    cfg.UNIFORMERV2.MLP_FACTOR = uniformer_cfg.get("MLP_FACTOR", 4.0)
    cfg.UNIFORMERV2.BACKBONE_DROP_PATH_RATE = uniformer_cfg.get(
        "BACKBONE_DROP_PATH_RATE", 0.0
    )
    cfg.UNIFORMERV2.DROP_PATH_RATE = uniformer_cfg.get("DROP_PATH_RATE", 0.0)
    cfg.UNIFORMERV2.MLP_DROPOUT = uniformer_cfg.get(
        "MLP_DROPOUT", [0.5] * cfg.UNIFORMERV2.N_LAYERS
    )
    cfg.UNIFORMERV2.CLS_DROPOUT = uniformer_cfg.get("CLS_DROPOUT", 0.5)
    cfg.UNIFORMERV2.RETURN_LIST = uniformer_cfg.get("RETURN_LIST", [8, 9, 10, 11])
    cfg.UNIFORMERV2.TEMPORAL_DOWNSAMPLE = uniformer_cfg.get("TEMPORAL_DOWNSAMPLE", False)
    cfg.UNIFORMERV2.DW_REDUCTION = uniformer_cfg.get("DW_REDUCTION", 1.5)
    cfg.UNIFORMERV2.NO_LMHRA = uniformer_cfg.get("NO_LMHRA", False)
    cfg.UNIFORMERV2.DOUBLE_LMHRA = uniformer_cfg.get("DOUBLE_LMHRA", True)
    cfg.UNIFORMERV2.PRETRAIN = uniformer_cfg.get("PRETRAIN", "")
    cfg.UNIFORMERV2.FROZEN = uniformer_cfg.get("FROZEN", False)

    # ReID config
    cfg.REID = CN()
    reid_cfg = yaml_cfg.get("REID", {})
    cfg.REID.EMBED_DIM = reid_cfg.get("EMBED_DIM", 512)
    cfg.REID.NECK_FEAT = reid_cfg.get("NECK_FEAT", "after")

    # Loss config
    cfg.LOSS = CN()
    loss_cfg = yaml_cfg.get("LOSS", {})
    cfg.LOSS.ID_WEIGHT = loss_cfg.get("ID_WEIGHT", 1.0)
    cfg.LOSS.TRIPLET_WEIGHT = loss_cfg.get("TRIPLET_WEIGHT", 1.0)
    cfg.LOSS.TRIPLET_MARGIN = loss_cfg.get("TRIPLET_MARGIN", 0.3)
    cfg.LOSS.TRIPLET_DISTANCE = loss_cfg.get("TRIPLET_DISTANCE", "euclidean")
    cfg.LOSS.USE_TRIPLET = loss_cfg.get("USE_TRIPLET", True)
    cfg.LOSS.USE_LABEL_SMOOTH = loss_cfg.get("USE_LABEL_SMOOTH", True)
    cfg.LOSS.LABEL_SMOOTH_EPSILON = loss_cfg.get("LABEL_SMOOTH_EPSILON", 0.1)

    # Solver config
    cfg.SOLVER = CN()
    solver_cfg = yaml_cfg.get("SOLVER", {})
    cfg.SOLVER.BASE_LR = solver_cfg.get("BASE_LR", 0.0001)
    cfg.SOLVER.OPTIMIZER = solver_cfg.get("OPTIMIZER", "AdamW")
    cfg.SOLVER.MOMENTUM = solver_cfg.get("MOMENTUM", 0.9)
    cfg.SOLVER.WEIGHT_DECAY = solver_cfg.get("WEIGHT_DECAY", 0.05)
    cfg.SOLVER.WARMUP_EPOCHS = solver_cfg.get("WARMUP_EPOCHS", 50)
    cfg.SOLVER.MAX_EPOCHS = solver_cfg.get("MAX_EPOCHS", 100)
    cfg.SOLVER.STEPS = solver_cfg.get("STEPS", [40, 70])
    cfg.SOLVER.GAMMA = solver_cfg.get("GAMMA", 0.1)
    cfg.SOLVER.COSINE_AFTER_WARMUP = solver_cfg.get("COSINE_AFTER_WARMUP", True)
    cfg.SOLVER.COSINE_END_LR = solver_cfg.get("COSINE_END_LR", 1e-6)
    cfg.SOLVER.CHECKPOINT_PERIOD = solver_cfg.get("CHECKPOINT_PERIOD", 10)
    cfg.SOLVER.EVAL_PERIOD = solver_cfg.get("EVAL_PERIOD", 1)
    cfg.SOLVER.LOG_PERIOD = solver_cfg.get("LOG_PERIOD", 1)
    cfg.SOLVER.CLIP_GRADIENTS = CN()
    clip_cfg = solver_cfg.get("CLIP_GRADIENTS", {})
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = clip_cfg.get("ENABLED", True)
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = clip_cfg.get("CLIP_VALUE", 1.0)

    # Test config
    cfg.TEST = CN()
    test_cfg = yaml_cfg.get("TEST", {})
    cfg.TEST.BATCH_SIZE = test_cfg.get("BATCH_SIZE", 16)
    cfg.TEST.WEIGHT = test_cfg.get("WEIGHT", "")

    # Other configs
    cfg.NUM_GPUS = yaml_cfg.get("NUM_GPUS", 1)
    cfg.GPU_IDS = yaml_cfg.get("GPU_IDS", [0])
    cfg.OUTPUT_DIR = yaml_cfg.get("OUTPUT_DIR", "")
    cfg.SEED = yaml_cfg.get("SEED", 42)

    cfg.freeze()
    return cfg


def load_model(checkpoint_path, config_path, device='cuda'):
    """Load UniFormerV2 ReID model with config."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Infer num_classes and embed_dim from checkpoint
    state_dict = checkpoint['model_state_dict']
    num_classes = None
    embed_dim = None

    for key in state_dict.keys():
        if 'classifier.weight' in key:
            num_classes = state_dict[key].shape[0]
            embed_dim = state_dict[key].shape[1]
            logger.info(f"Detected {num_classes} classes from checkpoint")
            logger.info(f"Detected embed_dim={embed_dim} from checkpoint")
            break

    if num_classes is None:
        logger.warning("Could not infer num_classes from checkpoint, using config default")

    if embed_dim is None:
        logger.warning("Could not infer embed_dim from checkpoint, using config default")

    # Load config with inferred num_classes
    cfg = load_config(config_path, num_classes=num_classes)

    # Override parameters for inference
    cfg.defrost()

    # Override num_classes if inferred from checkpoint (config may have 0)
    if num_classes is not None and (cfg.MODEL.NUM_CLASSES == 0 or cfg.MODEL.NUM_CLASSES != num_classes):
        cfg.MODEL.NUM_CLASSES = num_classes
        logger.info(f"✓ Overriding config with inferred num_classes={num_classes}")

    # Override embed_dim if inferred from checkpoint (config may have wrong value)
    if embed_dim is not None and cfg.REID.EMBED_DIM != embed_dim:
        cfg.REID.EMBED_DIM = embed_dim
        logger.info(f"✓ Overriding config with inferred embed_dim={embed_dim}")

    # Disable pretrain loading (we're loading from checkpoint, not pretrained backbone)
    cfg.UNIFORMERV2.PRETRAIN = ""

    cfg.freeze()
    # Create model
    model = Uniformerv2ReID(cfg)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)

    logger.info(f"✓ Model loaded (epoch {checkpoint.get('epoch', 'NA')}, "
                f"Rank-1: {checkpoint.get('rank1', float('nan')):.1%})")
    return model, cfg


def process_video(video_path, T=16, out_size=224):
    """Uniformly sample T frames over the whole video and return tensor + RGB frames.

    Returns:
        tensor: Normalized tensor [1, C, T, H, W] with size out_size
        frames_224: List of RGB frames resized to 224x224 for model input
        frames_original: List of RGB frames in original size for visualization
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        raise RuntimeError("Video open failed")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Video info: frames={total}, fps={fps}, size={width}x{height}")

    frames_224 = []
    frames_original = []

    if total and total > 0:
        # Uniform indices across the entire duration
        idxs = np.linspace(0, max(total - 1, 0), num=T).astype(int)
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, frame = cap.read()
            if not ret:
                # Try sequential read if random seek failed
                ret2, frame2 = cap.read()
                if not ret2:
                    break
                frame = frame2

            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Keep original size for visualization
            frames_original.append(frame_rgb.copy())

            # Resize to 224x224 for model input
            frame_224 = cv2.resize(frame_rgb, (out_size, out_size))
            frames_224.append(frame_224)
    else:
        # Fallback: sequential read (some containers don't expose frame count)
        while len(frames_224) < T:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Keep original size for visualization
            frames_original.append(frame_rgb.copy())

            # Resize to 224x224 for model input
            frame_224 = cv2.resize(frame_rgb, (out_size, out_size))
            frames_224.append(frame_224)

    cap.release()

    if len(frames_224) == 0:
        raise RuntimeError("No frames extracted")

    # If fewer than T, pad with last real frame (log it)
    if len(frames_224) < T:
        logger.warning(f"Extracted only {len(frames_224)} frames; padding to {T} by repeating the last frame.")
        while len(frames_224) < T:
            frames_224.append(frames_224[-1].copy())
            frames_original.append(frames_original[-1].copy())

    # Normalize to tensor [1, C, T, H, W] using 224x224 frames
    frames_np = (np.stack(frames_224).astype(np.float32) / 255.0)
    mean = np.array([0.45, 0.45, 0.45]).reshape(1, 1, 1, 3)
    std = np.array([0.225, 0.225, 0.225]).reshape(1, 1, 1, 3)
    frames_np = (frames_np - mean) / std
    tensor = torch.from_numpy(frames_np).float().permute(3, 0, 1, 2).unsqueeze(0)

    return tensor, frames_224, frames_original  # frames are RGB uint8


def reshape_cam_3d(cam_flat, expect_T=8, expect_H=14, expect_W=14):
    """
    Reshape flat token CAM to [T, H, W] with heuristics and checks.
    Support multiple models (MViTv2, UniFormerV2, etc.)
    """
    num_tokens = cam_flat.shape[1]
    
    # Try multiple expected shapes in order
    candidate_shapes = [
        (expect_T, expect_H, expect_W),  # Use provided defaults first
        (8, 7, 7),    # MViTv2 with 7x7 spatial
        (8, 14, 14),  # MViTv2 with 14x14 spatial
        (16, 14, 14), # Full resolution
        (4, 7, 7),    # Downsampled MViTv2
    ]
    
    T, H, W = None, None, None
    
    # First try: exact match with known shapes
    for t_try, h_try, w_try in candidate_shapes:
        if num_tokens == t_try * h_try * w_try:
            T, H, W = t_try, h_try, w_try
            logger.info(f"Matched known shape: T={T}, H={H}, W={W}")
            break
    
    # Second try: factorize with preferred spatial sizes
    if T is None:
        logger.warning(f"Token count {num_tokens} doesn't match known shapes, inferring...")
        
        preferred_spatial = [7*7, 14*14, 8*8, 4*4, 16*16]  # Common spatial sizes
        
        for spatial_size in preferred_spatial:
            if num_tokens % spatial_size == 0:
                T = num_tokens // spatial_size
                H = W = int(spatial_size ** 0.5)
                logger.info(f"Inferred with preferred spatial: T={T}, H={H}, W={W}")
                break
        
        # Third try: general factorization (prefer T=8, then powers of 2)
        if T is None:
            found = False
            for T_try in [8, 4, 2, 16, 1]:
                if num_tokens % T_try == 0:
                    rem = num_tokens // T_try
                    # Try to make spatial dimensions square or close to square
                    H_try = int(round(rem ** 0.5))
                    for H_candidate in range(H_try, 0, -1):
                        if rem % H_candidate == 0:
                            W_candidate = rem // H_candidate
                            T, H, W = T_try, H_candidate, W_candidate
                            found = True
                            break
                if found:
                    break
            
            if found:
                logger.info(f"Inferred general factorization: T={T}, H={H}, W={W}")
            else:
                raise RuntimeError(f"Cannot factorize {num_tokens} into T*H*W")
    
    # Verify the factorization
    if T * H * W != num_tokens:
        raise RuntimeError(f"Shape inference error: {T}*{H}*{W}={T*H*W} != {num_tokens}")

    cam_3d = cam_flat[0].reshape(T, H, W)  # [T, H, W]
    cam_3d = np.maximum(cam_3d, 0)        # safety
    return cam_3d


def write_video_from_frames(frames_dir: Path, out_dir: Path, size=(224, 224), fps_out=10):
    """
    Assemble frames/*.jpg into a video, trying mp4 first then AVI as fallback.
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    video_path_mp4 = out_dir / "cam_video.mp4"
    video_path_avi = out_dir / "cam_video.avi"

    w, h = size[0], size[1]
    fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path_mp4), fourcc_mp4, fps_out, (w, h))

    use_avi = False
    if not out.isOpened():
        logger.warning("mp4v writer not opened, falling back to AVI/XVID.")
        fourcc_avi = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(video_path_avi), fourcc_avi, fps_out, (w, h))
        use_avi = True
        if not out.isOpened():
            raise RuntimeError("Failed to open any video writer (mp4v and XVID)")

    t = 0
    while True:
        frame_path = frames_dir / f"frame_{t:03d}.jpg"
        if not frame_path.exists():
            break
        frame = cv2.imread(str(frame_path))
        if frame is None:
            logger.warning(f"Failed to read {frame_path}, stopping.")
            break
        if (frame.shape[1], frame.shape[0]) != (w, h):
            frame = cv2.resize(frame, (w, h))
        out.write(frame)
        t += 1

    out.release()
    logger.info(f"✓ Saved video to {video_path_avi if use_avi else video_path_mp4}")


def main():
    # Fix random seeds for reproducible results
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="/home/zhang.13617/Desktop/zhang.13617/NBA/mask/Aaron Gordon/freethrow/003.mp4",
                        help="Video path to process")
    parser.add_argument("--checkpoint", default="/home/zhang.13617/Desktop/zhang.13617/NBA/ckpt/mask/best_model.pth",
                        help="Checkpoint path for UniFormerV2 model")
    parser.add_argument("--config", default=None,
                        help="Optional config file path. If omitted, default config is used with parameters inferred from checkpoint")
    parser.add_argument("--output", default="./cam_uniformer_output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target_id", type=int, default=-1, help="Target class id for CAM; -1 means use argmax")
    parser.add_argument("--frames", type=int, default=16, help="Number of frames to sample from video")
    parser.add_argument("--method", type=str, default="originalcam",
                        choices=["originalcam", "gradcam", "gradcam++", "layercam", "scorecam"],
                        help="CAM method to use (default: originalcam - gradient-free, simple, effective)")
    parser.add_argument("--scorecam_batch_size", type=int, default=32,
                        help="Batch size for ScoreCAM channel processing (default: 32)")
    parser.add_argument("--finer", action="store_true",
                        help="Enable finer mode: use target class score minus second-highest class score for better discrimination")
    parser.add_argument("--bn_folding", action="store_true",
                        help="Enable BN folding: compute score directly from pre-BN features to avoid gradient attenuation through BN layer")
    parser.add_argument("--train_mode", action="store_true",
                        help="Use training mode instead of eval mode for stronger gradients (predictions may be less accurate)")
    args = parser.parse_args()

    logger.info("="*80)
    logger.info(f"UniFormerV2 CAM Visualization - Method: {args.method.upper()}")
    logger.info("="*80)

    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True, parents=True)

    # Load model and video
    model, cfg = load_model(args.checkpoint, args.config, args.device)
    input_tensor, frames_224, frames_original = process_video(args.video, T=args.frames, out_size=224)
    input_tensor = input_tensor.to(args.device)

    # Get original frame size for output video
    original_height, original_width = frames_original[0].shape[:2]
    logger.info(f"Original frame size: {original_width}x{original_height}")

    logger.info(f"Input tensor: {tuple(input_tensor.shape)}")  # [1, C, T, H, W]
    num_frames = input_tensor.shape[2]

    # Setup CAM on last resblock of backbone transformer
    # UniFormerV2: model.backbone.transformer.resblocks[-1]
    target_layer = model.backbone.transformer.resblocks[-1]

    if args.method == "originalcam":
        cam_generator = OriginalCAM(model, target_layer)
        logger.info("Using Original CAM method (gradient-free, simple, effective)")
    elif args.method == "layercam":
        cam_generator = LayerCAM(model, target_layer)
        logger.info("Using LayerCAM method")
    elif args.method == "scorecam":
        cam_generator = ScoreCAM(model, target_layer, finer=args.finer)
        logger.info(f"Using ScoreCAM method (gradient-free){' with FINER mode' if args.finer else ''}")
    elif args.method == "gradcam++":
        cam_generator = GradCAMPlusPlus(model, target_layer)
        logger.info("Using GradCAM++ method (second-order gradients)")
        cam_generator = SimpleGradCAM(model, target_layer, finer=args.finer, bn_folding=args.bn_folding)
        logger.info(f"Using GradCAM method{' with FINER mode' if args.finer else ''}{' with BN Folding' if args.bn_folding else ''}")
    else:
        cam_generator = SimpleGradCAM(model, target_layer, finer=args.finer, bn_folding=args.bn_folding)
        logger.info(f"Using GradCAM method{' with FINER mode' if args.finer else ''}{' with BN Folding' if args.bn_folding else ''}")

    logger.info("Generating CAM...")

    # Gradient-free methods (originalcam, scorecam) don't need batch duplication or gradients
    is_gradient_free = args.method in ["originalcam", "scorecam"]

    if is_gradient_free:
        input_tensor_batched = input_tensor  # No need to duplicate
        logger.info("Using single input (gradient-free method)")
    else:
        # Duplicate input for better BatchNorm statistics
        input_tensor_batched = input_tensor.repeat(2, 1, 1, 1, 1)  # [2, C, T, H, W]
        logger.info(f"Batched input: {tuple(input_tensor_batched.shape)}")

    # ============================================================
    # Model Mode Selection
    # ============================================================
    if args.train_mode and not is_gradient_free:
        model.train()
        logger.info("Using TRAIN mode (stronger gradients, predictions may be less accurate)")
    else:
        model.eval()
        if is_gradient_free:
            logger.info("Using EVAL mode (accurate predictions, gradient-free CAM)")
        else:
            logger.info("Using EVAL mode (accurate predictions, may have weak gradients)")

    # Enable gradients only for gradient-based methods
    if not is_gradient_free:
        torch.set_grad_enabled(True)
        input_tensor_batched.requires_grad_(True)

    # Generate CAM
    if args.method == "scorecam":
        cam_flat = cam_generator.generate_cam(input_tensor_batched, target_id=args.target_id,
                                              batch_size=args.scorecam_batch_size)
    else:
        cam_flat = cam_generator.generate_cam(input_tensor_batched, target_id=args.target_id)
    if cam_flat is None:
        logger.error("Failed to generate CAM")
        return
    logger.info(f"✓ CAM generated: {cam_flat.shape}")

    # Reshape flat tokens -> [T, H, W]
    # For UniFormerV2 with 16-patch and 224x224 input: 14x14 spatial tokens
    # With temporal_downsample=False and T=16: expect 16x14x14
    cam_3d = reshape_cam_3d(cam_flat, expect_T=args.frames, expect_H=14, expect_W=14)  # [T,H,W]
    T_cam, H_cam, W_cam = cam_3d.shape
    logger.info(f"CAM reshaped to: T={T_cam}, H={H_cam}, W={W_cam}")

    # Overlay per-time CAM on per-frame image (use original size frames for high-res visualization)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    # Create directory for CAM activations (raw numpy arrays)
    cam_dir = out_dir / "cam_activations"
    cam_dir.mkdir(exist_ok=True)

    num_saved = 0
    for t in range(num_frames):
        frame = frames_original[t]  # RGB uint8 [H, W, C]

        # Map frame index to CAM time index
        cam_idx = int(round((t / max(num_frames - 1, 1)) * (T_cam - 1)))
        cam_2d = cam_3d[cam_idx]

        # Normalize each frame's CAM to [0, 1] independently
        cam_2d_norm = cam_2d - cam_2d.min()
        if cam_2d_norm.max() > 0:
            cam_2d_norm = cam_2d_norm / cam_2d_norm.max()

        # Resize CAM to original frame size and colorize
        cam_resized = cv2.resize(cam_2d_norm, (frame.shape[1], frame.shape[0]))

        # Save raw CAM activation (normalized, resized to original frame size)
        cam_npy_path = cam_dir / f"frame_{t:03d}_cam.npy"
        np.save(str(cam_npy_path), cam_resized)

        cam_uint8 = np.uint8(np.clip(cam_resized * 255.0, 0, 255))
        cam_colored = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)

        # Overlay
        vis = np.uint8(0.5 * cam_colored + 0.5 * frame)

        # Save frame
        out_path = frames_dir / f"frame_{t:03d}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        num_saved += 1

    logger.info(f"✓ Saved {num_saved} frames with CAM overlays to {frames_dir}")
    logger.info(f"✓ Saved {num_saved} CAM activations (.npy) to {cam_dir}")

    # Create video from saved frames (use original frame size)
    write_video_from_frames(frames_dir, out_dir, size=(original_width, original_height), fps_out=10)

    logger.info("="*80)
    logger.info("✓✓✓ SUCCESS! ✓✓✓")
    logger.info("="*80)


if __name__ == "__main__":
    main()
