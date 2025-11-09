import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from models.mvitv2_reid import MViTReID
from slowfast.config.defaults import get_cfg as sf_get_cfg
from yacs.config import CfgNode
import utils.logging as logging
from cam_util import SimpleGradCAM, ScoreCAM, LayerCAM, GradCAMPlusPlus, OriginalCAM

logger = logging.get_logger(__name__)


def load_model(checkpoint_path, device='cuda'):
    """Load MViTv2 ReID model with config."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Infer num_classes from checkpoint
    state_dict = checkpoint['model_state_dict']
    num_classes = None
    for key in state_dict.keys():
        if 'classifier.weight' in key:
            num_classes = state_dict[key].shape[0]
            logger.info(f"Detected {num_classes} classes from checkpoint")
            break

    if num_classes is None:
        logger.warning("Could not infer num_classes from checkpoint, using default 99")
        num_classes = 99

    cfg = sf_get_cfg()
    cfg.MODEL.ARCH = "mvit"
    cfg.MODEL.MODEL_NAME = "MViT"
    cfg.MODEL.NUM_CLASSES = num_classes
    cfg.DATA.NUM_FRAMES = 16  # KEY: 16 frames!
    cfg.DATA.TRAIN_CROP_SIZE = 224
    cfg.DATA.TEST_CROP_SIZE = 224
    cfg.DATA.HEIGHT = 224
    cfg.DATA.INPUT_CHANNEL_NUM = [3]

    # MViT config
    cfg.MVIT.MODE = "conv"
    cfg.MVIT.CLS_EMBED_ON = True
    cfg.MVIT.PATCH_KERNEL = [3, 7, 7]
    cfg.MVIT.PATCH_STRIDE = [2, 4, 4]
    cfg.MVIT.PATCH_PADDING = [1, 3, 3]
    cfg.MVIT.EMBED_DIM = 96
    cfg.MVIT.NUM_HEADS = 1
    cfg.MVIT.MLP_RATIO = 4.0
    cfg.MVIT.QKV_BIAS = True
    cfg.MVIT.DROPPATH_RATE = 0.2
    cfg.MVIT.DEPTH = 16
    cfg.MVIT.NORM = "layernorm"
    cfg.MVIT.USE_ABS_POS = False
    cfg.MVIT.REL_POS_SPATIAL = True
    cfg.MVIT.REL_POS_TEMPORAL = True
    cfg.MVIT.SEP_POS_EMBED = False
    cfg.MVIT.USE_FIXED_SINCOS_POS = False
    cfg.MVIT.DIM_MUL_IN_ATT = True
    cfg.MVIT.RESIDUAL_POOLING = True
    cfg.MVIT.USE_MEAN_POOLING = False
    cfg.MVIT.DIM_MUL = [[1, 2.0], [3, 2.0], [14, 2.0]]
    cfg.MVIT.HEAD_MUL = [[1, 2.0], [3, 2.0], [14, 2.0]]
    cfg.MVIT.POOL_KVQ_KERNEL = [3, 3, 3]
    cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE = [1, 8, 8]
    cfg.MVIT.POOL_Q_STRIDE = [
        [0, 1, 1, 1], [1, 1, 2, 2], [2, 1, 1, 1], [3, 1, 2, 2],
        [4, 1, 1, 1], [5, 1, 1, 1], [6, 1, 1, 1], [7, 1, 1, 1],
        [8, 1, 1, 1], [9, 1, 1, 1], [10, 1, 1, 1], [11, 1, 1, 1],
        [12, 1, 1, 1], [13, 1, 1, 1], [14, 1, 2, 2], [15, 1, 1, 1],
    ]

    cfg.REID = CfgNode()
    cfg.REID.EMBED_DIM = 512
    cfg.REID.NECK_FEAT = "after"
    cfg.MVIT.PRETRAIN = ""
    cfg.MVIT.FROZEN = False

    model = MViTReID(cfg)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)

    logger.info(f"✓ Model loaded (epoch {checkpoint.get('epoch', 'NA')}, "
                f"Rank-1: {checkpoint.get('rank1', float('nan')):.1%})")
    return model


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


def reshape_cam_3d(cam_flat, expect_T=8, expect_H=7, expect_W=7):
    """
    Reshape flat token CAM to [T, H, W] with heuristics and checks.
    """
    num_tokens = cam_flat.shape[1]
    expected = expect_T * expect_H * expect_W
    if num_tokens == expected:
        T, H, W = expect_T, expect_H, expect_W
    else:
        logger.warning(f"Token count mismatch: got {num_tokens}, expected {expected}. Trying to infer T,H,W...")
        # Heuristic: prefer T=8 if divisible by 7*7
        if num_tokens % (expect_H * expect_W) == 0:
            T = num_tokens // (expect_H * expect_W)
            H, W = expect_H, expect_W
        else:
            # cube-ish fallback
            T = int(round(num_tokens ** (1/3)))
            rem = max(num_tokens // max(T, 1), 1)
            H = int(round(rem ** 0.5))
            W = max(rem // max(H, 1), 1)
            if T * H * W != num_tokens:
                raise RuntimeError(f"Cannot reshape CAM tokens: {num_tokens} != {T}*{H}*{W}")
        logger.info(f"Inferred CAM shape: T={T}, H={H}, W={W}")

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
    # Fix random seeds for reproducible results (important for Dropout/BatchNorm in training mode)
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="/home/zhang.13617/Desktop/zhang.13617/NBA/mask/Aaron Gordon/freethrow/000.mp4")
    parser.add_argument("--checkpoint", default="/home/zhang.13617/Desktop/zhang.13617/NBA/ckpt/mvit_app_model.pth")
    parser.add_argument("--output", default="./cam_output")
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
    args = parser.parse_args()

    logger.info("="*80)
    logger.info(f"MViT CAM Visualization - Method: {args.method.upper()}")
    logger.info("="*80)

    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True, parents=True)

    # Load model and video
    model = load_model(args.checkpoint, args.device)
    input_tensor, frames_224, frames_original = process_video(args.video, T=args.frames, out_size=224)
    input_tensor = input_tensor.to(args.device)

    # Get original frame size for output video
    original_height, original_width = frames_original[0].shape[:2]
    logger.info(f"Original frame size: {original_width}x{original_height}")

    logger.info(f"Input tensor: {tuple(input_tensor.shape)}")  # [1, C, T, H, W]
    num_frames = input_tensor.shape[2]

    # Setup CAM on last block
    # (Adjust here if you want a different layer)
    target_layer = model.backbone.blocks[-1]

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
    else:
        cam_generator = SimpleGradCAM(model, target_layer, finer=args.finer)
        logger.info(f"Using GradCAM method{' with FINER mode' if args.finer else ''}")

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
    # EVAL MODE - Accurate predictions
    # ============================================================
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
    # Expected layout for late-stage tokens is about 8 x 7 x 7; infer if mismatched
    cam_3d = reshape_cam_3d(cam_flat, expect_T=8, expect_H=7, expect_W=7)  # [T,H,W]
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