#!/usr/bin/env python3
"""
Batch Grad-CAM visualization pipeline for MViTv2 ReID models.

Highlights:
- Uses PyAV frame extraction consistent with the training dataloader.
- Processes all "test" entries from a CSV split file.
- Runs both random-segment and uniform-segment sampling strategies for each clip.
- Supports multiple CAM methods (OriginalCAM, ScoreCAM, GradCAM, etc.).
- Optional train mode toggle for stronger gradients with BN-sensible duplication.
- Saves PNG overlays, raw CAM activations, and optional mp4 videos under
  `output_root/model_name/checkpoint/video_type/identity/clip/sampling/method`.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import av
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT / "nba_reid"))

import utils.logging as logging  # noqa: E402
from cam_util import (  # noqa: E402
    GradCAMPlusPlus,
    LayerCAM,
    OriginalCAM,
    ScoreCAM,
    SimpleGradCAM,
)
from run_gradcam_mvit import load_model, reshape_cam_3d  # noqa: E402

logger = logging.get_logger(__name__)

SAMPLING_MODE_MAP = {
    "random": "random_segments",
    "uniform": "uniform_segments",
}

def set_deterministic(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_eval_transform(height: int, width: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def read_all_frames_pyav(video_path: Path) -> List[np.ndarray]:
    try:
        container = av.open(str(video_path))
    except Exception as exc:  # pragma: no cover - error path
        raise RuntimeError(f"Failed to open video via PyAV: {video_path}") from exc

    frames: List[np.ndarray] = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()

    if not frames:
        raise RuntimeError(f"Video has 0 frames: {video_path}")
    return frames


def _uniform_segment_sampling_range(
    start_idx: int, end_idx: int, num_frames: int
) -> List[int]:
    if start_idx >= end_idx:
        return [start_idx] * num_frames

    indices = np.arange(start_idx, end_idx)
    available = len(indices)
    if available == 0:
        return [start_idx] * num_frames

    num_pads = num_frames - (available % num_frames)
    if num_pads != num_frames:
        pad_value = end_idx - 1
        indices = np.concatenate(
            [indices, np.full(num_pads, pad_value, dtype=np.int64)]
        )

    segments = np.array_split(indices, num_frames)
    sampled = [int(segment[0]) if len(segment) > 0 else start_idx for segment in segments]
    return sampled


def _get_sampling_range(total_frames: int) -> Tuple[int, int]:
    return 0, max(total_frames, 1)


def _random_segment_sampling(total_frames: int, num_frames: int) -> List[int]:
    start_idx, end_idx = _get_sampling_range(total_frames)
    indices = np.arange(start_idx, end_idx)
    available = len(indices)

    if available == 0:
        return [start_idx] * num_frames

    num_pads = num_frames - (available % num_frames)
    if num_pads != num_frames:
        pad_value = end_idx - 1
        indices = np.concatenate([indices, np.full(num_pads, pad_value, dtype=np.int64)])

    segments = np.array_split(indices, num_frames)
    sampled = [int(np.random.choice(segment)) for segment in segments if len(segment) > 0]

    if len(sampled) < num_frames:
        sampled.extend([sampled[-1]] * (num_frames - len(sampled)))

    sampled = [min(max(idx, 0), max(total_frames - 1, 0)) for idx in sampled]
    return sampled


def _uniform_segment_sampling(total_frames: int, num_frames: int) -> List[int]:
    start_idx, end_idx = _get_sampling_range(total_frames)
    indices = _uniform_segment_sampling_range(start_idx, end_idx, num_frames)
    if len(indices) < num_frames:
        indices.extend([indices[-1]] * (num_frames - len(indices)))
    indices = [min(max(idx, 0), max(total_frames - 1, 0)) for idx in indices]
    return indices


def sample_frame_indices(total_frames: int, num_frames: int) -> Dict[str, List[int]]:
    return {
        "random_segments": _random_segment_sampling(total_frames, num_frames),
        "uniform_segments": _uniform_segment_sampling(total_frames, num_frames),
    }


def prepare_tensor_and_frames(
    frames: Sequence[np.ndarray],
    indices: Sequence[int],
    transform: T.Compose,
) -> Tuple[torch.Tensor, List[np.ndarray]]:
    processed: List[torch.Tensor] = []
    selected_frames: List[np.ndarray] = []

    for idx in indices:
        chosen = frames[idx]
        selected_frames.append(chosen)
        pil_img = Image.fromarray(chosen)
        processed.append(transform(pil_img))

    video_tensor = torch.stack(processed, dim=0).permute(1, 0, 2, 3)
    return video_tensor, selected_frames


def instantiate_cam_generator(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    method: str,
    finer: bool,
    bn_folding: bool,
):
    if method == "originalcam":
        generator = OriginalCAM(model, target_layer)
    elif method == "layercam":
        generator = LayerCAM(model, target_layer)
    elif method == "scorecam":
        generator = ScoreCAM(model, target_layer, finer=finer)
    elif method == "gradcam++":
        generator = GradCAMPlusPlus(model, target_layer)
    else:
        generator = SimpleGradCAM(model, target_layer, finer=finer, bn_folding=bn_folding)
    return generator


def save_frames_as_png(
    frames_dir: Path,
    cam_dir: Path,
    frames_original: Sequence[np.ndarray],
    cam_3d: np.ndarray,
) -> None:
    import cv2

    frames_dir.mkdir(parents=True, exist_ok=True)
    cam_dir.mkdir(parents=True, exist_ok=True)

    num_frames = len(frames_original)
    t_cam = cam_3d.shape[0]

    for t in range(num_frames):
        frame = frames_original[t]
        cam_idx = int(round((t / max(num_frames - 1, 1)) * (t_cam - 1)))
        cam_2d = cam_3d[cam_idx]
        cam_norm = cam_2d - cam_2d.min()
        if cam_norm.max() > 0:
            cam_norm = cam_norm / cam_norm.max()

        cam_resized = cv2.resize(cam_norm, (frame.shape[1], frame.shape[0]))
        np.save(str(cam_dir / f"frame_{t:03d}_cam.npy"), cam_resized)

        cam_uint8 = np.uint8(np.clip(cam_resized * 255.0, 0, 255))
        cam_colored = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)

        overlay = np.uint8(0.5 * cam_colored + 0.5 * frame)
        out_path = frames_dir / f"frame_{t:03d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def write_video_from_png(
    frames_dir: Path, out_dir: Path, size: Tuple[int, int], fps_out: int = 10
) -> None:
    import cv2

    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        logger.warning(f"No PNG frames found in {frames_dir}, skipping video export.")
        return

    w, h = size
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = out_dir / "cam_video.mp4"
    avi_path = out_dir / "cam_video.avi"

    fourcc_mp4 = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4_path), fourcc_mp4, fps_out, (w, h))
    use_avi = False

    if not writer.isOpened():
        logger.warning("mp4v writer failed, falling back to XVID/AVI.")
        fourcc_avi = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(avi_path), fourcc_avi, fps_out, (w, h))
        use_avi = True
        if not writer.isOpened():
            raise RuntimeError("Failed to open video writer for both MP4 and AVI formats.")

    for frame_path in frame_files:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            logger.warning(f"Unable to read frame {frame_path}, stopping video assembly.")
            break
        if (frame.shape[1], frame.shape[0]) != (w, h):
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)

    writer.release()
    logger.info(f"Saved CAM video to {avi_path if use_avi else mp4_path}")


def load_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def ensure_metadata(video_root: Path, meta: Dict[str, object]) -> None:
    meta_path = video_root / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def checkpoint_slug(path: Path) -> str:
    """Return a filesystem-safe slug for a checkpoint path.

    Requirement: completely drop the filename like "best_model.pth" from the
    output folder naming. We therefore use only the immediate parent directory
    name as the slug, which distinguishes checkpoints living in different
    experiment folders.

    Fallback: if parent is missing, use the stem; if the stem equals
    "best_model", fall back to the grandparent name or a generic "checkpoint".
    """
    stem = path.stem if path.suffix else path.name
    parent_name = path.parent.name if path.parent is not None else ""
    if parent_name:
        base = parent_name
    else:
        # no parent; avoid returning a meaningless "best_model"
        if stem and stem.lower() != "best_model":
            base = stem
        else:
            gp = path.parent.parent.name if path.parent and path.parent.parent else ""
            base = gp or "checkpoint"

    # Replace any problematic characters just in case
    safe = (
        base.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )
    return safe


def process_video_entry(
    entry: Dict[str, str],
    model: torch.nn.Module,
    cam_generators: Dict[str, object],
    methods: Sequence[str],
    args: argparse.Namespace,
    transform: T.Compose,
    output_checkpoint_root: Path,
) -> Tuple[int, int]:
    video_path = Path(entry["video_path"])
    if not video_path.exists():
        logger.warning(f"Video not found, skipping: {video_path}")
        return 0, 1

    try:
        frames = read_all_frames_pyav(video_path)
    except Exception as exc:
        logger.error(f"Failed to read {video_path}: {exc}")
        return 0, 1

    total_frames = len(frames)
    sampling_sets = sample_frame_indices(total_frames, args.frames)
    sampling_sets = {name: sampling_sets[name] for name in args.sampling_keys if name in sampling_sets}

    if not sampling_sets:
        logger.error("No valid sampling modes selected; skipping video.")
        return 0, 1

    video_root = (
        output_checkpoint_root
        / entry.get("identity_folder", entry.get("identity_display", "unknown"))
        / Path(entry.get("filename", video_path.stem)).stem
    )
    video_root.mkdir(parents=True, exist_ok=True)

    ensure_metadata(
        video_root,
        {
            "video_path": str(video_path),
            "identity": entry.get("identity_display", ""),
            "shot_type": entry.get("shot_type", ""),
            "video_type": entry.get("video_type", ""),
            "total_frames": total_frames,
            "sample_indices": sampling_sets,
        },
    )

    success = 0
    failure = 0

    for sampling_name, indices in sampling_sets.items():
        sampling_dir = video_root / sampling_name
        sampling_dir.mkdir(parents=True, exist_ok=True)

        video_tensor, selected_frames = prepare_tensor_and_frames(frames, indices, transform)
        input_tensor = video_tensor.unsqueeze(0).to(args.device)
        original_height, original_width = selected_frames[0].shape[:2]

        for method in methods:
            cam_generator = cam_generators[method]
            method_dir = sampling_dir / method
            frames_dir = method_dir / "frames"
            cam_dir = method_dir / "cam_activations"
            method_dir.mkdir(parents=True, exist_ok=True)

            is_gradient_free = method in ("originalcam", "scorecam")

            if args.train_mode and not is_gradient_free:
                model.train()
            else:
                model.eval()

            if is_gradient_free:
                input_tensor_batched = input_tensor
            else:
                input_tensor_batched = input_tensor.repeat(2, 1, 1, 1, 1)

            try:
                if not is_gradient_free:
                    torch.set_grad_enabled(True)
                    input_tensor_batched.requires_grad_(True)
                else:
                    torch.set_grad_enabled(False)

                if method == "scorecam":
                    cam_flat = cam_generator.generate_cam(
                        input_tensor_batched,
                        target_id=args.target_id,
                        batch_size=args.scorecam_batch_size,
                    )
                else:
                    cam_flat = cam_generator.generate_cam(
                        input_tensor_batched, target_id=args.target_id
                    )
            finally:
                torch.set_grad_enabled(False)

            if cam_flat is None:
                logger.error(
                    f"CAM generation returned None for {video_path} [{method}] ({sampling_name})"
                )
                failure += 1
                continue

            try:
                cam_3d = reshape_cam_3d(cam_flat, expect_T=8, expect_H=7, expect_W=7)
                save_frames_as_png(frames_dir, cam_dir, selected_frames, cam_3d)
                if not args.skip_video:
                    write_video_from_png(
                        frames_dir,
                        method_dir,
                        size=(original_width, original_height),
                        fps_out=args.video_fps,
                    )
                success += 1
            except Exception as exc:
                logger.error(
                    f"Failed to save outputs for {video_path} [{method}] ({sampling_name}): {exc}"
                )
                failure += 1

    return success, failure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch MViT Grad-CAM generator")
    parser.add_argument("--csv", required=True, type=Path, help="Path to train_test_split CSV")
    parser.add_argument("--checkpoints", required=True, nargs="+", type=Path, help="Checkpoint paths")
    parser.add_argument("--output-root", type=Path, default=Path("./batch_cam_output"),
                        help="Base output directory")
    parser.add_argument("--model-name", type=str, default="MViTv2",
                        help="Model name folder inserted above checkpoint level")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--methods", nargs="+", default=["originalcam", "scorecam"],
                        choices=["originalcam", "scorecam", "gradcam", "gradcam++", "layercam"])
    parser.add_argument("--sampling", nargs="+", default=["uniform"],
                        choices=["random", "uniform"],
                        help="Sampling strategy(ies) to use per clip (default: random uniform)")
    parser.add_argument("--modality", choices=["appearance", "mask"], default="appearance",
                        help="Select appearance or mask")
    parser.add_argument("--frames", type=int, default=16, help="Number of frames sampled per clip")
    parser.add_argument("--target-id", type=int, default=-1, help="Target class ID for CAM; -1 uses argmax")
    parser.add_argument("--scorecam-batch-size", type=int, default=64,
                        help="Batch size for ScoreCAM perturbation evaluation")
    parser.add_argument("--video-fps", type=int, default=10, help="FPS for rendered CAM videos")
    parser.add_argument("--skip-video", action="store_true", help="Skip writing mp4/avi videos")
    parser.add_argument("--finer", action="store_true",
                        help="Use finer score difference when supported (GradCAM/ScoreCAM variants)")
    parser.add_argument("--train-mode", action="store_true",
                        help="Use model.train() for gradient-based methods (stronger gradients)")
    parser.add_argument("--bn-folding", action="store_true",
                        help="Enable BN folding for GradCAM variants")
    parser.add_argument("--video-type", default=None, help="Optional filter for video_type column")
    parser.add_argument("--shot-type", default=None, help="Optional filter for shot_type column")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of videos")
    return parser.parse_args()


def filter_entries(
    rows: Iterable[Dict[str, str]], video_type: str | None, shot_type: str | None
) -> List[Dict[str, str]]:
    selected: List[Dict[str, str]] = []
    for row in rows:
        if row.get("split", "").lower() != "test":
            continue
        if video_type and row.get("video_type") != video_type:
            continue
        if shot_type and row.get("shot_type") != shot_type:
            continue
        selected.append(row)
    return selected


def main() -> None:
    logging.setup_logging()
    set_deterministic()
    args = parse_args()

    sampling_keys: List[str] = []
    for mode in args.sampling:
        key = SAMPLING_MODE_MAP[mode]
        if key not in sampling_keys:
            sampling_keys.append(key)
    args.sampling_keys = sampling_keys

    csv_rows = load_csv_rows(args.csv)
    entries = filter_entries(csv_rows, args.video_type, args.shot_type)
    if args.limit is not None:
        entries = entries[: args.limit]

    entries = [row for row in entries if row.get("video_type") == args.modality]

    if not entries:
        logger.warning("No test entries matched the provided filters.")
        return

    # MViT expects 224x224 frames by default
    height = 224
    width = 224
    transform = build_eval_transform(height, width)

    logger.info(f"Loaded {len(entries)} test entries from {args.csv}")

    total_success = 0
    total_failure = 0
    model_root = args.output_root / args.model_name
    model_root.mkdir(parents=True, exist_ok=True)

    for ckpt_path in args.checkpoints:
        if not ckpt_path.exists():
            logger.error(f"Checkpoint not found: {ckpt_path}")
            continue

        ckpt_name = checkpoint_slug(ckpt_path)
        output_checkpoint_root = model_root / ckpt_name
        output_checkpoint_root.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info(f"Processing checkpoint: {ckpt_path}")
        logger.info("=" * 80)

        model = load_model(str(ckpt_path), device=args.device)
        target_layer = model.backbone.blocks[-1]

        cam_generators = {
            method: instantiate_cam_generator(
                model, target_layer, method, args.finer, args.bn_folding
            )
            for method in args.methods
        }

        success_ckpt = 0
        failure_ckpt = 0

        for entry in entries:
            s, f = process_video_entry(
                entry,
                model,
                cam_generators,
                args.methods,
                args,
                transform,
                output_checkpoint_root,
            )
            success_ckpt += s
            failure_ckpt += f

        logger.info(
            f"Checkpoint {ckpt_name}: success={success_ckpt}, failure={failure_ckpt}"
        )
        total_success += success_ckpt
        total_failure += failure_ckpt

        del model
        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("-" * 80)
    logger.info(f"Completed batch CAM generation. Success={total_success}, Failure={total_failure}")
    if total_failure > 0:
        logger.info("Please review the logs above for failures.")


if __name__ == "__main__":
    main()

