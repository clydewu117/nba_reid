#!/usr/bin/env python3
"""
Batch Grad-CAM visualization pipeline for paired appearance/mask VideoMAEv2 clips.

Key extensions over `batch_videomaev2_cam.py`:
- Reads paired appearance/mask videos listed in the train/test split CSV (test split only).
- Uses the mask video to derive per-frame white-region masks; CAM overlays are restricted
  to those white regions for both modalities.
- Supports independent configs/checkpoints for appearance and mask models while sharing
  sampling metadata and output layout.
- Reuses utility functions from `batch_videomaev2_cam.py` to avoid duplication.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT / "nba_reid"))

import utils.logging as logging  # noqa: E402
from batch_videomaev2_cam import (  # noqa: E402
    SAMPLING_MODE_MAP,
    build_eval_transform,
    checkpoint_slug,
    ensure_metadata,
    instantiate_cam_generator,
    load_csv_rows,
    prepare_tensor_and_frames,
    read_all_frames_pyav,
    sample_frame_indices,
    set_deterministic,
    write_video_from_png,
)
from run_gradcam_videomaev2 import load_config, load_model, reshape_cam_3d  # noqa: E402

logger = logging.get_logger(__name__)


@dataclass
class PairedEntry:
    key: str
    mask_entry: Dict[str, str]
    appearance_entry: Optional[Dict[str, str]]
    sampling_sets: Dict[str, List[int]]
    mask_total_frames: int


def derive_pair_key(video_path: Path) -> str:
    """
    Produce a stable key identifying paired appearance/mask clips.

    The key is the relative path components that follow either "appearance" or "mask".
    If neither component is present, the video stem is used as a fallback.
    """
    parts = video_path.parts
    for marker in ("appearance", "mask"):
        if marker in parts:
            idx = parts.index(marker)
            return str(Path(*parts[idx + 1 :]))
    return video_path.stem


def stable_seed_from_key(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def select_sampling_sets(
    total_frames: int,
    num_frames: int,
    sampling_keys: Sequence[str],
    seed_key: str,
) -> Dict[str, List[int]]:
    """
    Deterministically select sampling indices for a given video using the shared helper.
    """
    rng_state = np.random.get_state()
    np.random.seed(stable_seed_from_key(seed_key))
    try:
        sampling_all = sample_frame_indices(total_frames, num_frames)
    finally:
        np.random.set_state(rng_state)
    sampling_sets = {
        name: sampling_all[name] for name in sampling_keys if name in sampling_all
    }
    if not sampling_sets:
        raise RuntimeError("No valid sampling modes selected for video.")
    return sampling_sets


def build_paired_entries(
    rows: Iterable[Dict[str, str]],
    sampling_keys: Sequence[str],
    num_frames: int,
) -> List[PairedEntry]:
    mask_entries: Dict[str, Dict[str, str]] = {}
    appearance_entries: Dict[str, Dict[str, str]] = {}

    for row in rows:
        video_path = Path(row.get("video_path", ""))
        if not video_path:
            continue
        key = derive_pair_key(video_path)
        video_type = row.get("video_type", "").lower()
        if video_type == "mask":
            mask_entries[key] = row
        elif video_type == "appearance":
            appearance_entries[key] = row

    missing_masks = sorted(set(appearance_entries) - set(mask_entries))
    if missing_masks:
        for key in missing_masks:
            row = appearance_entries[key]
            logger.warning(
                "Skipping appearance entry without matching mask: %s (%s)",
                key,
                row.get("video_path", ""),
            )

    paired: List[PairedEntry] = []
    for key, mask_entry in mask_entries.items():
        mask_path = Path(mask_entry["video_path"])
        if not mask_path.exists():
            logger.warning("Mask video not found, skipping: %s", mask_path)
            continue
        try:
            mask_frames = read_all_frames_pyav(mask_path)
        except Exception as exc:
            logger.error("Failed to read mask video %s: %s", mask_path, exc)
            continue
        total_frames = len(mask_frames)
        if total_frames == 0:
            logger.warning("Mask video has 0 frames, skipping: %s", mask_path)
            continue
        sampling_sets = select_sampling_sets(
            total_frames=total_frames,
            num_frames=num_frames,
            sampling_keys=sampling_keys,
            seed_key=key,
        )
        paired.append(
            PairedEntry(
                key=key,
                mask_entry=mask_entry,
                appearance_entry=appearance_entries.get(key),
                sampling_sets=sampling_sets,
                mask_total_frames=total_frames,
            )
        )
        del mask_frames
    return paired


def extract_white_region_mask(frame: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    frame_float = frame.astype(np.float32)
    if frame_float.max() > 1.0:
        frame_float /= 255.0
    if frame_float.ndim == 3:
        frame_gray = frame_float.mean(axis=2)
    else:
        frame_gray = frame_float
    mask = (frame_gray >= threshold).astype(np.float32)
    return mask


def resize_region_mask(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    import cv2

    h, w = target_hw
    if mask.shape != (h, w):
        resized = cv2.resize(
            mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST
        )
    else:
        resized = mask.astype(np.float32)
    return (resized > 0.5).astype(np.float32)


def compute_region_masks_from_frames(
    frames: Sequence[np.ndarray],
) -> List[np.ndarray]:
    return [extract_white_region_mask(frame) for frame in frames]


def save_frames_with_mask(
    frames_dir: Path,
    cam_dir: Path,
    frames_original: Sequence[np.ndarray],
    cam_3d: np.ndarray,
    region_masks: Optional[Sequence[np.ndarray]],
) -> None:
    import cv2

    frames_dir.mkdir(parents=True, exist_ok=True)
    cam_dir.mkdir(parents=True, exist_ok=True)

    num_frames = len(frames_original)
    if num_frames == 0:
        return
    t_cam = cam_3d.shape[0]

    for idx, frame in enumerate(frames_original):
        cam_idx = int(round((idx / max(num_frames - 1, 1)) * (t_cam - 1)))
        cam_2d = cam_3d[cam_idx]
        cam_norm = cam_2d - cam_2d.min()
        if cam_norm.max() > 0:
            cam_norm = cam_norm / cam_norm.max()

        cam_resized = cv2.resize(cam_norm, (frame.shape[1], frame.shape[0]))
        mask_bin = None
        if region_masks is not None:
            mask_raw = region_masks[min(idx, len(region_masks) - 1)]
            mask_bin = resize_region_mask(mask_raw, frame.shape[:2])
            cam_resized = cam_resized * mask_bin

        np.save(str(cam_dir / f"frame_{idx:03d}_cam.npy"), cam_resized)

        cam_uint8 = np.uint8(np.clip(cam_resized * 255.0, 0, 255))
        cam_colored = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)

        if mask_bin is not None:
            mask_rgb = mask_bin[:, :, None]
            cam_colored = (cam_colored * mask_rgb).astype(np.uint8)
            blended = (0.5 * cam_colored + 0.5 * frame).astype(np.uint8)
            overlay = np.where(mask_rgb > 0, blended, frame).astype(np.uint8)
        else:
            overlay = np.uint8(0.5 * cam_colored + 0.5 * frame)

        out_path = frames_dir / f"frame_{idx:03d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def run_cam_methods(
    model: torch.nn.Module,
    cam_generators: Mapping[str, object],
    methods: Sequence[str],
    args: argparse.Namespace,
    video_tensor: torch.Tensor,
    selected_frames: Sequence[np.ndarray],
    region_masks: Optional[Sequence[np.ndarray]],
    sampling_dir: Path,
) -> Tuple[int, int]:
    success, failure = 0, 0
    input_tensor = video_tensor.unsqueeze(0).to(args.device)
    original_height, original_width = selected_frames[0].shape[:2]

    for method in methods:
        method_dir = sampling_dir / method
        frames_dir = method_dir / "frames"
        cam_dir = method_dir / "cam_activations"
        method_dir.mkdir(parents=True, exist_ok=True)

        cam_generator = cam_generators[method]
        is_gradient_free = method in ("originalcam", "scorecam")

        if is_gradient_free:
            input_tensor_batched = input_tensor
        else:
            input_tensor_batched = input_tensor.repeat(2, 1, 1, 1, 1)

        try:
            model.eval()
            if is_gradient_free:
                torch.set_grad_enabled(False)
            else:
                torch.set_grad_enabled(True)
                input_tensor_batched.requires_grad_(True)

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
            logger.error("CAM generation returned None for method %s", method)
            failure += 1
            continue

        cam_3d = reshape_cam_3d(cam_flat, expect_T=8, expect_H=14, expect_W=14)

        try:
            save_frames_with_mask(frames_dir, cam_dir, selected_frames, cam_3d, region_masks)
            if not args.skip_video:
                write_video_from_png(
                    frames_dir,
                    method_dir,
                    size=(original_width, original_height),
                    fps_out=args.video_fps,
                )
            success += 1
        except Exception as exc:
            logger.error("Failed to save outputs for method %s: %s", method, exc)
            failure += 1

    return success, failure


def process_entry_generic(
    entry: Dict[str, str],
    frames: Sequence[np.ndarray],
    sampling_sets: Mapping[str, Sequence[int]],
    region_masks_map: Optional[Mapping[str, Sequence[np.ndarray]]],
    transform,
    model: torch.nn.Module,
    cam_generators: Mapping[str, object],
    methods: Sequence[str],
    args: argparse.Namespace,
    output_checkpoint_root: Path,
    extra_metadata: Optional[Dict[str, object]] = None,
) -> Tuple[int, int]:
    video_path = Path(entry["video_path"])
    identity = entry.get("identity_folder", entry.get("identity_display", "unknown"))
    clip_name = Path(entry.get("filename", video_path.stem)).stem

    video_root = output_checkpoint_root / identity / clip_name
    video_root.mkdir(parents=True, exist_ok=True)

    meta = {
        "video_path": str(video_path),
        "identity": entry.get("identity_display", ""),
        "shot_type": entry.get("shot_type", ""),
        "video_type": entry.get("video_type", ""),
        "total_frames": len(frames),
        "sample_indices": sampling_sets,
    }
    if extra_metadata:
        meta.update(extra_metadata)
    ensure_metadata(video_root, meta)

    success_total, failure_total = 0, 0

    for sampling_name, indices in sampling_sets.items():
        sampling_dir = video_root / sampling_name
        sampling_dir.mkdir(parents=True, exist_ok=True)

        try:
            video_tensor, selected_frames = prepare_tensor_and_frames(
                frames, indices, transform
            )
        except IndexError as exc:
            logger.error(
                "Sampling indices out of range for %s (%s): %s",
                video_path,
                sampling_name,
                exc,
            )
            failure_total += len(methods)
            continue

        if not selected_frames:
            logger.warning(
                "No frames selected for %s (%s), skipping sampling.",
                video_path,
                sampling_name,
            )
            failure_total += len(methods)
            continue

        region_masks = None
        if region_masks_map is not None and sampling_name in region_masks_map:
            masks = region_masks_map[sampling_name]
            if len(masks) != len(selected_frames):
                logger.warning(
                    "Region mask count mismatch for %s (%s); expected %d got %d.",
                    video_path,
                    sampling_name,
                    len(selected_frames),
                    len(masks),
                )
            else:
                region_masks = [
                    resize_region_mask(mask, frame.shape[:2])
                    for mask, frame in zip(masks, selected_frames)
                ]

        success, failure = run_cam_methods(
            model,
            cam_generators,
            methods,
            args,
            video_tensor,
            selected_frames,
            region_masks,
            sampling_dir,
        )
        success_total += success
        failure_total += failure

    return success_total, failure_total


def prepare_region_masks_for_pair(
    mask_entry: Dict[str, str],
    sampling_sets: Mapping[str, Sequence[int]],
) -> Tuple[Dict[str, List[np.ndarray]], List[np.ndarray]]:
    mask_path = Path(mask_entry["video_path"])
    frames = read_all_frames_pyav(mask_path)
    region_masks_map: Dict[str, List[np.ndarray]] = {}

    for sampling_name, indices in sampling_sets.items():
        selected_frames = [frames[idx] for idx in indices]
        region_masks_map[sampling_name] = compute_region_masks_from_frames(
            selected_frames
        )
    return region_masks_map, frames


def process_mask_entry(
    pair: PairedEntry,
    model: torch.nn.Module,
    cam_generators: Mapping[str, object],
    methods: Sequence[str],
    args: argparse.Namespace,
    transform,
    output_checkpoint_root: Path,
) -> Tuple[int, int]:
    mask_path = Path(pair.mask_entry["video_path"])
    if not mask_path.exists():
        logger.warning("Mask video missing on disk, skipping: %s", mask_path)
        return 0, 1

    try:
        region_masks_map, frames = prepare_region_masks_for_pair(
            pair.mask_entry, pair.sampling_sets
        )
    except Exception as exc:
        logger.error("Failed to prepare mask region data for %s: %s", mask_path, exc)
        return 0, 1

    try:
        success, failure = process_entry_generic(
            entry=pair.mask_entry,
            frames=frames,
            sampling_sets=pair.sampling_sets,
            region_masks_map=region_masks_map,
            transform=transform,
            model=model,
            cam_generators=cam_generators,
            methods=methods,
            args=args,
            output_checkpoint_root=output_checkpoint_root,
            extra_metadata={
                "paired_appearance_video_path": pair.appearance_entry.get("video_path", "")
                if pair.appearance_entry
                else "",
            },
        )
    finally:
        del frames

    return success, failure


def process_appearance_entry(
    pair: PairedEntry,
    model: torch.nn.Module,
    cam_generators: Mapping[str, object],
    methods: Sequence[str],
    args: argparse.Namespace,
    transform,
    output_checkpoint_root: Path,
) -> Tuple[int, int]:
    if not pair.appearance_entry:
        return 0, 0

    appearance_path = Path(pair.appearance_entry["video_path"])
    if not appearance_path.exists():
        logger.warning("Appearance video missing on disk, skipping: %s", appearance_path)
        return 0, 1

    mask_frames: Optional[List[np.ndarray]] = None
    try:
        region_masks_map, mask_frames = prepare_region_masks_for_pair(
            pair.mask_entry, pair.sampling_sets
        )
    except Exception as exc:
        logger.error(
            "Failed to prepare mask region data for appearance %s: %s",
            appearance_path,
            exc,
        )
        return 0, 1
    finally:
        if mask_frames is not None:
            del mask_frames

    try:
        appearance_frames = read_all_frames_pyav(appearance_path)
    except Exception as exc:
        logger.error("Failed to read appearance video %s: %s", appearance_path, exc)
        return 0, 1

    success, failure = process_entry_generic(
        entry=pair.appearance_entry,
        frames=appearance_frames,
        sampling_sets=pair.sampling_sets,
        region_masks_map=region_masks_map,
        transform=transform,
        model=model,
        cam_generators=cam_generators,
        methods=methods,
        args=args,
        output_checkpoint_root=output_checkpoint_root,
        extra_metadata={
            "paired_mask_video_path": pair.mask_entry.get("video_path", ""),
        },
    )

    del appearance_frames
    return success, failure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch VideoMAEv2 Grad-CAM generator for paired appearance/mask videos"
    )
    parser.add_argument(
        "--csv", required=True, type=Path, help="Path to train_test_split CSV"
    )
    parser.add_argument(
        "--appearance-checkpoints",
        required=True,
        nargs="+",
        type=Path,
        help="One or more checkpoints for the appearance model",
    )
    parser.add_argument(
        "--appearance-config",
        required=True,
        type=Path,
        help="YAML config for the appearance model",
    )
    parser.add_argument(
        "--mask-checkpoints",
        required=True,
        nargs="+",
        type=Path,
        help="One or more checkpoints for the mask model",
    )
    parser.add_argument(
        "--mask-config",
        required=True,
        type=Path,
        help="YAML config for the mask model",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./batch_cam_output"),
        help="Base output directory",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="VideoMAEv2_masked",
        help="Top-level model name directory under the output root",
    )
    parser.add_argument(
        "--appearance-subdir",
        type=str,
        default="appearance",
        help="Subdirectory name for appearance outputs",
    )
    parser.add_argument(
        "--mask-subdir",
        type=str,
        default="mask",
        help="Subdirectory name for mask outputs",
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"], help="Device to run on"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["originalcam"],
        choices=["originalcam", "scorecam", "gradcam", "gradcam++", "layercam"],
    )
    parser.add_argument(
        "--sampling",
        nargs="+",
        default=["uniform"],
        choices=["random", "uniform"],
        help="Sampling strategy(ies) per clip",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=16,
        help="Number of frames sampled per clip",
    )
    parser.add_argument(
        "--target-id",
        type=int,
        default=-1,
        help="Target class ID for CAM; -1 uses argmax",
    )
    parser.add_argument(
        "--scorecam-batch-size",
        type=int,
        default=64,
        help="Batch size for ScoreCAM perturbation evaluation",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=10,
        help="FPS for rendered CAM videos",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Skip writing mp4/avi videos, keep PNG frames only",
    )
    parser.add_argument(
        "--finer",
        action="store_true",
        help="Use finer score difference when supported (GradCAM/ScoreCAM variants)",
    )
    parser.add_argument(
        "--video-type",
        default=None,
        help="Optional filter for the video_type column",
    )
    parser.add_argument(
        "--shot-type",
        default=None,
        help="Optional filter for the shot_type column",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of paired videos to process",
    )
    return parser.parse_args()


def filter_entries(
    rows: Iterable[Dict[str, str]],
    video_type: Optional[str],
    shot_type: Optional[str],
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


def instantiate_cam_generators(
    model: torch.nn.Module,
    methods: Sequence[str],
    args: argparse.Namespace,
) -> Dict[str, object]:
    target_layer = model.backbone.blocks[-1]
    return {
        method: instantiate_cam_generator(model, target_layer, method, args.finer)
        for method in methods
    }


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
    if not entries:
        logger.warning("No matching test entries found in CSV.")
        return

    paired_entries = build_paired_entries(entries, sampling_keys, args.frames)
    if args.limit is not None:
        paired_entries = paired_entries[: args.limit]

    if not paired_entries:
        logger.warning("No paired entries available after filtering.")
        return

    mask_cfg = load_config(str(args.mask_config))
    mask_transform = build_eval_transform(mask_cfg.DATA.HEIGHT, mask_cfg.DATA.WIDTH)
    appearance_cfg = load_config(str(args.appearance_config))
    appearance_transform = build_eval_transform(
        appearance_cfg.DATA.HEIGHT, appearance_cfg.DATA.WIDTH
    )

    output_root = args.output_root / args.model_name
    mask_root = output_root / args.mask_subdir
    appearance_root = output_root / args.appearance_subdir
    mask_root.mkdir(parents=True, exist_ok=True)
    appearance_root.mkdir(parents=True, exist_ok=True)

    logger.info("Loaded %d paired entries from %s", len(paired_entries), args.csv)

    mask_total_success, mask_total_failure = 0, 0
    for ckpt_path in args.mask_checkpoints:
        if not ckpt_path.exists():
            logger.error("Mask checkpoint not found: %s", ckpt_path)
            continue

        ckpt_name = checkpoint_slug(ckpt_path)
        output_checkpoint_root = mask_root / ckpt_name
        output_checkpoint_root.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("Processing mask checkpoint: %s", ckpt_path)
        logger.info("=" * 80)

        model = load_model(str(ckpt_path), str(args.mask_config), device=args.device)
        cam_generators = instantiate_cam_generators(model, args.methods, args)

        success_ckpt, failure_ckpt = 0, 0
        for pair in paired_entries:
            s, f = process_mask_entry(
                pair,
                model,
                cam_generators,
                args.methods,
                args,
                mask_transform,
                output_checkpoint_root,
            )
            success_ckpt += s
            failure_ckpt += f

        logger.info(
            "Mask checkpoint %s: success=%d, failure=%d",
            ckpt_name,
            success_ckpt,
            failure_ckpt,
        )
        mask_total_success += success_ckpt
        mask_total_failure += failure_ckpt

        del model
        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    appearance_total_success, appearance_total_failure = 0, 0
    for ckpt_path in args.appearance_checkpoints:
        if not ckpt_path.exists():
            logger.error("Appearance checkpoint not found: %s", ckpt_path)
            continue

        ckpt_name = checkpoint_slug(ckpt_path)
        output_checkpoint_root = appearance_root / ckpt_name
        output_checkpoint_root.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("Processing appearance checkpoint: %s", ckpt_path)
        logger.info("=" * 80)

        model = load_model(
            str(ckpt_path), str(args.appearance_config), device=args.device
        )
        cam_generators = instantiate_cam_generators(model, args.methods, args)

        success_ckpt, failure_ckpt = 0, 0
        for pair in paired_entries:
            s, f = process_appearance_entry(
                pair,
                model,
                cam_generators,
                args.methods,
                args,
                appearance_transform,
                output_checkpoint_root,
            )
            success_ckpt += s
            failure_ckpt += f

        logger.info(
            "Appearance checkpoint %s: success=%d, failure=%d",
            ckpt_name,
            success_ckpt,
            failure_ckpt,
        )
        appearance_total_success += success_ckpt
        appearance_total_failure += failure_ckpt

        del model
        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("-" * 80)
    logger.info(
        "Mask totals: success=%d, failure=%d",
        mask_total_success,
        mask_total_failure,
    )
    logger.info(
        "Appearance totals: success=%d, failure=%d",
        appearance_total_success,
        appearance_total_failure,
    )
    if mask_total_failure + appearance_total_failure > 0:
        logger.info("Please review logs above for failures.")


if __name__ == "__main__":
    main()


