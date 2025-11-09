"""
ScoreCAM: Gradient-free Class Activation Mapping

Reference: "Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks"
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import utils.logging as logging

logger = logging.get_logger(__name__)


class ScoreCAM:
    """
    ScoreCAM: Gradient-free class activation mapping using forward passing.
    Instead of using gradients, it measures the importance of each activation channel
    by masking the input with upsampled activations and observing the change in model confidence.
    """
    def __init__(self, model, target_layer, finer=False):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.finer = finer

        # Register forward hook to capture activations
        target_layer.register_forward_hook(self.save_activation)

    def save_activation(self, module, input, output):
        # MViT blocks may return (x, thw)
        if isinstance(output, tuple):
            activation_tensor = output[0]
        else:
            activation_tensor = output

        self.activations = activation_tensor.detach()  # No gradients needed for ScoreCAM
        logger.info(f"[ScoreCAM] Captured activation: {activation_tensor.shape}")

    def generate_cam(self, input_tensor, target_id=None, batch_size=32):
        """
        Generate ScoreCAM for input.

        Args:
            input_tensor: Input tensor [B, C, T, H, W]
            target_id: Target class id (-1 for predicted class)
            batch_size: Batch size for processing activation channels (to avoid OOM)
        """
        self.model.zero_grad()
        self.activations = None

        # Initial forward pass to get activations and predictions (NO GRADIENTS)
        with torch.no_grad():
            output = self.model(input_tensor)

        # Handle dict or tensor output
        logits = None
        if isinstance(output, dict):
            if 'cls_score' in output:
                logits = output['cls_score']  # [B, num_classes]
                logger.info(f"[ScoreCAM] Model output (cls_score) shape: {logits.shape}")
            else:
                logger.warning("[ScoreCAM] No 'cls_score' in model output; falling back to global_feat.")

        if self.activations is None:
            logger.error("[ScoreCAM] No activations captured during forward pass!")
            return None

        logger.info(f"[ScoreCAM] Activations shape: {self.activations.shape}")

        # Determine target class
        if logits is not None:
            # Show top-5 predictions
            probs = F.softmax(logits[0], dim=0)
            top5_probs, top5_ids = torch.topk(probs, min(5, logits.shape[1]))

            logger.info("="*70)
            logger.info("[ScoreCAM] Top-5 Predicted Classes:")
            for i, (pred_id, prob) in enumerate(zip(top5_ids, top5_probs)):
                logit_val = float(logits[0, pred_id])
                logger.info(f"  Rank {i+1}: Class {int(pred_id):3d} | Probability: {float(prob)*100:6.2f}% | Logit: {logit_val:+.3f}")
            logger.info("="*70)

            if target_id is None or target_id < 0 or target_id >= logits.shape[1]:
                chosen = int(torch.argmax(logits[0]).item())
            else:
                chosen = int(target_id)
            logger.info(f"[ScoreCAM] Using class {chosen} for CAM")
        else:
            logger.error("[ScoreCAM] Cannot generate ScoreCAM without classification scores!")
            return None

        # Handle different activation shapes
        # UniFormerV2: [L, NT, C] where L=1+HW, NT=batch*time
        # MViT: [B, N, C] where N=tokens
        input_sample = input_tensor[0:1]   # [1, C, T, H, W]
        T_input = input_sample.shape[2]
        
        if self.activations.dim() == 3 and self.activations.shape[0] > self.activations.shape[1]:
            # UniFormerV2 format: [L, NT, C]
            L, NT, C = self.activations.shape
            logger.info(f"[ScoreCAM] Detected UniFormerV2 format [L={L}, NT={NT}, C={C}]")
            
            # Infer N (batch) and T (time) from NT
            # Input was batched as [2, C, T, H, W], so N=2
            N = input_tensor.shape[0]
            T = NT // N
            logger.info(f"[ScoreCAM] Inferred N={N}, T={T} from NT={NT}")
            
            # Remove CLS token and reshape to include time dimension
            activations_no_cls = self.activations[1:, :, :]  # [HW, NT, C]
            HW = L - 1
            H = W = int(HW ** 0.5)
            
            # Reshape to [HW, N, T, C] then take first batch -> [HW, T, C]
            activations = activations_no_cls.reshape(HW, N, T, C)[:, 0, :, :]  # [HW, T, C]
            
            # Reshape to [H, W, T, C] then permute to [T, H, W, C]
            activations = activations.reshape(H, W, T, C).permute(2, 0, 1, 3)  # [T, H, W, C]
            
            # Flatten back to [T*H*W, C] for processing
            activations = activations.reshape(-1, C)
            logger.info(f"[ScoreCAM] After removing CLS and reshaping: {activations.shape} (T={T}, H={H}, W={W})")
            
        elif self.activations.dim() == 3:
            # MViT format: [B, N, C]
            logger.info(f"[ScoreCAM] Detected MViT format [B={self.activations.shape[0]}, N={self.activations.shape[1]}, C={self.activations.shape[2]}]")
            activations = self.activations[0]  # [N, C]
            # Remove CLS token if present (usually first token)
            try:
                if getattr(self.model.backbone, "cls_embed_on", False):
                    activations = activations[1:]
                    logger.info(f"[ScoreCAM] Removed CLS token, new shape: {activations.shape}")
            except Exception:
                pass
        else:
            raise ValueError(f"Unexpected activation shape: {self.activations.shape}")

        N_tokens, C_channels = activations.shape
        logger.info(f"[ScoreCAM] Processing {C_channels} activation channels from {N_tokens} tokens")

        # Reshape activations from [N, C] to [T, H, W, C]
        # Infer T, H, W from N_tokens
        # Support multiple models:
        # - MViTv2: typically 8x7x7=392 or 8x14x14=1568 tokens
        # - UniFormerV2: typically 32 tokens (after pooling)
        
        # Try multiple expected shapes in order
        candidate_shapes = [
            (8, 7, 7),    # MViTv2 with 7x7 spatial
            (8, 14, 14),  # MViTv2 with 14x14 spatial
            (16, 14, 14), # Full resolution
            (4, 7, 7),    # Downsampled MViTv2
        ]
        
        T_act, H_act, W_act = None, None, None
        
        # First try: exact match with known shapes
        for expect_T, expect_H, expect_W in candidate_shapes:
            if N_tokens == expect_T * expect_H * expect_W:
                T_act, H_act, W_act = expect_T, expect_H, expect_W
                logger.info(f"[ScoreCAM] Matched known shape: T={T_act}, H={H_act}, W={W_act}")
                break
        
        # Second try: factorize with preferred spatial sizes
        if T_act is None:
            logger.warning(f"[ScoreCAM] Token count {N_tokens} doesn't match known shapes, inferring...")
            
            preferred_spatial = [7*7, 14*14, 8*8, 4*4, 16*16]  # Common spatial sizes
            
            for spatial_size in preferred_spatial:
                if N_tokens % spatial_size == 0:
                    T_act = N_tokens // spatial_size
                    H_act = W_act = int(spatial_size ** 0.5)
                    logger.info(f"[ScoreCAM] Inferred with preferred spatial: T={T_act}, H={H_act}, W={W_act}")
                    break
            
            # Third try: general factorization (prefer T=8, then powers of 2)
            if T_act is None:
                found = False
                for T_try in [8, 4, 2, 16, 1]:
                    if N_tokens % T_try == 0:
                        rem = N_tokens // T_try
                        # Try to make spatial dimensions square or close to square
                        H_try = int(round(rem ** 0.5))
                        for H_candidate in range(H_try, 0, -1):
                            if rem % H_candidate == 0:
                                W_candidate = rem // H_candidate
                                T_act, H_act, W_act = T_try, H_candidate, W_candidate
                                found = True
                                break
                    if found:
                        break
                
                if found:
                    logger.info(f"[ScoreCAM] Inferred general factorization: T={T_act}, H={H_act}, W={W_act}")
                else:
                    raise RuntimeError(f"Cannot factorize {N_tokens} into T*H*W")
        
        # Verify the factorization
        if T_act * H_act * W_act != N_tokens:
            raise RuntimeError(f"Shape inference error: {T_act}*{H_act}*{W_act}={T_act*H_act*W_act} != {N_tokens}")

        # Reshape activations: [N, C] -> [T, H, W, C]
        activations_4d = activations.reshape(T_act, H_act, W_act, C_channels)  # [T, H, W, C]
        logger.info(f"[ScoreCAM] Activations reshaped to: {activations_4d.shape}")

        # Get input dimensions
        _, _, T_in, H_in, W_in = input_sample.shape

        # Compute importance scores for each channel
        weights = []

        # Process channels in batches to avoid OOM
        for start_idx in range(0, C_channels, batch_size):
            end_idx = min(start_idx + batch_size, C_channels)
            batch_weights = []

            for c in range(start_idx, end_idx):
                # Get activation map for channel c: [T, H, W]
                act_map = activations_4d[:, :, :, c].cpu().numpy()

                # Upsample to input size: [T, H, W] -> [T_in, H_in, W_in]
                # For temporal dimension, use nearest neighbor interpolation
                act_map_upsampled = np.zeros((T_in, H_in, W_in), dtype=np.float32)
                for t in range(T_in):
                    t_src = int(round((t / max(T_in - 1, 1)) * (T_act - 1)))
                    act_2d = act_map[t_src]  # [H, W]
                    act_2d_up = cv2.resize(act_2d, (W_in, H_in), interpolation=cv2.INTER_LINEAR)
                    act_map_upsampled[t] = act_2d_up

                # Normalize activation map to [0, 1]
                act_min = act_map_upsampled.min()
                act_max = act_map_upsampled.max()
                if act_max > act_min:
                    act_map_norm = (act_map_upsampled - act_min) / (act_max - act_min)
                else:
                    act_map_norm = np.zeros_like(act_map_upsampled)

                # Apply mask to input: [1, C, T, H, W] * [T, H, W]
                act_mask = torch.from_numpy(act_map_norm).float().to(input_sample.device)
                # Broadcast: [1, C, T, H, W] * [1, 1, T, H, W]
                masked_input = input_sample * act_mask.unsqueeze(0).unsqueeze(0)

                # Forward pass with masked input
                with torch.no_grad():
                    masked_output = self.model(masked_input)
                    if isinstance(masked_output, dict) and 'cls_score' in masked_output:
                        masked_logits = masked_output['cls_score']
                        # Get score for target class
                        if self.finer:
                            # Finer mode: use target class confidence minus second-highest class confidence
                            logits_sorted, indices_sorted = torch.sort(masked_logits[0], descending=True)
                            # Find second class (avoid choosing the same as target)
                            second_class = None
                            for idx in indices_sorted:
                                if int(idx.item()) != chosen:
                                    second_class = int(idx.item())
                                    break
                            if second_class is None:
                                second_class = chosen
                            score = float(masked_logits[0, chosen].cpu().item()) - float(masked_logits[0, second_class].cpu().item())
                        else:
                            score = float(masked_logits[0, chosen].cpu().item())
                    else:
                        logger.warning(f"[ScoreCAM] Channel {c}: No cls_score in output")
                        score = 0.0

                batch_weights.append(score)

            weights.extend(batch_weights)
            logger.info(f"[ScoreCAM] Processed channels {start_idx}-{end_idx-1}/{C_channels}")

        weights = np.array(weights)  # [C_channels]
        logger.info(f"[ScoreCAM] Weights stats: mean={weights.mean():.4f}, std={weights.std():.4f}, min={weights.min():.4f}, max={weights.max():.4f}")

        # Normalize weights
        if weights.max() > weights.min():
            weights = (weights - weights.min()) / (weights.max() - weights.min())
        else:
            logger.warning("[ScoreCAM] All weights are equal!")
            weights = np.ones_like(weights) / len(weights)

        # Compute weighted combination of activation maps
        # activations_4d: [T, H, W, C], weights: [C]
        cam_3d = np.zeros((T_act, H_act, W_act), dtype=np.float32)
        for c in range(C_channels):
            cam_3d += weights[c] * activations_4d[:, :, :, c].cpu().numpy()

        logger.info(f"[ScoreCAM] CAM before ReLU: mean={cam_3d.mean():.4f}, std={cam_3d.std():.4f}, min={cam_3d.min():.4f}, max={cam_3d.max():.4f}")

        # ReLU and normalize
        cam_3d = np.maximum(cam_3d, 0)
        logger.info(f"[ScoreCAM] CAM after ReLU: mean={cam_3d.mean():.4f}, std={cam_3d.std():.4f}, min={cam_3d.min():.4f}, max={cam_3d.max():.4f}")

        cam_3d = cam_3d - cam_3d.min()
        if cam_3d.max() > 0:
            cam_3d = cam_3d / cam_3d.max()
            logger.info(f"[ScoreCAM] CAM after normalization: mean={cam_3d.mean():.4f}, std={cam_3d.std():.4f}, min={cam_3d.min():.4f}, max={cam_3d.max():.4f}")
        else:
            logger.warning("[ScoreCAM] ⚠ WARNING: CAM max is 0!")

        # Return in same format as GradCAM: [1, N_tokens]
        return cam_3d.reshape(1, -1)
