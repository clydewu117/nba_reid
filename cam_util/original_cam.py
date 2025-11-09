"""
Original CAM (Class Activation Mapping) - Gradient-free method

Reference: "Learning Deep Features for Discriminative Localization" (Zhou et al., CVPR 2016)

Key advantages:
- No gradients needed (works perfectly in eval mode)
- Not affected by BatchNorm layers
- Simple and efficient
"""

import numpy as np
import torch
import torch.nn.functional as F
import utils.logging as logging

logger = logging.get_logger(__name__)


class OriginalCAM:
    """
    Original CAM implementation for vision transformers.

    Requires:
    - Access to spatial features before global pooling
    - Linear classifier weights

    Formula:
    CAM_c(x,y) = Σ_k w_k^c · A_k(x,y)

    where:
    - A_k(x,y) is activation of channel k at position (x,y)
    - w_k^c is weight of channel k for class c
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.spatial_features = None

        # Register forward hook to capture spatial features
        target_layer.register_forward_hook(self.save_spatial_features)
        logger.info("[OriginalCAM] Initialized - gradient-free method")

    def save_spatial_features(self, module, input, output):
        """Capture spatial features before pooling."""
        # Transformer output: [L, NT, C] where L=1+H*W
        if isinstance(output, tuple):
            features = output[0]
        else:
            features = output

        self.spatial_features = features.detach()  # No gradients needed!
        logger.info(f"[OriginalCAM] Captured spatial features: {features.shape}")

    def generate_cam(self, input_tensor, target_id=None):
        """
        Generate CAM without gradients.

        Args:
            input_tensor: Input tensor [B, C, T, H, W]
            target_id: Target class id (-1 for predicted class)
        """
        self.spatial_features = None

        # Forward pass (no gradients needed!)
        with torch.no_grad():
            output = self.model(input_tensor)

        # Handle dict or tensor output
        logits = None
        if isinstance(output, dict):
            if 'cls_score' in output:
                logits = output['cls_score']  # [B, num_classes]
                logger.info(f"[OriginalCAM] Model output (cls_score) shape: {logits.shape}")
            else:
                logger.error("[OriginalCAM] No 'cls_score' in model output!")
                return None
        else:
            logger.error("[OriginalCAM] Model output is not a dict!")
            return None

        if self.spatial_features is None:
            logger.error("[OriginalCAM] No spatial features captured!")
            return None

        logger.info(f"[OriginalCAM] Spatial features shape: {self.spatial_features.shape}")

        # Show top-5 predictions
        probs = F.softmax(logits[0], dim=0)
        top5_probs, top5_ids = torch.topk(probs, min(5, logits.shape[1]))

        logger.info("="*70)
        logger.info("[OriginalCAM] Top-5 Predicted Classes:")
        for i, (pred_id, prob) in enumerate(zip(top5_ids, top5_probs)):
            logit_val = float(logits[0, pred_id])
            logger.info(f"  Rank {i+1}: Class {int(pred_id):3d} | Probability: {float(prob)*100:6.2f}% | Logit: {logit_val:+.3f}")
        logger.info("="*70)

        # Determine target class
        if target_id is None or target_id < 0 or target_id >= logits.shape[1]:
            chosen = int(torch.argmax(logits[0]).item())
        else:
            chosen = int(target_id)
        logger.info(f"[OriginalCAM] Using class {chosen} for CAM")

        # Get classifier weights
        if not hasattr(self.model, 'reid_head') or not hasattr(self.model.reid_head, 'classifier'):
            logger.error("[OriginalCAM] Cannot find reid_head.classifier!")
            return None

        classifier = self.model.reid_head.classifier
        W = classifier.weight  # [num_classes, embed_dim]

        # Get weights for target class
        w_c = W[chosen]  # [embed_dim]
        logger.info(f"[OriginalCAM] Classifier weights for class {chosen}: shape={w_c.shape}")
        logger.info(f"[OriginalCAM] Weight stats: mean={w_c.mean():.4f}, std={w_c.std():.4f}, min={w_c.min():.4f}, max={w_c.max():.4f}")

        # Handle different spatial feature shapes
        # UniFormerV2: [L, NT, C] where L=1+HW, NT=batch*time
        if self.spatial_features.dim() == 3 and self.spatial_features.shape[0] > self.spatial_features.shape[1]:
            # UniFormerV2 format: [L, NT, C]
            L, NT, C = self.spatial_features.shape
            logger.info(f"[OriginalCAM] Detected UniFormerV2 format [L={L}, NT={NT}, C={C}]")

            # Infer batch size and time
            N = input_tensor.shape[0]
            T = NT // N
            logger.info(f"[OriginalCAM] Inferred N={N}, T={T} from NT={NT}")

            # Remove CLS token (first token)
            spatial_no_cls = self.spatial_features[1:, :, :]  # [HW, NT, C]
            HW = L - 1
            H = W = int(HW ** 0.5)
            logger.info(f"[OriginalCAM] Spatial resolution: H={H}, W={W}")

            # Reshape to [HW, N, T, C] then take first batch -> [HW, T, C]
            spatial = spatial_no_cls.reshape(HW, N, T, C)[:, 0, :, :]  # [HW, T, C]

            # Reshape to [H, W, T, C] then permute to [T, H, W, C]
            spatial = spatial.reshape(H, W, T, C).permute(2, 0, 1, 3)  # [T, H, W, C]

            logger.info(f"[OriginalCAM] Reshaped spatial features: {spatial.shape} (T={T}, H={H}, W={W})")

        elif self.spatial_features.dim() == 3:
            # MViT format: [B, N, C]
            logger.info(f"[OriginalCAM] Detected MViT format [B={self.spatial_features.shape[0]}, N={self.spatial_features.shape[1]}, C={self.spatial_features.shape[2]}]")
            spatial = self.spatial_features[0]  # [N, C]

            # Remove CLS token if present (usually first token)
            try:
                if getattr(self.model.backbone, "cls_embed_on", False):
                    spatial = spatial[1:]
                    logger.info(f"[OriginalCAM] Removed CLS token, new shape: {spatial.shape}")
            except Exception:
                pass

            # Infer temporal and spatial dimensions
            N_tokens = spatial.shape[0]

            # Try common MViT shapes: T*H*W
            candidate_shapes = [
                (8, 7, 7),    # MViTv2 typical: 8x7x7=392
                (16, 7, 7),   # 16x7x7=784
                (8, 14, 14),  # 8x14x14=1568
                (4, 7, 7),    # 4x7x7=196
                (1, 14, 14),  # Single frame 14x14=196
                (1, 7, 7),    # Single frame 7x7=49
            ]

            T, H, W = None, None, None
            for t_try, h_try, w_try in candidate_shapes:
                if N_tokens == t_try * h_try * w_try:
                    T, H, W = t_try, h_try, w_try
                    logger.info(f"[OriginalCAM] Matched MViT shape: T={T}, H={H}, W={W}")
                    break

            if T is None:
                # Fallback: try to factorize
                logger.warning(f"[OriginalCAM] Token count {N_tokens} doesn't match known MViT shapes, inferring...")

                # Prefer 7x7 spatial
                if N_tokens % 49 == 0:
                    T = N_tokens // 49
                    H = W = 7
                elif N_tokens % 196 == 0:
                    T = N_tokens // 196
                    H = W = 14
                else:
                    # General factorization
                    T = int(round(N_tokens ** (1/3)))
                    rem = max(N_tokens // max(T, 1), 1)
                    H = int(round(rem ** 0.5))
                    W = max(rem // max(H, 1), 1)
                    if T * H * W != N_tokens:
                        raise RuntimeError(f"Cannot factorize {N_tokens} into T*H*W")
                logger.info(f"[OriginalCAM] Inferred MViT shape: T={T}, H={H}, W={W}")

            spatial = spatial.reshape(T, H, W, -1)  # [T, H, W, C]
        else:
            raise ValueError(f"Unexpected spatial features shape: {self.spatial_features.shape}")

        # ========== Apply CAM: fold BN+Linear into equivalent weights ==========
        # spatial: [T, H, W, C_backbone]
        # Goal: CAM = spatial @ w_folded

        T_s, H_s, W_s, C_backbone = spatial.shape

        # Start with backbone features
        features = spatial  # [T, H, W, C_backbone]

        # Step 1: Apply feat_proj if exists (Linear: C_backbone -> C_embed)
        if hasattr(self.model.reid_head, 'feat_proj') and self.model.reid_head.feat_proj is not None:
            logger.info("[OriginalCAM] Folding feat_proj into weights")
            feat_proj = self.model.reid_head.feat_proj
            W_proj = feat_proj.weight  # [C_embed, C_backbone]
            b_proj = feat_proj.bias    # [C_embed]

            # Project features: [T, H, W, C_backbone] @ [C_backbone, C_embed]^T
            features_flat = features.reshape(-1, C_backbone)  # [T*H*W, C_backbone]
            features_proj = torch.matmul(features_flat, W_proj.T) + b_proj  # [T*H*W, C_embed]
            features = features_proj.reshape(T_s, H_s, W_s, -1)  # [T, H, W, C_embed]
            logger.info(f"[OriginalCAM] After feat_proj: {features.shape}")

        C_embed = features.shape[-1]

        # Step 2: Fold BN into classifier weights (in eval mode)
        if hasattr(self.model.reid_head, 'bottleneck'):
            logger.info("[OriginalCAM] Folding BN into classifier weights")
            bn = self.model.reid_head.bottleneck
            classifier = self.model.reid_head.classifier

            # BN parameters (eval mode uses running stats)
            gamma = bn.weight            # [C_embed]
            beta = bn.bias               # [C_embed]
            running_mean = bn.running_mean  # [C_embed]
            running_var = bn.running_var    # [C_embed]
            eps = bn.eps

            # Classifier parameters
            W = classifier.weight  # [num_classes, C_embed]
            b = classifier.bias if classifier.bias is not None else torch.zeros(W.shape[0], device=W.device)

            # Fold BN into Linear:
            # BN: y = gamma * (x - mean) / sqrt(var + eps) + beta
            # Linear: z = W @ y + b
            # Folded: z = W_folded @ x + b_folded
            scale = gamma / torch.sqrt(running_var + eps)  # [C_embed]
            shift = beta - gamma * running_mean / torch.sqrt(running_var + eps)  # [C_embed]

            W_folded = W * scale.unsqueeze(0)  # [num_classes, C_embed]
            # b_folded = W @ shift + b  # We only need weights for target class

            # Get folded weight for target class
            w_c_folded = W_folded[chosen]  # [C_embed]

            logger.info(f"[OriginalCAM] Folded weight stats: mean={w_c_folded.mean():.4f}, std={w_c_folded.std():.4f}, min={w_c_folded.min():.4f}, max={w_c_folded.max():.4f}")
            logger.info(f"[OriginalCAM] BN scale stats: mean={scale.mean():.4f}, std={scale.std():.4f}, min={scale.min():.4f}, max={scale.max():.4f}")
        else:
            # No BN, use classifier weights directly
            w_c_folded = w_c

        # Verify channel dimension matches
        if features.shape[-1] != w_c_folded.shape[0]:
            logger.error(f"[OriginalCAM] Channel mismatch: features has {features.shape[-1]} channels, weights have {w_c_folded.shape[0]}")
            return None

        # Weighted sum: [T, H, W, C] * [C] -> [T, H, W]
        # This computes: score = features @ w_c_folded (before bias)
        cam_3d = torch.sum(features * w_c_folded.reshape(1, 1, 1, -1), dim=3)  # [T, H, W]
        cam_3d = cam_3d.detach().cpu().numpy()

        logger.info(f"[OriginalCAM] CAM before ReLU: mean={cam_3d.mean():.4f}, std={cam_3d.std():.4f}, min={cam_3d.min():.4f}, max={cam_3d.max():.4f}")

        # ReLU and normalize
        cam_3d = np.maximum(cam_3d, 0)
        logger.info(f"[OriginalCAM] CAM after ReLU: mean={cam_3d.mean():.4f}, std={cam_3d.std():.4f}, min={cam_3d.min():.4f}, max={cam_3d.max():.4f}")

        cam_3d = cam_3d - cam_3d.min()
        if cam_3d.max() > 0:
            cam_3d = cam_3d / cam_3d.max()
            logger.info(f"[OriginalCAM] CAM after normalization: mean={cam_3d.mean():.4f}, std={cam_3d.std():.4f}")
        else:
            logger.warning("[OriginalCAM] ⚠ WARNING: CAM max is 0!")

        # Return in same format as other CAM methods: [1, N_tokens]
        return cam_3d.reshape(1, -1)
