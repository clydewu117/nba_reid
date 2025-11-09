"""
GradCAM++: Improved Visual Explanations via Second-Order Gradients

Reference: "Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks"
https://arxiv.org/abs/1710.11063

GradCAM++ uses second and third order gradients to compute better weights for
the activation maps, providing more accurate localization compared to GradCAM.
"""

import numpy as np
import torch
import torch.nn.functional as F
import utils.logging as logging

logger = logging.get_logger(__name__)


class GradCAMPlusPlus:
    """
    GradCAM++ implementation using second-order gradients.

    Key differences from GradCAM:
    - Uses second and third order derivatives for weight computation
    - Better handling of multiple instances of same class
    - More accurate localization, especially when multiple objects are present
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward hook to capture activations
        target_layer.register_forward_hook(self.save_activation)

    def save_activation(self, module, input, output):
        # MViT blocks may return (x, thw)
        if isinstance(output, tuple):
            activation_tensor = output[0]
        else:
            activation_tensor = output

        self.activations = activation_tensor

        # Force gradient retention for eval mode
        if activation_tensor.requires_grad:
            activation_tensor.retain_grad()
            logger.info(f"[GradCAM++] Retained grad on activation: {activation_tensor.shape}")

        def save_grad(grad):
            self.gradients = grad.clone()
            logger.info(f"[GradCAM++] Tensor hook: Saved gradient shape: {self.gradients.shape}")

        if activation_tensor.requires_grad:
            activation_tensor.register_hook(save_grad)
            logger.info(f"[GradCAM++] Registered tensor hook on activation: {activation_tensor.shape}")

    def generate_cam(self, input_tensor, target_id=None):
        """Generate GradCAM++ for input."""
        self.model.zero_grad()
        self.activations = None
        self.gradients = None

        # Forward pass (model now returns cls_score in both train/eval modes)
        output = self.model(input_tensor)

        # Handle dict or tensor output
        logits = None
        features = None
        if isinstance(output, dict):
            if 'cls_score' in output:
                logits = output['cls_score']  # [B, num_classes]
                logger.info(f"[GradCAM++] Model output (cls_score) shape: {logits.shape}, requires_grad: {logits.requires_grad}")
            else:
                logger.warning("[GradCAM++] No 'cls_score' in model output; falling back to global_feat.")
            if logits is None and 'global_feat' in output:
                features = output['global_feat']  # [B, D]
                logger.info(f"[GradCAM++] Model output (global_feat) shape: {features.shape}, requires_grad: {features.requires_grad}")
        else:
            features = output
            logger.info(f"[GradCAM++] Model output shape: {features.shape}, requires_grad: {features.requires_grad}")

        if self.activations is None:
            logger.error("[GradCAM++] No activations captured during forward pass!")
            return None

        logger.info(f"[GradCAM++] Activations shape: {self.activations.shape}, requires_grad: {self.activations.requires_grad}")

        # Select scalar score for backprop
        if logits is not None:
            # Compute probabilities and show top-5 predictions
            probs = F.softmax(logits[0], dim=0)
            top5_probs, top5_ids = torch.topk(probs, min(5, logits.shape[1]))

            logger.info("="*70)
            logger.info("[GradCAM++] Top-5 Predicted Classes:")
            for i, (pred_id, prob) in enumerate(zip(top5_ids, top5_probs)):
                logit_val = float(logits[0, pred_id])
                logger.info(f"  Rank {i+1}: Class {int(pred_id):3d} | Probability: {float(prob)*100:6.2f}% | Logit: {logit_val:+.3f}")
            logger.info("="*70)

            if target_id is None or target_id < 0 or target_id >= logits.shape[1]:
                chosen = int(torch.argmax(logits[0]).item())
            else:
                chosen = int(target_id)
            # Use logit directly
            score = logits[0, chosen]
            logger.info(f"[GradCAM++] Using class {chosen} for CAM (probability: {float(probs[chosen])*100:.2f}%, logit: {float(score):.3f})")
        else:
            # Fallback on a feature element
            score = features[0, 0] if features.dim() == 2 else features[0].view(-1)[0]
            logger.info("[GradCAM++] ⚠ WARNING: No cls_score available, using feature channel as fallback")
        logger.info(f"[GradCAM++] Gradient target score: {float(score.item()):.6f}, requires_grad: {score.requires_grad}")

        # Compute first-order gradients
        logger.info("[GradCAM++] Computing first-order gradients...")
        try:
            grads = torch.autograd.grad(outputs=score, inputs=self.activations,
                                        retain_graph=True, create_graph=True)
            self.gradients = grads[0]
            logger.info(f"[GradCAM++] First-order gradient computed! Shape: {self.gradients.shape}")
        except Exception as e:
            logger.error(f"[GradCAM++] Failed to compute gradients: {e}")
            return None

        if self.gradients is None:
            logger.error("[GradCAM++] Gradients are None!")
            return None

        # Handle different activation shapes
        # UniFormerV2: [L, NT, C] where L=1+HW, NT=batch*time
        # MViT: [B, N, C] where N=tokens
        
        if self.activations.dim() == 3 and self.activations.shape[0] > self.activations.shape[1]:
            # UniFormerV2 format: [L, NT, C]
            L, NT, C = self.activations.shape
            logger.info(f"[GradCAM++] Detected UniFormerV2 format [L={L}, NT={NT}, C={C}]")
            
            # Infer batch and time from NT
            if NT <= 32:
                N = 2
                T = NT // N
            else:
                N = 1
                T = NT
            logger.info(f"[GradCAM++] Inferred N={N}, T={T} from NT={NT}")
            
            # Remove CLS token and reshape to include time dimension
            activations_no_cls = self.activations[1:, :, :]  # [HW, NT, C]
            gradients_no_cls = self.gradients[1:, :, :]      # [HW, NT, C]
            HW = L - 1
            H = W = int(HW ** 0.5)
            
            # Reshape to [HW, N, T, C] then take first batch -> [HW, T, C]
            activations = activations_no_cls.reshape(HW, N, T, C)[:, 0, :, :]
            grads = gradients_no_cls.reshape(HW, N, T, C)[:, 0, :, :]
            
            # Reshape to [H, W, T, C] then permute to [T, H, W, C]
            activations = activations.reshape(H, W, T, C).permute(2, 0, 1, 3)
            grads = grads.reshape(H, W, T, C).permute(2, 0, 1, 3)
            
            # Flatten to [1, T*H*W, C] for processing
            activations = activations.reshape(1, -1, C)
            grads = grads.reshape(1, -1, C)
            
            logger.info(f"[GradCAM++] After removing CLS and reshaping: activations {activations.shape}, grads {grads.shape} (T={T}, H={H}, W={W})")
            
        elif self.activations.dim() == 3:
            # MViT format: [B, N, C]
            logger.info(f"[GradCAM++] Detected MViT format [B={self.activations.shape[0]}, N={self.activations.shape[1]}, C={self.activations.shape[2]}]")
            activations = self.activations[0:1]  # [1, N, C]
            grads = self.gradients[0:1]          # [1, N, C]
        else:
            raise ValueError(f"Unexpected activation shape: {self.activations.shape}")

        logger.info(f"[GradCAM++] Activations stats: mean={activations.mean():.4f}, std={activations.std():.4f}, min={activations.min():.4f}, max={activations.max():.4f}")
        logger.info(f"[GradCAM++] First-order gradients stats: mean={grads.mean():.4f}, std={grads.std():.4f}, min={grads.min():.4f}, max={grads.max():.4f}")

        # Compute second and third order gradients (GradCAM++ formula)
        # Convert to numpy for numerical stability
        grads_np = grads.detach().cpu().numpy()
        activations_np = activations.detach().cpu().numpy()

        # Compute gradient powers
        grads_power_2 = grads_np ** 2
        grads_power_3 = grads_np ** 3

        logger.info(f"[GradCAM++] Second-order (grads²) stats: mean={grads_power_2.mean():.4f}, std={grads_power_2.std():.4f}")
        logger.info(f"[GradCAM++] Third-order (grads³) stats: mean={grads_power_3.mean():.4f}, std={grads_power_3.std():.4f}")

        # GradCAM++ Equation 19:
        # α_ij^kc = (∂²y^c/∂A_ij^k²) / (2(∂²y^c/∂A_ij^k²) + Σ_ij A_ij^k (∂³y^c/∂A_ij^k³))
        #
        # For token-based models [B, N, C]:
        # sum_activations: sum over token dimension N -> [B, C]
        sum_activations = np.sum(activations_np, axis=1, keepdims=True)  # [1, 1, C]

        eps = 1e-6  # Small epsilon for numerical stability

        # Compute alpha weights (importance of each token-channel)
        # Broadcast sum_activations from [1, 1, C] to [1, N, C]
        denominator = 2 * grads_power_2 + sum_activations * grads_power_3 + eps
        alpha = grads_power_2 / denominator

        # Zero out alpha where gradients are zero (from eq.7 in paper)
        alpha = np.where(grads_np != 0, alpha, 0)

        logger.info(f"[GradCAM++] Alpha weights stats: mean={alpha.mean():.4f}, std={alpha.std():.4f}, min={alpha.min():.4f}, max={alpha.max():.4f}")

        # Compute final weights: w^c = Σ_n ReLU(∂y^c/∂A_n^c) * α_n^c
        # Apply ReLU to gradients
        relu_grads = np.maximum(grads_np, 0)

        # Weight each gradient by its alpha value
        weighted_grads = relu_grads * alpha

        # Sum over token dimension to get channel weights [1, 1, C]
        weights = np.sum(weighted_grads, axis=1, keepdims=True)

        logger.info(f"[GradCAM++] Channel weights stats: mean={weights.mean():.4f}, std={weights.std():.4f}, min={weights.min():.4f}, max={weights.max():.4f}")

        # Weighted combination over channels -> [1, N]
        cam = np.sum(weights * activations_np, axis=2)
        logger.info(f"[GradCAM++] CAM before ReLU: mean={cam.mean():.4f}, std={cam.std():.4f}, min={cam.min():.4f}, max={cam.max():.4f}")

        # Remove CLS token if present
        try:
            if getattr(self.model.backbone, "cls_embed_on", False):
                cam = cam[:, 1:]
                logger.info(f"[GradCAM++] Removed CLS token, new shape: {cam.shape}")
        except Exception:
            # if backbone not present/attr missing, just skip
            pass

        # ReLU and normalize
        cam = np.maximum(cam, 0)
        logger.info(f"[GradCAM++] CAM after ReLU: mean={cam.mean():.4f}, std={cam.std():.4f}, min={cam.min():.4f}, max={cam.max():.4f}")

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
            logger.info(f"[GradCAM++] CAM after normalization: mean={cam.mean():.4f}, std={cam.std():.4f}, min={cam.min():.4f}, max={cam.max():.4f}")
        else:
            logger.warning("[GradCAM++] ⚠ WARNING: CAM max is 0! All values are identical after ReLU.")

        return cam  # [1, N_tokens]
