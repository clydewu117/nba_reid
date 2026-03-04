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

    def _get_reid_head(self):
        reid_head = getattr(self.model, "reid_head", None)
        if reid_head is None and hasattr(self.model, "module"):
            reid_head = getattr(self.model.module, "reid_head", None)
        return reid_head

    def _forward_for_cam(self, input_tensor):
        """Run forward while forcing classification logits when available."""
        reid_head = self._get_reid_head()
        toggled = False
        original_mode = None
        if (
            reid_head is not None
            and hasattr(reid_head, "is_classification")
            and not reid_head.is_classification
        ):
            toggled = True
            original_mode = reid_head.is_classification
            reid_head.is_classification = True

        try:
            return self.model(input_tensor)
        finally:
            if toggled and original_mode is not None:
                reid_head.is_classification = original_mode

    def _extract_logits(self, output):
        """Extract classification logits from dict/tuple/tensor outputs."""
        logits = None

        if isinstance(output, dict):
            logits = output.get("cls_score", None)
        elif isinstance(output, tuple) and len(output) >= 1:
            logits = output[0]
        elif torch.is_tensor(output):
            logits = output

        if logits is None or logits.dim() != 2:
            return None

        classifier = getattr(self._get_reid_head(), "classifier", None)
        if classifier is not None:
            num_classes = classifier.weight.shape[0]
            in_dim = classifier.weight.shape[1]
            if logits.shape[1] != num_classes:
                if logits.shape[1] == in_dim:
                    logger.info("[GradCAM++] Converting embedding output to logits via classifier weights.")
                    logits = torch.matmul(logits, classifier.weight.t())
                    if classifier.bias is not None:
                        logits = logits + classifier.bias.unsqueeze(0)
                else:
                    logger.warning(
                        "[GradCAM++] Output dim %d incompatible with classifier (num_classes=%d, in_dim=%d).",
                        logits.shape[1], num_classes, in_dim
                    )
                    return None
        return logits

    def generate_cam(self, input_tensor, target_id=None):
        """Generate GradCAM++ for input."""
        self.model.zero_grad()
        self.activations = None
        self.gradients = None

        # MViTv2: patch mean for gradients; two-phase when choosing class
        has_cam_pooling = hasattr(self.model, "_cam_pooling")
        chosen = None
        if has_cam_pooling and (target_id is None or target_id < 0):
            # Phase 1: CLS path for correct chosen (avoids BN/patch-mean distribution mismatch)
            self.model._cam_pooling = "choose_class"
            try:
                with torch.no_grad():
                    output_cls = self._forward_for_cam(input_tensor)
                logits_cls = self._extract_logits(output_cls)
                if logits_cls is not None:
                    chosen = int(torch.argmax(logits_cls[0]).item())
                    logger.info(f"[GradCAM++] MViT two-phase: chosen class {chosen} from CLS path")
            finally:
                self.model._cam_pooling = None
        # Phase 2: patch mean path for gradient flow (activations captured here)
        if has_cam_pooling:
            self.model._cam_pooling = "gradient"
        try:
            output = self._forward_for_cam(input_tensor)
        finally:
            if has_cam_pooling:
                self.model._cam_pooling = None

        logits = self._extract_logits(output)
        if logits is not None:
            logger.info(f"[GradCAM++] Model logits shape: {logits.shape}, requires_grad: {logits.requires_grad}")

        if self.activations is None:
            logger.error("[GradCAM++] No activations captured during forward pass!")
            return None

        logger.info(f"[GradCAM++] Activations shape: {self.activations.shape}, requires_grad: {self.activations.requires_grad}")

        # Select scalar score for backprop
        if logits is not None:
            probs = F.softmax(logits[0], dim=0)
            top5_probs, top5_ids = torch.topk(probs, min(5, logits.shape[1]))
            logger.info("="*70)
            logger.info("[GradCAM++] Top-5 Predicted Classes:")
            for i, (pred_id, prob) in enumerate(zip(top5_ids, top5_probs)):
                logit_val = float(logits[0, pred_id])
                logger.info(f"  Rank {i+1}: Class {int(pred_id):3d} | Probability: {float(prob)*100:6.2f}% | Logit: {logit_val:+.3f}")
            logger.info("="*70)
            if chosen is not None:
                pass
            elif target_id is None or target_id < 0 or target_id >= logits.shape[1]:
                chosen = int(torch.argmax(logits[0]).item())
            else:
                chosen = int(target_id)
            # Use logit directly
            score = logits[0, chosen]
            logger.info(f"[GradCAM++] Using class {chosen} for CAM (probability: {float(probs[chosen])*100:.2f}%, logit: {float(score):.3f})")
        else:
            logger.error("[GradCAM++] Cannot compute GradCAM++ without classification logits.")
            return None
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
            # MViT format: [B, N, C] where N=tokens (1+THW when cls_embed_on)
            logger.info(f"[GradCAM++] Detected MViT format [B={self.activations.shape[0]}, N={self.activations.shape[1]}, C={self.activations.shape[2]}]")
            activations = self.activations[0:1]  # [1, N, C]
            grads = self.gradients[0:1]          # [1, N, C]
            # Remove CLS token for MViT (first token when cls_embed_on)
            if getattr(self.model.backbone, "cls_embed_on", False):
                activations = activations[:, 1:, :]
                grads = grads[:, 1:, :]
                logger.info(f"[GradCAM++] Removed CLS token for MViT, new shape: {activations.shape}")
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
        # (CLS token already removed from activations for MViT above)
        cam = np.sum(weights * activations_np, axis=2)
        logger.info(f"[GradCAM++] CAM before ReLU: mean={cam.mean():.4f}, std={cam.std():.4f}, min={cam.min():.4f}, max={cam.max():.4f}")

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
