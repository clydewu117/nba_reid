"""
LayerCAM: Element-wise multiplication of gradients and activations

Reference: "LayerCAM: Exploring Hierarchical Class Activation Maps"
More fine-grained localization compared to GradCAM.
"""

import torch
import torch.nn.functional as F
import utils.logging as logging

logger = logging.get_logger(__name__)


class LayerCAM:
    """
    LayerCAM: Element-wise multiplication of gradients and activations.
    More fine-grained localization compared to GradCAM.
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
            logger.info(f"[LayerCAM] Retained grad on activation: {activation_tensor.shape}")

        def save_grad(grad):
            self.gradients = grad.clone()
            logger.info(f"[LayerCAM] Tensor hook: Saved gradient shape: {self.gradients.shape}")

        if activation_tensor.requires_grad:
            activation_tensor.register_hook(save_grad)
            logger.info(f"[LayerCAM] Registered tensor hook on activation: {activation_tensor.shape}")

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
                    logger.info("[LayerCAM] Converting embedding output to logits via classifier weights.")
                    logits = torch.matmul(logits, classifier.weight.t())
                    if classifier.bias is not None:
                        logits = logits + classifier.bias.unsqueeze(0)
                else:
                    logger.warning(
                        "[LayerCAM] Output dim %d incompatible with classifier (num_classes=%d, in_dim=%d).",
                        logits.shape[1], num_classes, in_dim
                    )
                    return None
        return logits

    def generate_cam(self, input_tensor, target_id=None):
        """Generate LayerCAM for input."""
        self.model.zero_grad()
        self.activations = None
        self.gradients = None

        # Forward pass (model now returns cls_score in both train/eval modes)
        output = self._forward_for_cam(input_tensor)

        logits = self._extract_logits(output)
        if logits is not None:
            logger.info(f"[LayerCAM] Model logits shape: {logits.shape}, requires_grad: {logits.requires_grad}")

        if self.activations is None:
            logger.error("[LayerCAM] No activations captured during forward pass!")
            return None

        logger.info(f"[LayerCAM] Activations shape: {self.activations.shape}, requires_grad: {self.activations.requires_grad}")

        # Select scalar score for backprop
        if logits is not None:
            # Compute probabilities and show top-5 predictions
            probs = F.softmax(logits[0], dim=0)
            top5_probs, top5_ids = torch.topk(probs, min(5, logits.shape[1]))

            logger.info("="*70)
            logger.info("[LayerCAM] Top-5 Predicted Classes:")
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
            logger.info(f"[LayerCAM] Using class {chosen} for CAM (probability: {float(probs[chosen])*100:.2f}%, logit: {float(score):.3f})")
        else:
            logger.error("[LayerCAM] Cannot compute LayerCAM without classification logits.")
            return None
        logger.info(f"[LayerCAM] Gradient target score: {float(score.item()):.6f}, requires_grad: {score.requires_grad}")

        # Compute gradients manually
        logger.info("[LayerCAM] Computing gradients...")
        try:
            grads = torch.autograd.grad(outputs=score, inputs=self.activations,
                                        retain_graph=False, create_graph=False)
            self.gradients = grads[0]
            logger.info(f"[LayerCAM] Gradient computed! Shape: {self.gradients.shape}")
        except Exception as e:
            logger.error(f"[LayerCAM] Failed to compute gradients: {e}")
            return None

        if self.gradients is None:
            logger.error("[LayerCAM] Gradients are None!")
            return None

        # Handle different activation shapes
        # UniFormerV2: [L, NT, C] where L=1+HW, NT=batch*time
        # MViT: [B, N, C] where N=tokens
        
        if self.activations.dim() == 3 and self.activations.shape[0] > self.activations.shape[1]:
            # UniFormerV2 format: [L, NT, C]
            L, NT, C = self.activations.shape
            logger.info(f"[LayerCAM] Detected UniFormerV2 format [L={L}, NT={NT}, C={C}]")
            
            # Infer batch and time from NT
            if NT <= 32:
                N = 2
                T = NT // N
            else:
                N = 1
                T = NT
            logger.info(f"[LayerCAM] Inferred N={N}, T={T} from NT={NT}")
            
            # Remove CLS token and reshape to include time dimension
            activations_no_cls = self.activations[1:, :, :]  # [HW, NT, C]
            gradients_no_cls = self.gradients[1:, :, :]      # [HW, NT, C]
            HW = L - 1
            H = W = int(HW ** 0.5)
            
            # Reshape to [HW, N, T, C] then take first batch -> [HW, T, C]
            activations = activations_no_cls.reshape(HW, N, T, C)[:, 0, :, :]
            gradients = gradients_no_cls.reshape(HW, N, T, C)[:, 0, :, :]
            
            # Reshape to [H, W, T, C] then permute to [T, H, W, C]
            activations = activations.reshape(H, W, T, C).permute(2, 0, 1, 3)
            gradients = gradients.reshape(H, W, T, C).permute(2, 0, 1, 3)
            
            # Flatten to [1, T*H*W, C] for processing
            activations = activations.reshape(1, -1, C)
            gradients = gradients.reshape(1, -1, C)
            
            logger.info(f"[LayerCAM] After removing CLS and reshaping: activations {activations.shape}, gradients {gradients.shape} (T={T}, H={H}, W={W})")
            
        elif self.activations.dim() == 3:
            # MViT format: [B, N, C]
            logger.info(f"[LayerCAM] Detected MViT format [B={self.activations.shape[0]}, N={self.activations.shape[1]}, C={self.activations.shape[2]}]")
            activations = self.activations[0:1]  # [1, N, C]
            gradients = self.gradients[0:1]      # [1, N, C]
        else:
            raise ValueError(f"Unexpected activation shape: {self.activations.shape}")

        logger.info(f"[LayerCAM] Activations stats: mean={activations.mean():.4f}, std={activations.std():.4f}, min={activations.min():.4f}, max={activations.max():.4f}")
        logger.info(f"[LayerCAM] Gradients stats: mean={gradients.mean():.4f}, std={gradients.std():.4f}, min={gradients.min():.4f}, max={gradients.max():.4f}")

        # LayerCAM: Element-wise multiplication then sum over channels
        # No global average pooling of gradients
        cam = (gradients * activations).sum(dim=2)  # [1, N]
        logger.info(f"[LayerCAM] CAM before ReLU: mean={cam.mean():.4f}, std={cam.std():.4f}, min={cam.min():.4f}, max={cam.max():.4f}")

        # ReLU and normalize
        cam = F.relu(cam)
        logger.info(f"[LayerCAM] CAM after ReLU: mean={cam.mean():.4f}, std={cam.std():.4f}, min={cam.min():.4f}, max={cam.max():.4f}")

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
            logger.info(f"[LayerCAM] CAM after normalization: mean={cam.mean():.4f}, std={cam.std():.4f}, min={cam.min():.4f}, max={cam.max():.4f}")
        else:
            logger.warning("[LayerCAM] ⚠ WARNING: CAM max is 0! All values are identical after ReLU.")

        return cam.detach().cpu().numpy()  # [1, N_tokens]
