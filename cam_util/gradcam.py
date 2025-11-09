"""
GradCAM: Gradient-weighted Class Activation Mapping
"""

import torch
import torch.nn.functional as F
import utils.logging as logging

logger = logging.get_logger(__name__)


class SimpleGradCAM:
    def __init__(self, model, target_layer, finer=False, bn_folding=False):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.finer = finer
        self.bn_folding = bn_folding
        self.pre_bn_features = None

        # Register forward hook to capture activations
        target_layer.register_forward_hook(self.save_activation)

        # Register hook for BN folding (capture features before BN)
        if bn_folding and hasattr(model, 'reid_head'):
            if hasattr(model.reid_head, 'feat_proj') and model.reid_head.feat_proj is not None:
                # Hook on feat_proj if it exists
                model.reid_head.feat_proj.register_forward_hook(self.save_pre_bn_features)
                logger.info("BN Folding enabled: registered hook on feat_proj")
            elif hasattr(model, 'backbone'):
                # If no feat_proj, hook on backbone output directly
                model.backbone.register_forward_hook(self.save_pre_bn_features)
                logger.info("BN Folding enabled: registered hook on backbone (no feat_proj)")
            else:
                logger.warning("BN Folding requested but no feat_proj or backbone found")
                self.bn_folding = False

    def save_pre_bn_features(self, module, input, output):
        """Save features before BN layer for BN folding."""
        self.pre_bn_features = output
        if output.requires_grad:
            output.retain_grad()
            logger.info(f"[BN Folding] Retained grad on pre-BN features: {output.shape}")

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
            logger.info(f"Retained grad on activation: {activation_tensor.shape}")

        def save_grad(grad):
            self.gradients = grad.clone()
            logger.info(f"Tensor hook: Saved gradient shape: {self.gradients.shape}")

        if activation_tensor.requires_grad:
            activation_tensor.register_hook(save_grad)
            logger.info(f"Registered tensor hook on activation: {activation_tensor.shape}")

    def generate_cam(self, input_tensor, target_id=None):
        """Generate CAM for input."""
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
                logger.info(f"Model output (cls_score) shape: {logits.shape}, requires_grad: {logits.requires_grad}")
            else:
                logger.warning("No 'cls_score' in model output; falling back to global_feat.")
            if logits is None and 'global_feat' in output:
                features = output['global_feat']  # [B, D]
                logger.info(f"Model output (global_feat) shape: {features.shape}, requires_grad: {features.requires_grad}")
        else:
            features = output
            logger.info(f"Model output shape: {features.shape}, requires_grad: {features.requires_grad}")

        if self.activations is None:
            logger.error("No activations captured during forward pass!")
            return None

        logger.info(f"Activations shape: {self.activations.shape}, requires_grad: {self.activations.requires_grad}")

        # Select scalar score for backprop
        if logits is not None:
            # Compute probabilities and show top-5 predictions
            probs = F.softmax(logits[0], dim=0)
            top5_probs, top5_ids = torch.topk(probs, min(5, logits.shape[1]))

            logger.info("="*70)
            logger.info("Top-5 Predicted Classes:")
            for i, (pred_id, prob) in enumerate(zip(top5_ids, top5_probs)):
                logit_val = float(logits[0, pred_id])
                logger.info(f"  Rank {i+1}: Class {int(pred_id):3d} | Probability: {float(prob)*100:6.2f}% | Logit: {logit_val:+.3f}")
            logger.info("="*70)

            if target_id is None or target_id < 0 or target_id >= logits.shape[1]:
                chosen = int(torch.argmax(logits[0]).item())
            else:
                chosen = int(target_id)

            # ========== BN Folding: Compute score directly from pre-BN features ==========
            if self.bn_folding and self.pre_bn_features is not None and hasattr(self.model, 'reid_head'):
                logger.info("="*70)
                logger.info("[BN Folding] Computing score from pre-BN features")

                bn = self.model.reid_head.bottleneck
                classifier = self.model.reid_head.classifier

                # BN parameters (running stats in eval mode)
                gamma = bn.weight  # [embed_dim]
                beta = bn.bias  # [embed_dim]
                running_mean = bn.running_mean  # [embed_dim]
                running_var = bn.running_var  # [embed_dim]
                eps = bn.eps

                # Classifier parameters
                W = classifier.weight  # [num_classes, embed_dim]
                b = classifier.bias if classifier.bias is not None else torch.zeros(W.shape[0], device=W.device)

                # Fold BN into Linear:
                # BN: y = gamma * (x - mean) / sqrt(var + eps) + beta
                # Linear: z = W @ y + b
                # Merged: z = W' @ x + b'
                scale = gamma / torch.sqrt(running_var + eps)  # [embed_dim]
                shift = beta - gamma * running_mean / torch.sqrt(running_var + eps)  # [embed_dim]

                W_folded = W * scale.unsqueeze(0)  # [num_classes, embed_dim]
                b_folded = W @ shift + b  # [num_classes]

                logger.info(f"[BN Folding] scale stats: mean={scale.mean():.4f}, std={scale.std():.4f}, min={scale.min():.4f}, max={scale.max():.4f}")
                logger.info(f"[BN Folding] W_folded stats: mean={W_folded.mean():.4f}, std={W_folded.std():.4f}")

                # Compute logits directly from pre-BN features
                z_preBN = self.pre_bn_features  # [B, embed_dim]
                logits_folded = z_preBN @ W_folded.T + b_folded  # [B, num_classes]

                # Verify that folded logits match original logits
                logit_diff = (logits_folded - logits).abs().max()
                logger.info(f"[BN Folding] Max difference between folded and original logits: {logit_diff:.6f}")
                if logit_diff > 1e-3:
                    logger.warning(f"[BN Folding] Large difference detected! Using original score instead.")
                    # Fallback to original score
                    if self.finer:
                        logits_sorted, indices_sorted = torch.sort(logits[0], descending=True)
                        second_class = None
                        for idx in indices_sorted:
                            if int(idx.item()) != chosen:
                                second_class = int(idx.item())
                                break
                        if second_class is None:
                            second_class = chosen
                        score = logits[0, chosen] - logits[0, second_class]
                    else:
                        score = logits[0, chosen]
                else:
                    # Use folded score (computed from pre-BN features)
                    if self.finer:
                        logits_sorted, indices_sorted = torch.sort(logits_folded[0], descending=True)
                        second_class = None
                        for idx in indices_sorted:
                            if int(idx.item()) != chosen:
                                second_class = int(idx.item())
                                break
                        if second_class is None:
                            second_class = chosen
                        score = logits_folded[0, chosen] - logits_folded[0, second_class]
                        logger.info(f"[BN Folding] Using FINER mode with folded logits: class {chosen} vs {second_class}, diff: {float(score):.3f}")
                    else:
                        score = logits_folded[0, chosen]
                        logger.info(f"[BN Folding] Using folded logit for class {chosen}: {float(score):.3f}")
                logger.info("="*70)
            else:
                # Original path: use logits computed through BN
                if self.finer:
                    # Finer mode: use target class logit minus second-highest class logit
                    logits_sorted, indices_sorted = torch.sort(logits[0], descending=True)
                    # Find second class (avoid choosing the same as target)
                    second_class = None
                    for idx in indices_sorted:
                        if int(idx.item()) != chosen:
                            second_class = int(idx.item())
                            break
                    if second_class is None:
                        second_class = chosen
                    score = logits[0, chosen] - logits[0, second_class]
                    logger.info(f"Using FINER mode: class {chosen} (logit: {float(logits[0, chosen]):.3f}) vs class {second_class} (logit: {float(logits[0, second_class]):.3f}), diff: {float(score):.3f}")
                else:
                    # Use logit directly
                    score = logits[0, chosen]
                    logger.info(f"Using class {chosen} for CAM (probability: {float(probs[chosen])*100:.2f}%, logit: {float(score):.3f})")
        else:
            # Fallback on a feature element
            score = features[0, 0] if features.dim() == 2 else features[0].view(-1)[0]
            logger.info("⚠ WARNING: No cls_score available, using feature channel as fallback")
        logger.info(f"Gradient target score: {float(score.item()):.6f}, requires_grad: {score.requires_grad}")

        # Compute gradients manually
        logger.info("Computing gradients...")
        try:
            grads = torch.autograd.grad(outputs=score, inputs=self.activations,
                                        retain_graph=False, create_graph=False)
            self.gradients = grads[0]
            logger.info(f"Gradient computed! Shape: {self.gradients.shape}")
        except Exception as e:
            logger.error(f"Failed to compute gradients: {e}")
            return None

        if self.gradients is None:
            logger.error("Gradients are None!")
            return None

        # Handle different activation shapes
        # UniFormerV2: [L, NT, C] where L=1+HW, NT=batch*time
        # MViT: [B, N, C] where N=tokens
        
        if self.activations.dim() == 3 and self.activations.shape[0] > self.activations.shape[1]:
            # UniFormerV2 format: [L, NT, C]
            L, NT, C = self.activations.shape
            logger.info(f"Detected UniFormerV2 format [L={L}, NT={NT}, C={C}]")
            
            # Infer batch size from input (input_tensor was used in forward pass)
            # Assuming input_tensor has shape [B, C, T, H, W]
            # We need to infer N and T from NT
            # Heuristic: if NT is small (like 16), it's likely N*T with N=2 (batched)
            if NT <= 32:
                N = 2  # Assume batched input
                T = NT // N
            else:
                N = 1
                T = NT
            logger.info(f"Inferred N={N}, T={T} from NT={NT}")
            
            # Remove CLS token and reshape to include time dimension
            activations_no_cls = self.activations[1:, :, :]  # [HW, NT, C]
            gradients_no_cls = self.gradients[1:, :, :]      # [HW, NT, C]
            HW = L - 1
            H = W = int(HW ** 0.5)
            
            # Reshape to [HW, N, T, C] then take first batch -> [HW, T, C]
            activations = activations_no_cls.reshape(HW, N, T, C)[:, 0, :, :]  # [HW, T, C]
            gradients = gradients_no_cls.reshape(HW, N, T, C)[:, 0, :, :]      # [HW, T, C]
            
            # Reshape to [H, W, T, C] then permute to [T, H, W, C]
            activations = activations.reshape(H, W, T, C).permute(2, 0, 1, 3)  # [T, H, W, C]
            gradients = gradients.reshape(H, W, T, C).permute(2, 0, 1, 3)      # [T, H, W, C]
            
            # Flatten to [1, T*H*W, C] for processing
            activations = activations.reshape(1, -1, C)
            gradients = gradients.reshape(1, -1, C)
            
            logger.info(f"After removing CLS and reshaping: activations {activations.shape}, gradients {gradients.shape} (T={T}, H={H}, W={W})")
            
        elif self.activations.dim() == 3:
            # MViT format: [B, N, C]
            logger.info(f"Detected MViT format [B={self.activations.shape[0]}, N={self.activations.shape[1]}, C={self.activations.shape[2]}]")
            activations = self.activations[0:1]  # [1, N, C]
            gradients = self.gradients[0:1]      # [1, N, C]
        else:
            raise ValueError(f"Unexpected activation shape: {self.activations.shape}")

        logger.info(f"Activations stats: mean={activations.mean():.4f}, std={activations.std():.4f}, min={activations.min():.4f}, max={activations.max():.4f}")
        logger.info(f"Gradients stats: mean={gradients.mean():.4f}, std={gradients.std():.4f}, min={gradients.min():.4f}, max={gradients.max():.4f}")

        # Global average pooling of gradients -> weights [1, 1, C]
        weights = gradients.mean(dim=1, keepdim=True)
        logger.info(f"Weights stats: mean={weights.mean():.4f}, std={weights.std():.4f}, min={weights.min():.4f}, max={weights.max():.4f}")

        # Weighted combination over channels -> [1, N]
        cam = (weights * activations).sum(dim=2)
        logger.info(f"CAM before ReLU: mean={cam.mean():.4f}, std={cam.std():.4f}, min={cam.min():.4f}, max={cam.max():.4f}")

        # ReLU and normalize
        cam = F.relu(cam)
        logger.info(f"CAM after ReLU: mean={cam.mean():.4f}, std={cam.std():.4f}, min={cam.min():.4f}, max={cam.max():.4f}")

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
            logger.info(f"CAM after normalization: mean={cam.mean():.4f}, std={cam.std():.4f}, min={cam.min():.4f}, max={cam.max():.4f}")
        else:
            logger.warning("⚠ WARNING: CAM max is 0! All values are identical after ReLU.")

        return cam.detach().cpu().numpy()  # [1, N_tokens]
