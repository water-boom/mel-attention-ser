"""Causal Frame Masking Probe.

Evaluates the causal necessity of high-attention frames vs low-attention frames.
If the attention mechanism is genuinely capturing critical acoustic evidence:
- Masking Top-K% frames should cause steep performance degradation.
- Masking Bottom-K% frames should leave performance largely unaffected.
"""

from typing import Callable, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score


def apply_frame_mask(
    x: torch.Tensor,
    importance_scores: torch.Tensor,
    mask_ratio: float,
    mode: str = "top",
) -> torch.Tensor:
    """Mask frames of input tensor x based on importance scores.

    Args:
        x: Input tensor of shape (B, C, F, T)
        importance_scores: Frame importance of shape (B, T)
        mask_ratio: Float in [0.0, 1.0] fraction of frames to mask
        mode: 'top' (mask most important), 'bottom' (mask least important), 'random'
    Returns:
        Masked tensor of same shape
    """
    if mask_ratio <= 0.0:
        return x.clone()

    b, c, f, t = x.shape
    num_to_mask = int(round(mask_ratio * t))
    if num_to_mask <= 0:
        return x.clone()
    if num_to_mask >= t:
        return torch.zeros_like(x)

    x_masked = x.clone()

    if mode == "random":
        for i in range(b):
            perm = torch.randperm(t, device=x.device)
            indices = perm[:num_to_mask]
            x_masked[i, :, :, indices] = 0.0
    elif mode == "top":
        # Top largest scores
        _, indices = torch.topk(importance_scores, k=min(num_to_mask, importance_scores.shape[-1]), dim=-1, largest=True)
        for i in range(b):
            x_masked[i, :, :, indices[i]] = 0.0
    elif mode == "bottom":
        # Top smallest scores
        _, indices = torch.topk(importance_scores, k=min(num_to_mask, importance_scores.shape[-1]), dim=-1, largest=False)
        for i in range(b):
            x_masked[i, :, :, indices[i]] = 0.0
    else:
        raise ValueError(f"Unknown masking mode: {mode}")

    return x_masked


class CausalMaskingProbe:
    """Runs causal masking evaluation over a validation dataloader."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        ratios: Optional[List[float]] = None,
    ):
        self.model = model
        self.device = device
        self.ratios = ratios or [0.0, 0.1, 0.2, 0.3, 0.5]

    @torch.no_grad()
    def evaluate(self, data_loader) -> Dict[str, Dict[float, float]]:
        """Run Top-K, Bottom-K, and Random masking curves.

        Returns:
            Dict containing:
                'top': {0.0: f1, 0.1: f1, ...},
                'bottom': {0.0: f1, 0.1: f1, ...},
                'random': {0.0: f1, 0.1: f1, ...}
        """
        self.model.eval()
        results = {"top": {}, "bottom": {}, "random": {}}

        # First collect all features, labels, and importance scores
        all_features = []
        all_labels = []
        all_scores = []

        for feats, labels, _ in data_loader:
            feats = feats.to(self.device)
            attn_maps = self.model.get_attention_maps(feats)

            if attn_maps is None:
                # Fallback: use temporal energy envelope as importance score
                scores = feats[:, 0, :, :].pow(2).mean(dim=1)  # (B, T)
            else:
                # attn_maps is (B, num_heads, T_sub) -> average over heads & interpolate to T
                scores_sub = attn_maps.mean(dim=1, keepdim=True)  # (B, 1, T_sub)
                if scores_sub.shape[-1] != feats.shape[-1]:
                    scores_interp = F.interpolate(scores_sub, size=feats.shape[-1], mode="nearest")
                else:
                    scores_interp = scores_sub
                scores = scores_interp.squeeze(1)  # (B, T)

            all_features.append(feats.cpu())
            all_labels.append(labels.numpy())
            all_scores.append(scores.cpu())

        all_features = torch.cat(all_features, dim=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_scores = torch.cat(all_scores, dim=0)

        # Evaluate across each ratio and mode
        for mode in ["top", "bottom", "random"]:
            for r in self.ratios:
                masked_feats = apply_frame_mask(all_features, all_scores, r, mode=mode)
                
                # Predict in batches
                preds = []
                batch_size = 64
                for idx in range(0, len(masked_feats), batch_size):
                    batch = masked_feats[idx : idx + batch_size].to(self.device)
                    logits = self.model(batch)
                    pred = logits.argmax(dim=1).cpu().numpy()
                    preds.append(pred)

                all_preds = np.concatenate(preds, axis=0)
                macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
                results[mode][r] = macro_f1

        return results


def evaluate_causal_masking(model, data_loader, device, ratios=None):
    probe = CausalMaskingProbe(model, device, ratios)
    return probe.evaluate(data_loader)
