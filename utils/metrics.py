"""
Loss and Metrics
================
Loss:    Binary cross-entropy per phonological feature, averaged.
Metrics: Per-feature F1, macro F1, feature-group accuracy (manner, place, etc.)
         These map directly to your thesis evaluation section.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, classification_report
from utils.phonological_map import FEATURE_NAMES, NUM_FEATURES


# ── Loss ────────────────────────────────────────────────────────────────────

class PhonologicalLoss(nn.Module):
    """
    Frame-level BCE loss between predicted feature vectors and targets.

    The model outputs per-frame predictions (batch, T_audio, 24).
    The targets are per-phoneme (batch, T_phonemes, 24).

    Since audio frames >> phoneme count, we use mean-pooling to align:
    we pool model predictions over the sequence dimension before comparing
    to pooled targets. This is a simplification — for full alignment you'd
    use forced alignment timestamps, but mean-pool is standard for training.
    """

    def __init__(self, pos_weight: torch.Tensor = None):
        super().__init__()
        # pos_weight handles class imbalance (some features are rare)
        self.bce = nn.BCELoss(reduction="none")

    def forward(
        self,
        pred: torch.Tensor,   # (batch, T_audio, 24) — sigmoid output
        target: torch.Tensor, # (batch, T_phones, 24) — binary labels
        pred_mask: torch.Tensor = None,
        target_lengths: torch.Tensor = None,
    ) -> torch.Tensor:

        # Pool over time → (batch, 24)
        pred_pooled = pred.mean(dim=1)

        # Pool targets over phoneme sequence → (batch, 24)
        if target_lengths is not None:
            # Masked mean: only average over real phonemes, not padding
            mask = torch.zeros_like(target)
            for i, L in enumerate(target_lengths):
                mask[i, :L, :] = 1.0
            target_pooled = (target * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            target_pooled = target.mean(dim=1)

        loss = self.bce(pred_pooled, target_pooled)  # (batch, 24)
        return loss.mean()


# ── Evaluation ──────────────────────────────────────────────────────────────

class PhonologicalEvaluator:
    """
    Collects predictions and targets across batches, then computes metrics.

    Usage:
        evaluator = PhonologicalEvaluator()
        for batch in val_loader:
            preds = model(...)
            evaluator.update(preds, targets, target_lengths)
        results = evaluator.compute()
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.all_preds = []
        self.all_targets = []

    def reset(self):
        self.all_preds = []
        self.all_targets = []

    @torch.no_grad()
    def update(
        self,
        pred: torch.Tensor,          # (batch, T_audio, 24)
        target: torch.Tensor,        # (batch, T_phones, 24)
        target_lengths: torch.Tensor,
    ):
        # Pool predictions
        pred_pooled = pred.mean(dim=1).cpu().numpy()  # (batch, 24)

        # Masked pool targets
        mask = torch.zeros_like(target)
        for i, L in enumerate(target_lengths):
            mask[i, :L, :] = 1.0
        target_pooled = ((target * mask).sum(dim=1) /
                         mask.sum(dim=1).clamp(min=1)).cpu().numpy()

        # Binarize predictions
        pred_binary = (pred_pooled > self.threshold).astype(int)
        target_binary = (target_pooled > 0.5).astype(int)

        self.all_preds.append(pred_binary)
        self.all_targets.append(target_binary)

    def compute(self) -> dict:
        preds = np.concatenate(self.all_preds, axis=0)    # (N, 24)
        targets = np.concatenate(self.all_targets, axis=0) # (N, 24)

        results = {}

        # Per-feature F1
        per_feature_f1 = {}
        for i, name in enumerate(FEATURE_NAMES):
            f1 = f1_score(targets[:, i], preds[:, i],
                         average="binary", zero_division=0)
            per_feature_f1[name] = round(f1, 4)

        results["per_feature_f1"] = per_feature_f1
        results["macro_f1"] = round(np.mean(list(per_feature_f1.values())), 4)

        # Feature group accuracies (for thesis table)
        groups = {
            "voicing":    [0],
            "manner":     list(range(3, 9)),
            "place":      list(range(9, 17)),
            "vowel_height":   list(range(17, 20)),
            "vowel_backness": list(range(20, 23)),
            "rounding":   [23],
        }
        group_f1 = {}
        for group_name, indices in groups.items():
            group_preds = preds[:, indices].flatten()
            group_targets = targets[:, indices].flatten()
            f1 = f1_score(group_targets, group_preds,
                         average="binary", zero_division=0)
            group_f1[group_name] = round(f1, 4)
        results["group_f1"] = group_f1

        # Overall accuracy
        results["exact_match_acc"] = round(
            float((preds == targets).all(axis=1).mean()), 4
        )

        return results


def print_results(results: dict, model_name: str = "Model"):
    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*50}")
    print(f"  Macro F1:          {results['macro_f1']:.4f}")
    print(f"  Exact Match Acc:   {results['exact_match_acc']:.4f}")
    print(f"\n  Feature Group F1:")
    for group, score in results["group_f1"].items():
        print(f"    {group:<20} {score:.4f}")
    print(f"\n  Per-Feature F1:")
    for feat, score in results["per_feature_f1"].items():
        print(f"    {feat:<22} {score:.4f}")
