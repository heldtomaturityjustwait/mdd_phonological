"""
SCTC-SB Loss and Evaluation Metrics
=====================================
SCTC-SB = Separable CTC with Shared Blank
From: Shahin & Ahmed, Interspeech 2024

Key idea: treat each of the 35 phonological features as a separate
binary CTC sequence. Compute CTC loss for each feature independently,
then multiply the conditional probabilities (= sum the log probs).
Share a single blank token across all features.

This is the correct loss for utterance-level phonological feature
sequence prediction — it handles alignment automatically without
needing forced frame-level labels.

Evaluation uses per-feature accuracy and FAR/FRR as in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
from utils.phonological_map import FEATURE_NAMES, NUM_FEATURES


# ── SCTC-SB Loss ─────────────────────────────────────────────────────────────

class SCTCSBLoss(nn.Module):
    """
    Separable CTC with Shared Blank loss.

    For each of the 35 features, treats it as a binary (0/1) CTC problem:
    - Class 0: feature absent
    - Class 1: feature present
    - Class 2: blank (shared across all features)

    The model output (batch, T, 35) is split into per-feature logits.
    For feature i, we construct a 3-class output:
      [logit_absent, logit_present, blank_logit]
    where blank_logit is a shared learned parameter.

    SCTC-SB loss = sum of per-feature CTC losses
    (equivalent to product of conditional probabilities)
    """

    def __init__(self):
        super().__init__()
        # Shared blank logit — learned scalar
        self.blank_logit = nn.Parameter(torch.zeros(1))
        self.ctc_loss = nn.CTCLoss(blank=2, zero_infinity=True,
                                   reduction="mean")

    def forward(self, logits, targets, input_lengths, target_lengths):
        """
        Args:
            logits: (batch, T, 35) — raw logits from model
            targets: (batch, max_seq_len, 35) — binary feature sequences
            input_lengths: (batch,) — actual T per sample
            target_lengths: (batch,) — actual phoneme count per sample
        Returns:
            scalar loss
        """
        batch, T, _ = logits.shape
        total_loss = 0.0

        # Expand blank logit to (batch, T, 1)
        blank = self.blank_logit.expand(batch, T, 1)

        for feat_idx in range(NUM_FEATURES):
            # Per-feature logits: (batch, T, 2) — [absent, present]
            feat_logits = logits[:, :, feat_idx].unsqueeze(-1)  # (batch, T, 1)
            # Negate for absent class (present logit vs zero baseline)
            absent_logits = -feat_logits
            # Stack: [absent, present, blank] → (batch, T, 3)
            three_class = torch.cat([absent_logits, feat_logits, blank], dim=-1)
            # Log softmax over 3 classes
            log_probs = F.log_softmax(three_class, dim=-1)
            # CTC expects (T, batch, C)
            log_probs = log_probs.transpose(0, 1)

            # Targets for this feature: extract binary sequence per sample
            # targets[:, :, feat_idx] → (batch, max_seq)
            # Values are 0 or 1 → class indices for CTC
            feat_targets = targets[:, :, feat_idx].long()  # (batch, max_seq)

            # Flatten targets for CTCLoss (expects 1D concatenated)
            flat_targets = []
            for b in range(batch):
                L = target_lengths[b].item()
                flat_targets.append(feat_targets[b, :L])
            flat_targets = torch.cat(flat_targets, dim=0)

            feat_loss = self.ctc_loss(
                log_probs,
                flat_targets,
                input_lengths,
                target_lengths,
            )
            total_loss = total_loss + feat_loss

        return total_loss / NUM_FEATURES


# ── Evaluation ────────────────────────────────────────────────────────────────

class PhonologicalEvaluator:
    """
    Evaluation matching Shahin & Ahmed (2024):
    - Per-feature accuracy
    - FAR (False Acceptance Rate) = mispronounced accepted as correct
    - FRR (False Rejection Rate) = correct rejected as mispronounced
    - Macro F1 across features

    For phonological feature detection (not MDD), FAR/FRR measure
    how often the model incorrectly predicts absent/present.
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.all_preds = []    # list of (N, 35) arrays
        self.all_targets = []  # list of (N, 35) arrays

    @torch.no_grad()
    def update(self, logits, targets, target_lengths=None):
        """
        logits: (batch, T, 35) — raw model output
        targets: (batch, max_seq, 35) OR (batch, 35) — binary feature labels
        """
        # Pool logits over time → (batch, 35)
        probs = torch.sigmoid(logits.mean(dim=1)).cpu().numpy()
        pred_binary = (probs > self.threshold).astype(int)

        if targets.dim() == 3:
            # Utterance-level: pool targets too
            if target_lengths is not None:
                mask = torch.zeros_like(targets)
                for i, L in enumerate(target_lengths):
                    mask[i, :L, :] = 1.0
                tgt = ((targets * mask).sum(dim=1) /
                       mask.sum(dim=1).clamp(min=1)).cpu().numpy()
            else:
                tgt = targets.mean(dim=1).cpu().numpy()
        else:
            tgt = targets.cpu().numpy()

        tgt_binary = (tgt > 0.5).astype(int)
        self.all_preds.append(pred_binary)
        self.all_targets.append(tgt_binary)

    def compute(self):
        preds = np.concatenate(self.all_preds, axis=0)    # (N, 35)
        targets = np.concatenate(self.all_targets, axis=0) # (N, 35)

        results = {}

        # Per-feature metrics
        per_feature = {}
        for i, name in enumerate(FEATURE_NAMES):
            p, t = preds[:, i], targets[:, i]
            f1 = f1_score(t, p, average="binary", zero_division=0)
            acc = float((p == t).mean())
            # FAR: predicted present (1) but target absent (0)
            absent_mask = (t == 0)
            far = float(p[absent_mask].mean()) if absent_mask.any() else 0.0
            # FRR: predicted absent (0) but target present (1)
            present_mask = (t == 1)
            frr = float((1 - p[present_mask]).mean()) if present_mask.any() else 0.0
            per_feature[name] = {
                "f1": round(f1, 4),
                "acc": round(acc, 4),
                "far": round(far, 4),
                "frr": round(frr, 4),
            }

        results["per_feature"] = per_feature
        results["macro_f1"] = round(
            np.mean([v["f1"] for v in per_feature.values()]), 4)
        results["macro_acc"] = round(
            np.mean([v["acc"] for v in per_feature.values()]), 4)
        results["macro_far"] = round(
            np.mean([v["far"] for v in per_feature.values()]), 4)
        results["macro_frr"] = round(
            np.mean([v["frr"] for v in per_feature.values()]), 4)
        results["exact_match_acc"] = round(
            float((preds == targets).all(axis=1).mean()), 4)

        return results


def print_results(results, model_name="Model"):
    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    print(f"  Macro F1:          {results['macro_f1']:.4f}")
    print(f"  Macro Accuracy:    {results['macro_acc']:.4f}")
    print(f"  Macro FAR:         {results['macro_far']:.4f}")
    print(f"  Macro FRR:         {results['macro_frr']:.4f}")
    print(f"  Exact Match Acc:   {results['exact_match_acc']:.4f}")
    print(f"\n  Per-Feature (F1 / ACC / FAR / FRR):")
    for feat, m in results["per_feature"].items():
        print(f"    {feat:<16} F1={m['f1']:.3f}  "
              f"ACC={m['acc']:.3f}  FAR={m['far']:.3f}  FRR={m['frr']:.3f}")
