"""
Collate functions for DataLoader batching.
Wav2Vec2 and Whisper need different input preparation.
"""

import torch
import numpy as np
from transformers import Wav2Vec2Processor, WhisperProcessor


# ── Wav2Vec2 collator ───────────────────────────────────────────────────────

class Wav2Vec2Collator:
    def __init__(self, model_name="facebook/wav2vec2-large"):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

    def __call__(self, batch):
        audios = [item["audio"] for item in batch]
        feature_matrices = [item["feature_matrix"] for item in batch]

        # Pad audio to same length
        inputs = self.processor(
            audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )

        # Pad phoneme feature matrices to same num_phonemes
        max_phonemes = max(fm.shape[0] for fm in feature_matrices)
        num_features = feature_matrices[0].shape[1]

        padded_features = torch.zeros(len(batch), max_phonemes, num_features)
        feature_lengths = torch.zeros(len(batch), dtype=torch.long)

        for i, fm in enumerate(feature_matrices):
            n = fm.shape[0]
            padded_features[i, :n, :] = torch.tensor(fm)
            feature_lengths[i] = n

        return {
            "input_values": inputs.input_values,
            "attention_mask": inputs.attention_mask,
            "phon_targets": padded_features,
            "phon_lengths": feature_lengths,
            "phonemes": [item["phonemes"] for item in batch],
            "speaker": [item["speaker"] for item in batch],
        }


# ── Whisper collator ────────────────────────────────────────────────────────

class WhisperCollator:
    def __init__(self, model_name="openai/whisper-small"):
        self.processor = WhisperProcessor.from_pretrained(model_name)

    def __call__(self, batch):
        audios = [item["audio"] for item in batch]
        feature_matrices = [item["feature_matrix"] for item in batch]

        # Whisper processor pads to 30s mel spectrogram internally
        inputs = self.processor(
            audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )

        max_phonemes = max(fm.shape[0] for fm in feature_matrices)
        num_features = feature_matrices[0].shape[1]

        padded_features = torch.zeros(len(batch), max_phonemes, num_features)
        feature_lengths = torch.zeros(len(batch), dtype=torch.long)

        for i, fm in enumerate(feature_matrices):
            n = fm.shape[0]
            padded_features[i, :n, :] = torch.tensor(fm)
            feature_lengths[i] = n

        return {
            "input_features": inputs.input_features,
            "phon_targets": padded_features,
            "phon_lengths": feature_lengths,
            "phonemes": [item["phonemes"] for item in batch],
            "speaker": [item["speaker"] for item in batch],
        }
