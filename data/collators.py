"""Collators for Wav2Vec2 and Whisper."""

import torch
import numpy as np
from transformers import Wav2Vec2Processor, WhisperProcessor
from utils.phonological_map import NUM_FEATURES


class Wav2Vec2Collator:
    def __init__(self, model_name="facebook/wav2vec2-large-robust"):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

    def __call__(self, batch):
        audios = [item["audio"] for item in batch]
        inputs = self.processor(
            audios, sampling_rate=16000,
            return_tensors="pt", padding=True,
            return_attention_mask=True,
        )

        # Pad feature matrices to max phoneme length in batch
        max_ph = max(item["num_phonemes"] for item in batch)
        feat_padded = torch.zeros(len(batch), max_ph, NUM_FEATURES)
        ph_lengths  = torch.zeros(len(batch), dtype=torch.long)

        for i, item in enumerate(batch):
            fm = torch.tensor(item["feature_matrix"])
            n  = fm.shape[0]
            feat_padded[i, :n, :] = fm
            ph_lengths[i] = n

        # Input lengths in frames (wav2vec2 downsamples by ~320)
        input_lengths = torch.tensor(
            [v.shape[-1] // 320 for v in inputs.input_values],
            dtype=torch.long
        )

        return {
            "input_values":   inputs.input_values,
            "attention_mask": inputs.get(
                "attention_mask",
                torch.ones_like(inputs.input_values)
            ),
            "phon_targets":    feat_padded,
            "target_lengths":  ph_lengths,
            "input_lengths":   input_lengths,
            "phonemes": [item["phonemes"] for item in batch],
            "speaker":  [item["speaker"]  for item in batch],
        }


class WhisperCollator:
    def __init__(self, model_name="openai/whisper-small"):
        self.processor = WhisperProcessor.from_pretrained(model_name)

    def __call__(self, batch):
        audios = [item["audio"] for item in batch]
        inputs = self.processor(
            audios, sampling_rate=16000,
            return_tensors="pt", padding=True,
        )

        max_ph = max(item["num_phonemes"] for item in batch)
        feat_padded = torch.zeros(len(batch), max_ph, NUM_FEATURES)
        ph_lengths  = torch.zeros(len(batch), dtype=torch.long)

        for i, item in enumerate(batch):
            fm = torch.tensor(item["feature_matrix"])
            n  = fm.shape[0]
            feat_padded[i, :n, :] = fm
            ph_lengths[i] = n

        # Whisper encoder output length: input_features is always 3000 frames
        # but actual content length depends on audio duration
        # Use fixed 1500 (Whisper encoder downsamples 3000 → 1500)
        input_lengths = torch.full(
            (len(batch),), 1500, dtype=torch.long
        )

        return {
            "input_features": inputs.input_features,
            "phon_targets":   feat_padded,
            "target_lengths": ph_lengths,
            "input_lengths":  input_lengths,
            "phonemes": [item["phonemes"] for item in batch],
            "speaker":  [item["speaker"]  for item in batch],
        }
