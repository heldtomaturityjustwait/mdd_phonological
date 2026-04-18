import torch
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, WhisperProcessor
from utils.phonological_map import NUM_FEATURES


class Wav2Vec2Collator:
    def __init__(self, model_name="facebook/wav2vec2-large-robust"):
        # Use FeatureExtractor only — robust model has no CTC tokenizer
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

    def __call__(self, batch):
        audios = [item["audio"] for item in batch]
        inputs = self.processor(
            audios, sampling_rate=16000,
            return_tensors="pt", padding=True,
            return_attention_mask=True,
        )

        max_ph = max(item["num_phonemes"] for item in batch)
        feat_padded = torch.zeros(len(batch), max_ph, NUM_FEATURES)
        ph_lengths  = torch.zeros(len(batch), dtype=torch.long)

        for i, item in enumerate(batch):
            fm = torch.tensor(item["feature_matrix"])
            n  = fm.shape[0]
            feat_padded[i, :n, :] = fm
            ph_lengths[i] = n

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
            "phon_targets":   feat_padded,
            "target_lengths": ph_lengths,
            "input_lengths":  input_lengths,
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
