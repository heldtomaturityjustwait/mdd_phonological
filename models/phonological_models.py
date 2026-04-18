"""
Phonological Feature Models — following Shahin & Ahmed (Interspeech 2024)
==========================================================================
Architecture: Encoder → Linear layer → SCTC-SB loss

Key design decisions from the paper:
- Single LINEAR layer on top of wav2vec2 (not a deep MLP)
- 35 phonological features as separate binary sequences
- CTC-style sequence output (not utterance pooling)
- wav2vec2-large-robust for Wav2Vec2 model
- We add Whisper-small as a novel comparison

Both models output (batch, time, 35) logits — one per feature per frame.
SCTC-SB loss handles the multilabel sequence alignment.
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, WhisperModel
from utils.phonological_map import NUM_FEATURES


class Wav2Vec2ForPhonology(nn.Module):
    """
    Wav2Vec2-large-robust + single linear layer.
    Follows Shahin & Ahmed (2024) exactly.
    """
    def __init__(self, model_name="facebook/wav2vec2-large-robust"):
        super().__init__()
        print(f"Loading {model_name}...")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        hidden_size = self.wav2vec2.config.hidden_size  # 1024

        # Freeze CNN feature extractor
        self.wav2vec2.feature_extractor._freeze_parameters()

        # Freeze bottom half of transformer layers
        num_layers = len(self.wav2vec2.encoder.layers)
        freeze_until = num_layers // 2
        for i, layer in enumerate(self.wav2vec2.encoder.layers):
            if i < freeze_until:
                for param in layer.parameters():
                    param.requires_grad = False

        # Single linear layer — exactly as in the paper
        self.linear = nn.Linear(hidden_size, NUM_FEATURES)

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Wav2Vec2ForPhonology — total: {total/1e6:.1f}M | "
              f"trainable: {trainable/1e6:.1f}M")

    def forward(self, input_values, attention_mask=None):
        """
        Returns:
            logits: (batch, time, 35) — raw logits for SCTC-SB loss
        """
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state  # (batch, T, 1024)
        return self.linear(hidden)           # (batch, T, 35)


class WhisperForPhonology(nn.Module):
    """
    Whisper-small encoder + single linear layer.
    Novel comparison against Wav2Vec2 — our contribution beyond the paper.
    """
    def __init__(self, model_name="openai/whisper-small"):
        super().__init__()
        print(f"Loading {model_name}...")
        whisper = WhisperModel.from_pretrained(model_name)
        self.encoder = whisper.encoder
        hidden_size = whisper.config.d_model  # 768

        # Freeze convolutional stem
        for param in self.encoder.conv1.parameters():
            param.requires_grad = False
        for param in self.encoder.conv2.parameters():
            param.requires_grad = False

        # Freeze bottom 2/3 of encoder layers
        num_layers = len(self.encoder.layers)
        freeze_until = int(num_layers * 0.66)
        for i, layer in enumerate(self.encoder.layers):
            if i < freeze_until:
                for param in layer.parameters():
                    param.requires_grad = False

        # Single linear layer — same as Wav2Vec2 model for fair comparison
        self.linear = nn.Linear(hidden_size, NUM_FEATURES)

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"WhisperForPhonology — total: {total/1e6:.1f}M | "
              f"trainable: {trainable/1e6:.1f}M")

    def forward(self, input_features):
        """
        Returns:
            logits: (batch, time, 35) — raw logits for SCTC-SB loss
        """
        encoder_out = self.encoder(input_features)
        hidden = encoder_out.last_hidden_state  # (batch, T, 768)
        return self.linear(hidden)              # (batch, T, 35)
