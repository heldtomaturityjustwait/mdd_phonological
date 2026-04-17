"""
Phonological Feature Models
============================
Two architectures sharing the same phonological head design:
  1. Wav2Vec2ForPhonology   — wav2vec2-large encoder
  2. WhisperForPhonology    — whisper-small encoder (decoder discarded)

Both output per-frame phonological feature vectors (batch, time, 24).
The head is identical — only the encoder differs.
This ensures the comparison is fair: same head, same data, different encoder.
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, WhisperModel
from utils.phonological_map import NUM_FEATURES


# ── Shared phonological head ────────────────────────────────────────────────

class PhonologicalHead(nn.Module):
    """
    Maps encoder hidden states → binary phonological feature vectors.
    Input:  (batch, time, hidden_size)
    Output: (batch, time, NUM_FEATURES=24)  — sigmoid activated
    """
    def __init__(self, hidden_size: int, num_features: int = NUM_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_features),
            nn.Sigmoid(),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


# ── Model 1: Wav2Vec2 ────────────────────────────────────────────────────────

class Wav2Vec2ForPhonology(nn.Module):
    """
    wav2vec2-large encoder + phonological feature head.

    Freezing strategy:
      - CNN feature extractor: always frozen (as recommended by HuggingFace)
      - Bottom 12 transformer layers: frozen
      - Top 12 transformer layers: fine-tuned
      - Phonological head: fully trained
    """

    def __init__(self, model_name: str = "facebook/wav2vec2-large"):
        super().__init__()
        print(f"Loading {model_name}...")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        hidden_size = self.wav2vec2.config.hidden_size  # 1024 for large

        # Freeze CNN feature extractor
        self.wav2vec2.feature_extractor._freeze_parameters()

        # Freeze bottom half of transformer layers
        num_layers = len(self.wav2vec2.encoder.layers)
        freeze_until = num_layers // 2  # freeze bottom 12 of 24
        for i, layer in enumerate(self.wav2vec2.encoder.layers):
            if i < freeze_until:
                for param in layer.parameters():
                    param.requires_grad = False

        self.phon_head = PhonologicalHead(hidden_size)

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Wav2Vec2ForPhonology — total: {total/1e6:.1f}M | "
              f"trainable: {trainable/1e6:.1f}M")

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            input_values: (batch, audio_len) — raw waveform
            attention_mask: (batch, audio_len) — optional
        Returns:
            phon_logits: (batch, time_frames, 24) — sigmoid probabilities
        """
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state  # (batch, T, 1024)
        return self.phon_head(hidden)       # (batch, T, 24)


# ── Model 2: Whisper ─────────────────────────────────────────────────────────

class WhisperForPhonology(nn.Module):
    """
    whisper-small encoder + phonological feature head.
    Decoder is discarded entirely — we only use the encoder.

    Freezing strategy:
      - Bottom 8 encoder layers: frozen
      - Top 4 encoder layers: fine-tuned
      - Phonological head: fully trained
    """

    def __init__(self, model_name: str = "openai/whisper-small"):
        super().__init__()
        print(f"Loading {model_name}...")
        whisper = WhisperModel.from_pretrained(model_name)
        self.encoder = whisper.encoder
        hidden_size = whisper.config.d_model  # 768 for whisper-small

        # Freeze bottom layers
        num_layers = len(self.encoder.layers)
        freeze_until = int(num_layers * 0.66)  # freeze bottom 2/3
        for i, layer in enumerate(self.encoder.layers):
            if i < freeze_until:
                for param in layer.parameters():
                    param.requires_grad = False

        # Freeze convolutional stem
        for param in self.encoder.conv1.parameters():
            param.requires_grad = False
        for param in self.encoder.conv2.parameters():
            param.requires_grad = False

        self.phon_head = PhonologicalHead(hidden_size)

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"WhisperForPhonology — total: {total/1e6:.1f}M | "
              f"trainable: {trainable/1e6:.1f}M")

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_features: (batch, 80, 3000) — mel spectrogram from WhisperProcessor
        Returns:
            phon_logits: (batch, time_frames, 24) — sigmoid probabilities
        """
        encoder_out = self.encoder(input_features)
        hidden = encoder_out.last_hidden_state  # (batch, T, 768)
        return self.phon_head(hidden)            # (batch, T, 24)
