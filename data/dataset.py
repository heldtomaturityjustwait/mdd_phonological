"""
L2-Arctic Dataset Loader
========================
Expects L2-Arctic in this folder structure:

l2arctic/
  YBAA/          ← speaker folder (L1: Arabic)
    wav/
      arctic_a0001.wav
      ...
    annotation/
      arctic_a0001.TextGrid   ← forced alignment with phoneme boundaries
  ZHAA/          ← speaker folder (L1: Mandarin)
    ...
  ...

Download: https://psi.engr.tamu.edu/l2-arctic-corpus/
All 24 speakers are used.

TextGrid format per utterance:
  - "words" tier: word-level intervals
  - "phones" tier: phoneme-level intervals with ARPABET labels
  - "mispronunciation" tier (some speakers): error annotations
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import soundfile as sf
import tgt  # pip install tgt — TextGrid parser


# ── Speaker list (all 24 L2-Arctic speakers) ───────────────────────────────
ALL_SPEAKERS = [
    "YBAA", "ZHAA", "BWC", "LXC", "HJK", "YDCK",  # Arabic, Mandarin, Korean
    "ERMS", "MBMPS", "NCC", "TXHC", "RKFL", "SVBI",  # Spanish, Hindi
    "OGI", "EBVS", "TNI", "HQTV", "PNV", "THV",   # Vietnamese
    "ABA", "SKA", "AAAA", "HQTV", "BWC", "ZHAA",  # others
]
# Use the actual speaker IDs from your downloaded copy — adjust if needed


def load_textgrid(tg_path: str):
    """Parse TextGrid and return list of (phoneme, start, end) tuples."""
    tg = tgt.io.read_textgrid(tg_path)
    phones_tier = tg.get_tier_by_name("phones")
    phonemes = []
    for interval in phones_tier.intervals:
        label = interval.text.strip().upper().rstrip("012")
        if label and label not in ("", "SIL", "SP", "SPN", "<UNK>"):
            phonemes.append({
                "phoneme": label,
                "start": interval.start_time,
                "end": interval.end_time,
            })
    return phonemes


def load_audio(wav_path: str, target_sr: int = 16000):
    """Load wav file, resample to 16kHz if needed, return numpy array."""
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo → mono
    if sr != target_sr:
        import resampy
        audio = resampy.resample(audio, sr, target_sr)
    return audio.astype(np.float32)


class L2ArcticDataset(Dataset):
    """
    Returns one utterance per item.
    Each item is a dict with:
      - audio: np.ndarray (T,) at 16kHz
      - phonemes: list of str (ARPABET)
      - feature_matrix: np.ndarray (num_phonemes, 24)
      - speaker: str
      - utt_id: str
    """

    def __init__(
        self,
        root_dir: str,
        speakers: list = None,
        max_audio_len: float = 10.0,  # seconds — skip very long utterances
        cache_path: str = None,       # optional: save/load parsed metadata
    ):
        self.root_dir = Path(root_dir)
        self.speakers = speakers or ALL_SPEAKERS
        self.max_samples = int(max_audio_len * 16000)

        from utils.phonological_map import get_feature_matrix
        self.get_feature_matrix = get_feature_matrix

        # Load or build metadata
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached metadata from {cache_path}")
            with open(cache_path) as f:
                self.samples = json.load(f)
        else:
            self.samples = self._build_sample_list()
            if cache_path:
                with open(cache_path, "w") as f:
                    json.dump(self.samples, f)
                print(f"Cached metadata to {cache_path}")

        print(f"Dataset: {len(self.samples)} utterances from "
              f"{len(self.speakers)} speakers")

    def _build_sample_list(self):
        samples = []
        for speaker in self.speakers:
            speaker_dir = self.root_dir / speaker
            if not speaker_dir.exists():
                print(f"  Warning: speaker dir not found: {speaker_dir}")
                continue

            wav_dir = speaker_dir / "wav"
            ann_dir = speaker_dir / "annotation"

            if not wav_dir.exists() or not ann_dir.exists():
                continue

            for wav_file in sorted(wav_dir.glob("*.wav")):
                utt_id = wav_file.stem
                tg_file = ann_dir / f"{utt_id}.TextGrid"

                if not tg_file.exists():
                    continue

                try:
                    phonemes_info = load_textgrid(str(tg_file))
                    if len(phonemes_info) < 2:
                        continue

                    samples.append({
                        "wav_path": str(wav_file),
                        "tg_path": str(tg_file),
                        "phonemes": [p["phoneme"] for p in phonemes_info],
                        "speaker": speaker,
                        "utt_id": utt_id,
                    })
                except Exception as e:
                    print(f"  Skipping {utt_id}: {e}")

        random.shuffle(samples)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        audio = load_audio(s["wav_path"])

        # Truncate if too long
        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]

        feature_matrix = self.get_feature_matrix(s["phonemes"])

        return {
            "audio": audio,
            "phonemes": s["phonemes"],
            "feature_matrix": feature_matrix,  # (num_phonemes, 24)
            "speaker": s["speaker"],
            "utt_id": s["utt_id"],
        }


def get_train_val_test_split(dataset, train=0.8, val=0.1, seed=42):
    """Speaker-independent split: hold out speakers for val/test."""
    random.seed(seed)
    speakers = list({s["speaker"] for s in dataset.samples})
    random.shuffle(speakers)

    n = len(speakers)
    n_train = int(n * train)
    n_val = int(n * val)

    train_spk = set(speakers[:n_train])
    val_spk = set(speakers[n_train:n_train + n_val])
    test_spk = set(speakers[n_train + n_val:])

    train_idx = [i for i, s in enumerate(dataset.samples)
                 if s["speaker"] in train_spk]
    val_idx   = [i for i, s in enumerate(dataset.samples)
                 if s["speaker"] in val_spk]
    test_idx  = [i for i, s in enumerate(dataset.samples)
                 if s["speaker"] in test_spk]

    print(f"Split — train: {len(train_idx)} utts ({len(train_spk)} spk) | "
          f"val: {len(val_idx)} utts ({len(val_spk)} spk) | "
          f"test: {len(test_idx)} utts ({len(test_spk)} spk)")

    from torch.utils.data import Subset
    return (Subset(dataset, train_idx),
            Subset(dataset, val_idx),
            Subset(dataset, test_idx))
