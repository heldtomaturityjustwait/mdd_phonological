"""
L2-Arctic Dataset — Utterance Level
=====================================
Returns full utterances with phoneme sequence targets.
SCTC-SB loss handles the sequence alignment — no forced frame alignment needed.
"""

import os, json, random
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from pathlib import Path
import soundfile as sf
import tgt
from utils.phonological_map import get_feature_matrix, NUM_FEATURES

ALL_SPEAKERS = [
    "ABA", "ASI", "BWC", "ERMS", "EBVS", "HJK", "HQTV",
    "LXC", "MBMPS", "NCC", "NJS", "OGI", "PNV", "RKFL",
    "SKA", "SVBI", "THV", "TNI", "TXHC", "TLV", "YBAA",
    "YDCK", "YKWK", "ZHAA"
]

def load_audio(wav_path, target_sr=16000):
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import resampy
        audio = resampy.resample(audio, sr, target_sr)
    return audio.astype(np.float32)

def load_textgrid_phones(tg_path):
    tg = tgt.io.read_textgrid(tg_path)
    phones_tier = tg.get_tier_by_name("phones")
    phonemes = []
    for interval in phones_tier.intervals:
        label = interval.text.strip().upper().rstrip("012")
        if label and label not in ("", "SIL", "SP", "SPN", "<UNK>"):
            phonemes.append(label)
    return phonemes


class L2ArcticDataset(Dataset):
    def __init__(self, root_dir, speakers=None,
                 max_audio_len=10.0, cache_path=None):
        self.root_dir = Path(root_dir)
        self.speakers = speakers or ALL_SPEAKERS
        self.max_samples = int(max_audio_len * 16000)

        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached metadata from {cache_path}")
            with open(cache_path) as f:
                self.samples = json.load(f)
        else:
            self.samples = self._build_sample_list()
            if cache_path:
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump(self.samples, f)

        print(f"Dataset: {len(self.samples)} utterances, "
              f"{len(self.speakers)} speakers")

    def _build_sample_list(self):
        samples = []
        for speaker in self.speakers:
            spk_dir = self.root_dir / speaker
            if not spk_dir.exists():
                continue
            wav_dir = spk_dir / "wav"
            # Find TextGrid directory
            tg_dir = None
            for name in ["textgrid", "TextGrid", "annotation"]:
                c = spk_dir / name
                if c.exists():
                    tg_dir = c
                    break
            if not wav_dir.exists() or tg_dir is None:
                continue

            for wav_file in sorted(wav_dir.glob("*.wav")):
                utt_id = wav_file.stem
                tg_file = None
                for ext in [".TextGrid", ".textgrid"]:
                    c = tg_dir / f"{utt_id}{ext}"
                    if c.exists():
                        tg_file = c
                        break
                if tg_file is None:
                    continue
                try:
                    phonemes = load_textgrid_phones(str(tg_file))
                    if len(phonemes) < 2:
                        continue
                    samples.append({
                        "wav_path": str(wav_file),
                        "phonemes": phonemes,
                        "speaker": speaker,
                        "utt_id": utt_id,
                    })
                except Exception:
                    continue

        random.shuffle(samples)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        audio = load_audio(s["wav_path"])
        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]

        # (num_phonemes, 35) — binary feature matrix
        feat_matrix = get_feature_matrix(s["phonemes"])

        return {
            "audio": audio,
            "feature_matrix": feat_matrix.astype(np.float32),
            "num_phonemes": len(s["phonemes"]),
            "phonemes": s["phonemes"],
            "speaker": s["speaker"],
        }


def get_train_val_test_split(dataset, train=0.8, val=0.1, seed=42):
    random.seed(seed)
    speakers = list({s["speaker"] for s in dataset.samples})
    random.shuffle(speakers)
    n = len(speakers)
    n_train = int(n * train)
    n_val   = int(n * val)
    train_spk = set(speakers[:n_train])
    val_spk   = set(speakers[n_train:n_train + n_val])
    test_spk  = set(speakers[n_train + n_val:])

    train_idx = [i for i, s in enumerate(dataset.samples) if s["speaker"] in train_spk]
    val_idx   = [i for i, s in enumerate(dataset.samples) if s["speaker"] in val_spk]
    test_idx  = [i for i, s in enumerate(dataset.samples) if s["speaker"] in test_spk]

    print(f"Split — train: {len(train_idx)} ({len(train_spk)} spk) | "
          f"val: {len(val_idx)} ({len(val_spk)} spk) | "
          f"test: {len(test_idx)} ({len(test_spk)} spk)")
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)
