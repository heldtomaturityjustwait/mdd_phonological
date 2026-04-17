"""
ARPABET → Phonological Feature Vectors
Binary encoding of phonological features for all 40 ARPABET phonemes.

Feature dimensions (20 total):
  [0]  voiced
  [1]  consonant
  [2]  vowel
  [3]  manner: stop
  [4]  manner: fricative
  [5]  manner: affricate
  [6]  manner: nasal
  [7]  manner: approximant
  [8]  manner: lateral
  [9]  place: bilabial
  [10] place: labiodental
  [11] place: dental
  [12] place: alveolar
  [13] place: postalveolar
  [14] place: palatal
  [15] place: velar
  [16] place: glottal
  [17] vowel height: high
  [18] vowel height: mid
  [19] vowel height: low
  [20] vowel backness: front
  [21] vowel backness: central
  [22] vowel backness: back
  [23] rounded
"""

import numpy as np

# fmt: off
# Each row: phoneme → 24-dim binary feature vector
# Columns: voiced, consonant, vowel, stop, fricative, affricate, nasal,
#          approximant, lateral, bilabial, labiodental, dental, alveolar,
#          postalveolar, palatal, velar, glottal, high, mid, low,
#          front, central, back, rounded

PHONEME_FEATURE_MAP = {
    # ── Stops ──────────────────────────────────────────────────────────────
    "P":  [0,1,0, 1,0,0,0,0,0, 1,0,0,0,0,0,0,0, 0,0,0, 0,0,0, 0],
    "B":  [1,1,0, 1,0,0,0,0,0, 1,0,0,0,0,0,0,0, 0,0,0, 0,0,0, 0],
    "T":  [0,1,0, 1,0,0,0,0,0, 0,0,0,1,0,0,0,0, 0,0,0, 0,0,0, 0],
    "D":  [1,1,0, 1,0,0,0,0,0, 0,0,0,1,0,0,0,0, 0,0,0, 0,0,0, 0],
    "K":  [0,1,0, 1,0,0,0,0,0, 0,0,0,0,0,0,1,0, 0,0,0, 0,0,0, 0],
    "G":  [1,1,0, 1,0,0,0,0,0, 0,0,0,0,0,0,1,0, 0,0,0, 0,0,0, 0],

    # ── Fricatives ─────────────────────────────────────────────────────────
    "F":  [0,1,0, 0,1,0,0,0,0, 0,1,0,0,0,0,0,0, 0,0,0, 0,0,0, 0],
    "V":  [1,1,0, 0,1,0,0,0,0, 0,1,0,0,0,0,0,0, 0,0,0, 0,0,0, 0],
    "TH": [0,1,0, 0,1,0,0,0,0, 0,0,1,0,0,0,0,0, 0,0,0, 0,0,0, 0],
    "DH": [1,1,0, 0,1,0,0,0,0, 0,0,1,0,0,0,0,0, 0,0,0, 0,0,0, 0],
    "S":  [0,1,0, 0,1,0,0,0,0, 0,0,0,1,0,0,0,0, 0,0,0, 0,0,0, 0],
    "Z":  [1,1,0, 0,1,0,0,0,0, 0,0,0,1,0,0,0,0, 0,0,0, 0,0,0, 0],
    "SH": [0,1,0, 0,1,0,0,0,0, 0,0,0,0,1,0,0,0, 0,0,0, 0,0,0, 0],
    "ZH": [1,1,0, 0,1,0,0,0,0, 0,0,0,0,1,0,0,0, 0,0,0, 0,0,0, 0],
    "HH": [0,1,0, 0,1,0,0,0,0, 0,0,0,0,0,0,0,1, 0,0,0, 0,0,0, 0],

    # ── Affricates ─────────────────────────────────────────────────────────
    "CH": [0,1,0, 0,0,1,0,0,0, 0,0,0,0,1,0,0,0, 0,0,0, 0,0,0, 0],
    "JH": [1,1,0, 0,0,1,0,0,0, 0,0,0,0,1,0,0,0, 0,0,0, 0,0,0, 0],

    # ── Nasals ─────────────────────────────────────────────────────────────
    "M":  [1,1,0, 0,0,0,1,0,0, 1,0,0,0,0,0,0,0, 0,0,0, 0,0,0, 0],
    "N":  [1,1,0, 0,0,0,1,0,0, 0,0,0,1,0,0,0,0, 0,0,0, 0,0,0, 0],
    "NG": [1,1,0, 0,0,0,1,0,0, 0,0,0,0,0,0,1,0, 0,0,0, 0,0,0, 0],

    # ── Approximants ───────────────────────────────────────────────────────
    "W":  [1,1,0, 0,0,0,0,1,0, 1,0,0,0,0,0,1,0, 0,0,0, 0,0,0, 0],
    "Y":  [1,1,0, 0,0,0,0,1,0, 0,0,0,0,0,1,0,0, 0,0,0, 0,0,0, 0],
    "R":  [1,1,0, 0,0,0,0,1,0, 0,0,0,0,1,0,0,0, 0,0,0, 0,0,0, 0],

    # ── Laterals ───────────────────────────────────────────────────────────
    "L":  [1,1,0, 0,0,0,0,0,1, 0,0,0,1,0,0,0,0, 0,0,0, 0,0,0, 0],

    # ── Vowels (monophthongs) ───────────────────────────────────────────────
    # voiced, not consonant, vowel, no manner, no place, height, backness, rounded
    "IY": [1,0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 1,0,0, 1,0,0, 0],  # fleece
    "IH": [1,0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 1,0,0, 1,0,0, 0],  # kit
    "EH": [1,0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,1,0, 1,0,0, 0],  # dress
    "AE": [1,0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,1, 1,0,0, 0],  # trap
    "AA": [1,0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,1, 0,0,1, 0],  # lot
    "AO": [1,0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,1, 0,0,1, 1],  # thought
    "UH": [1,0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 1,0,0, 0,0,1, 1],  # foot
    "UW": [1,0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 1,0,0, 0,0,1, 1],  # goose
    "AH": [1,0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,1,0, 0,1,0, 0],  # strut/schwa
    "ER": [1,0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,1,0, 0,1,0, 0],  # nurse

    # ── Diphthongs ─────────────────────────────────────────────────────────
    "EY": [1,0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,1,0, 1,0,0, 0],  # face
    "AY": [1,0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,1, 1,0,0, 0],  # price
    "OY": [1,0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,1, 0,0,1, 1],  # choice
    "AW": [1,0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,1, 0,0,1, 0],  # mouth
    "OW": [1,0,1, 0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,1,0, 0,0,1, 1],  # goat
}
# fmt: on

FEATURE_NAMES = [
    "voiced", "consonant", "vowel",
    "stop", "fricative", "affricate", "nasal", "approximant", "lateral",
    "bilabial", "labiodental", "dental", "alveolar", "postalveolar",
    "palatal", "velar", "glottal",
    "high", "mid", "low",
    "front", "central", "back",
    "rounded",
]

NUM_FEATURES = len(FEATURE_NAMES)  # 24
ALL_PHONEMES = list(PHONEME_FEATURE_MAP.keys())
NUM_PHONEMES = len(ALL_PHONEMES)   # 40

PHONEME_TO_IDX = {p: i for i, p in enumerate(ALL_PHONEMES)}


def get_feature_vector(phoneme: str) -> np.ndarray:
    """Return 24-dim binary feature vector for a phoneme."""
    phoneme = phoneme.upper().rstrip("012")  # strip stress markers
    if phoneme not in PHONEME_FEATURE_MAP:
        return np.zeros(NUM_FEATURES, dtype=np.float32)
    return np.array(PHONEME_FEATURE_MAP[phoneme], dtype=np.float32)


def get_feature_matrix(phoneme_sequence: list) -> np.ndarray:
    """Return (T, 24) feature matrix for a sequence of phonemes."""
    return np.stack([get_feature_vector(p) for p in phoneme_sequence])


def decode_features_to_description(feature_vector: np.ndarray) -> dict:
    """
    Convert a predicted feature vector to a human-readable diagnosis.
    Used for generating feedback to the user.
    """
    desc = {}
    f = feature_vector

    desc["voiced"] = bool(f[0] > 0.5)
    desc["type"] = "vowel" if f[2] > 0.5 else "consonant"

    if f[2] < 0.5:  # consonant
        manners = ["stop","fricative","affricate","nasal","approximant","lateral"]
        manner_scores = f[3:9]
        desc["manner"] = manners[int(manner_scores.argmax())] if manner_scores.max() > 0.3 else "unknown"

        places = ["bilabial","labiodental","dental","alveolar",
                  "postalveolar","palatal","velar","glottal"]
        place_scores = f[9:17]
        desc["place"] = places[int(place_scores.argmax())] if place_scores.max() > 0.3 else "unknown"
    else:  # vowel
        heights = ["high","mid","low"]
        height_scores = f[17:20]
        desc["height"] = heights[int(height_scores.argmax())] if height_scores.max() > 0.3 else "unknown"

        backness = ["front","central","back"]
        back_scores = f[20:23]
        desc["backness"] = backness[int(back_scores.argmax())] if back_scores.max() > 0.3 else "unknown"

        desc["rounded"] = bool(f[23] > 0.5)

    return desc
