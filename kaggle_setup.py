"""
Kaggle Setup Script
===================
Run this cell FIRST in your Kaggle notebook before anything else.
Copy-paste each section into separate cells.
"""

# ═══════════════════════════════════════════════════════════════
# CELL 1 — Install dependencies
# ═══════════════════════════════════════════════════════════════

"""
!pip install -q transformers==4.40.0
!pip install -q tgt          # TextGrid parser for L2-Arctic
!pip install -q soundfile
!pip install -q resampy
!pip install -q scikit-learn
!pip install -q accelerate
"""

# ═══════════════════════════════════════════════════════════════
# CELL 2 — Verify GPU
# ═══════════════════════════════════════════════════════════════

"""
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
"""

# ═══════════════════════════════════════════════════════════════
# CELL 3 — Upload your project files OR clone from GitHub
# ═══════════════════════════════════════════════════════════════

"""
# Option A: if you have the files in a zip, upload via Kaggle Datasets
# Then they appear at /kaggle/input/your-dataset-name/

# Option B: clone from your GitHub repo (recommended)
!git clone https://github.com/YOUR_USERNAME/mdd_phonological.git
%cd mdd_phonological
"""

# ═══════════════════════════════════════════════════════════════
# CELL 4 — L2-Arctic data
# ═══════════════════════════════════════════════════════════════

"""
# L2-Arctic is freely available. Two ways to get it on Kaggle:

# Way 1: Upload as Kaggle Dataset (recommended — do this before your run)
#   - Download from: https://psi.engr.tamu.edu/l2-arctic-corpus/
#   - Upload to Kaggle Datasets
#   - Add as input to your notebook
#   - It will appear at /kaggle/input/l2-arctic/

# Way 2: Download directly in notebook (slow, ~10 min)
!wget -q "https://psi.engr.tamu.edu/l2-arctic-corpus/L2-ARCTIC.zip" -O /tmp/l2arctic.zip
!unzip -q /tmp/l2arctic.zip -d /kaggle/working/l2arctic
DATA_DIR = "/kaggle/working/l2arctic"
"""

# ═══════════════════════════════════════════════════════════════
# CELL 5 — Train Wav2Vec2
# ═══════════════════════════════════════════════════════════════

"""
!python train.py \
    --model wav2vec2 \
    --data_dir /kaggle/input/l2-arctic \
    --epochs 15
"""

# ═══════════════════════════════════════════════════════════════
# CELL 6 — Train Whisper (run after Wav2Vec2 finishes OR in separate session)
# ═══════════════════════════════════════════════════════════════

"""
!python train.py \
    --model whisper \
    --data_dir /kaggle/input/l2-arctic \
    --epochs 15
"""

# ═══════════════════════════════════════════════════════════════
# CELL 7 — Compare results side by side
# ═══════════════════════════════════════════════════════════════

"""
import json
import pandas as pd

with open("outputs/wav2vec2_results.json") as f:
    w2v_res = json.load(f)["test_results"]

with open("outputs/whisper_results.json") as f:
    wh_res = json.load(f)["test_results"]

# Summary table
summary = pd.DataFrame({
    "Metric": ["Macro F1", "Exact Match Acc",
               "Voicing F1", "Manner F1", "Place F1",
               "Vowel Height F1", "Vowel Backness F1"],
    "Wav2Vec2-large": [
        w2v_res["macro_f1"],
        w2v_res["exact_match_acc"],
        w2v_res["group_f1"]["voicing"],
        w2v_res["group_f1"]["manner"],
        w2v_res["group_f1"]["place"],
        w2v_res["group_f1"]["vowel_height"],
        w2v_res["group_f1"]["vowel_backness"],
    ],
    "Whisper-small": [
        wh_res["macro_f1"],
        wh_res["exact_match_acc"],
        wh_res["group_f1"]["voicing"],
        wh_res["group_f1"]["manner"],
        wh_res["group_f1"]["place"],
        wh_res["group_f1"]["vowel_height"],
        wh_res["group_f1"]["vowel_backness"],
    ],
})
print(summary.to_string(index=False))
"""
