"""
Training Script
===============
Trains either Wav2Vec2ForPhonology or WhisperForPhonology on L2-Arctic.

Usage (Kaggle):
    # Train Wav2Vec2
    python train.py --model wav2vec2 --data_dir /kaggle/input/l2-arctic

    # Train Whisper
    python train.py --model whisper --data_dir /kaggle/input/l2-arctic

    # Resume from checkpoint
    python train.py --model wav2vec2 --resume outputs/wav2vec2_best.pt
"""

import os
import sys
import json
import argparse
import time
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from data.dataset import L2ArcticDataset, get_train_val_test_split
from data.collators import Wav2Vec2Collator, WhisperCollator
from models.phonological_models import Wav2Vec2ForPhonology, WhisperForPhonology
from utils.metrics import PhonologicalLoss, PhonologicalEvaluator, print_results


# ── Config ──────────────────────────────────────────────────────────────────

def get_config(model_type: str) -> dict:
    base = {
        "seed": 42,
        "num_epochs": 20,
        "eval_every": 1,          # eval every N epochs
        "save_dir": "outputs",
        "gradient_accumulation": 4,
        "max_grad_norm": 1.0,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "num_workers": 2,
        "pin_memory": True,
    }
    if model_type == "wav2vec2":
        base.update({
            "model_name": "facebook/wav2vec2-large",
            "batch_size": 4,      # T4: 16GB, wav2vec2-large needs small batch
            "learning_rate": 1e-4,
            "collator_class": "Wav2Vec2Collator",
        })
    else:  # whisper
        base.update({
            "model_name": "openai/whisper-small",
            "batch_size": 8,      # Whisper encoder lighter than wav2vec2-large
            "learning_rate": 5e-5,
            "collator_class": "WhisperCollator",
        })
    return base


# ── Training step ────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, scaler,
                loss_fn, config, device, model_type, epoch):
    model.train()
    total_loss = 0.0
    steps = 0
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        # Move inputs to device
        if model_type == "wav2vec2":
            inputs = {
                "input_values": batch["input_values"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
        else:
            inputs = {"input_features": batch["input_features"].to(device)}

        phon_targets = batch["phon_targets"].to(device)
        phon_lengths = batch["phon_lengths"].to(device)

        # Forward
        with autocast():
            pred = model(**inputs)
            loss = loss_fn(pred, phon_targets, target_lengths=phon_lengths)
            loss = loss / config["gradient_accumulation"]

        scaler.scale(loss).backward()

        # Gradient accumulation step
        if (i + 1) % config["gradient_accumulation"] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["max_grad_norm"]
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            steps += 1

        total_loss += loss.item() * config["gradient_accumulation"]

        if i % 20 == 0:
            print(f"  Epoch {epoch} | Step {i}/{len(loader)} | "
                  f"Loss: {loss.item() * config['gradient_accumulation']:.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, loss_fn, evaluator, device, model_type):
    model.eval()
    evaluator.reset()
    total_loss = 0.0

    for batch in loader:
        if model_type == "wav2vec2":
            inputs = {
                "input_values": batch["input_values"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
        else:
            inputs = {"input_features": batch["input_features"].to(device)}

        phon_targets = batch["phon_targets"].to(device)
        phon_lengths = batch["phon_lengths"].to(device)

        with autocast():
            pred = model(**inputs)
            loss = loss_fn(pred, phon_targets, target_lengths=phon_lengths)

        evaluator.update(pred, phon_targets, phon_lengths)
        total_loss += loss.item()

    results = evaluator.compute()
    results["val_loss"] = round(total_loss / len(loader), 4)
    return results


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["wav2vec2", "whisper"],
                        required=True)
    parser.add_argument("--data_dir", type=str, default="/kaggle/input/l2-arctic")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = get_config(args.model)
    if args.epochs:
        config["num_epochs"] = args.epochs

    # Reproducibility
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    os.makedirs(config["save_dir"], exist_ok=True)

    # ── Dataset ──
    print("\nLoading L2-Arctic dataset...")
    dataset = L2ArcticDataset(
        root_dir=args.data_dir,
        cache_path=f"{config['save_dir']}/metadata_cache.json",
    )
    train_set, val_set, test_set = get_train_val_test_split(dataset)

    # ── Collators ──
    if args.model == "wav2vec2":
        collator = Wav2Vec2Collator(config["model_name"])
    else:
        collator = WhisperCollator(config["model_name"])

    train_loader = DataLoader(
        train_set, batch_size=config["batch_size"], shuffle=True,
        collate_fn=collator, num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )
    val_loader = DataLoader(
        val_set, batch_size=config["batch_size"] * 2, shuffle=False,
        collate_fn=collator, num_workers=config["num_workers"],
    )
    test_loader = DataLoader(
        test_set, batch_size=config["batch_size"] * 2, shuffle=False,
        collate_fn=collator, num_workers=config["num_workers"],
    )

    # ── Model ──
    if args.model == "wav2vec2":
        model = Wav2Vec2ForPhonology(config["model_name"])
    else:
        model = WhisperForPhonology(config["model_name"])
    model = model.to(device)

    # ── Optimizer & scheduler ──
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    total_steps = (len(train_loader) // config["gradient_accumulation"]
                   * config["num_epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    scaler = GradScaler()

    loss_fn = PhonologicalLoss()
    evaluator = PhonologicalEvaluator()

    # ── Resume ──
    start_epoch = 1
    best_macro_f1 = 0.0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 1) + 1
        best_macro_f1 = ckpt.get("best_macro_f1", 0.0)

    history = []
    save_path = f"{config['save_dir']}/{args.model}_best.pt"

    # ── Training loop ──
    print(f"\nStarting training: {config['num_epochs']} epochs\n")
    for epoch in range(start_epoch, config["num_epochs"] + 1):
        t0 = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            scaler, loss_fn, config, device, args.model, epoch
        )

        if epoch % config["eval_every"] == 0:
            val_results = evaluate(
                model, val_loader, loss_fn, evaluator, device, args.model
            )
            elapsed = time.time() - t0
            print(f"\nEpoch {epoch}/{config['num_epochs']} "
                  f"({elapsed:.0f}s) | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_results['val_loss']:.4f} | "
                  f"Macro F1: {val_results['macro_f1']:.4f}")

            history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                **val_results,
            })

            # Save best
            if val_results["macro_f1"] > best_macro_f1:
                best_macro_f1 = val_results["macro_f1"]
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_macro_f1": best_macro_f1,
                    "config": config,
                    "val_results": val_results,
                }, save_path)
                print(f"  ✓ Saved best model (macro F1: {best_macro_f1:.4f})")

    # ── Final test evaluation ──
    print("\nLoading best model for test evaluation...")
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    test_results = evaluate(
        model, test_loader, loss_fn, evaluator, device, args.model
    )
    print_results(test_results, model_name=f"{args.model.upper()} (Test)")

    # Save results
    results_path = f"{config['save_dir']}/{args.model}_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "config": config,
            "history": history,
            "test_results": test_results,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
