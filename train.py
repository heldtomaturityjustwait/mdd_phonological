import os, sys, json, argparse, time, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset   import L2ArcticDataset, get_train_val_test_split
from data.collators import Wav2Vec2Collator, WhisperCollator
from models.phonological_models import Wav2Vec2ForPhonology, WhisperForPhonology
from utils.metrics  import SCTCSBLoss, PhonologicalEvaluator, print_results


def get_config(model_type):
    base = dict(
        seed=42, num_epochs=20, eval_every=1,
        save_dir="outputs", gradient_accumulation=4,
        max_grad_norm=1.0, weight_decay=0.01, num_workers=2,
    )
    if model_type == "wav2vec2":
        base.update(dict(
            model_name="facebook/wav2vec2-large-robust",
            batch_size=4, learning_rate=1e-4,
        ))
    else:
        base.update(dict(
            model_name="openai/whisper-small",
            batch_size=8, learning_rate=5e-5,
        ))
    return base


def train_epoch(model, loader, optimizer, scheduler, scaler,
                loss_fn, config, device, model_type, epoch):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        if model_type == "wav2vec2":
            inputs = dict(
                input_values=batch["input_values"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
        else:
            inputs = dict(input_features=batch["input_features"].to(device))

        phon_targets   = batch["phon_targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        input_lengths  = batch["input_lengths"].to(device)

        with autocast("cuda"):
            logits = model(**inputs)
            T = logits.shape[1]
            clamped_il = input_lengths.clamp(max=T)
            loss = loss_fn(logits, phon_targets,
                           clamped_il, target_lengths)
            loss = loss / config["gradient_accumulation"]

        scaler.scale(loss).backward()

        if (i + 1) % config["gradient_accumulation"] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * config["gradient_accumulation"]

        if i % 20 == 0:
            print(f"  Epoch {epoch} | Step {i}/{len(loader)} | "
                  f"Loss: {loss.item()*config['gradient_accumulation']:.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, loss_fn, evaluator, device, model_type):
    model.eval()
    evaluator.reset()
    total_loss = 0.0

    for batch in loader:
        if model_type == "wav2vec2":
            inputs = dict(
                input_values=batch["input_values"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
        else:
            inputs = dict(input_features=batch["input_features"].to(device))

        phon_targets   = batch["phon_targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        input_lengths  = batch["input_lengths"].to(device)

        with autocast("cuda"):
            logits = model(**inputs)
            T = logits.shape[1]
            clamped_il = input_lengths.clamp(max=T)
            loss = loss_fn(logits, phon_targets, clamped_il, target_lengths)

        evaluator.update(logits, phon_targets, target_lengths)
        total_loss += loss.item()

    results = evaluator.compute()
    results["val_loss"] = round(total_loss / len(loader), 4)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["wav2vec2", "whisper"], required=True)
    parser.add_argument("--data_dir", default="/content/drive/MyDrive/l2arctic")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = get_config(args.model)
    if args.epochs:
        config["num_epochs"] = args.epochs

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(config["save_dir"], exist_ok=True)

    cache = f"{config['save_dir']}/metadata_cache.json"
    dataset = L2ArcticDataset(args.data_dir, cache_path=cache)
    train_set, val_set, test_set = get_train_val_test_split(dataset)

    collator = (Wav2Vec2Collator(config["model_name"])
                if args.model == "wav2vec2"
                else WhisperCollator(config["model_name"]))

    train_loader = DataLoader(train_set, batch_size=config["batch_size"],
                              shuffle=True, collate_fn=collator,
                              num_workers=config["num_workers"])
    val_loader   = DataLoader(val_set,   batch_size=config["batch_size"]*2,
                              shuffle=False, collate_fn=collator,
                              num_workers=config["num_workers"])
    test_loader  = DataLoader(test_set,  batch_size=config["batch_size"]*2,
                              shuffle=False, collate_fn=collator,
                              num_workers=config["num_workers"])

    model = (Wav2Vec2ForPhonology(config["model_name"])
             if args.model == "wav2vec2"
             else WhisperForPhonology(config["model_name"]))
    model = model.to(device)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"], weight_decay=config["weight_decay"])
    total_steps = (len(train_loader) // config["gradient_accumulation"]
                   * config["num_epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    scaler    = GradScaler("cuda")
    loss_fn   = SCTCSBLoss().to(device)
    evaluator = PhonologicalEvaluator()

    start_epoch   = 1
    best_macro_f1 = 0.0
    save_path = f"{config['save_dir']}/{args.model}_best.pt"

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch   = ckpt.get("epoch", 1) + 1
        best_macro_f1 = ckpt.get("best_macro_f1", 0.0)

    history = []
    print(f"\nStarting training: {config['num_epochs']} epochs\n")

    for epoch in range(start_epoch, config["num_epochs"] + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 scaler, loss_fn, config, device,
                                 args.model, epoch)

        if epoch % config["eval_every"] == 0:
            val_res = evaluate(model, val_loader, loss_fn,
                               evaluator, device, args.model)
            elapsed = time.time() - t0
            print(f"\nEpoch {epoch}/{config['num_epochs']} ({elapsed:.0f}s) | "
                  f"Train: {train_loss:.4f} | Val: {val_res['val_loss']:.4f} | "
                  f"Macro F1: {val_res['macro_f1']:.4f} | "
                  f"Macro ACC: {val_res['macro_acc']:.4f}")

            history.append({"epoch": epoch,
                            "train_loss": round(train_loss, 4), **val_res})

            if val_res["macro_f1"] > best_macro_f1:
                best_macro_f1 = val_res["macro_f1"]
                torch.save({"epoch": epoch, "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "best_macro_f1": best_macro_f1,
                            "config": config, "val_results": val_res},
                           save_path)
                print(f"  ✓ Saved best (macro F1: {best_macro_f1:.4f})")

    print("\nLoading best model for test evaluation...")
    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_res = evaluate(model, test_loader, loss_fn,
                        evaluator, device, args.model)
    print_results(test_res, f"{args.model.upper()} (Test)")

    with open(f"{config['save_dir']}/{args.model}_results.json", "w") as f:
        json.dump({"config": config, "history": history,
                   "test_results": test_res}, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
