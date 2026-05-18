"""
Train a vocal emotion recognition model on RAVDESS.

Usage:
    python train.py --config config/default.yaml --model.type cnn
    python train.py --config config/default.yaml --model.type lstm
    python train.py --config config/default.yaml --model.type gru
    python train.py --config config/default.yaml --model.type mlp
"""
import os, sys
_site = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site-packages")
if os.path.isdir(_site) and _site not in sys.path:
    sys.path.insert(0, _site)

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from addict import Dict
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data_classes.ravdess_dataset import RAVDESSDataset
from utils import (
    build_model,
    compute_metrics,
    dataset_mode,
    plot_confusion_matrix,
    plot_training_curves,
    print_report,
    set_seed,
)


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train RAVDESS emotion classifier")
    parser.add_argument("--config", default="config/default.yaml")
    # allow overrides like --model.type lstm
    parser.add_argument("--model.type", dest="model_type", default=None)
    parser.add_argument("--training.epochs", dest="epochs", type=int, default=None)
    parser.add_argument("--training.batch_size", dest="batch_size", type=int, default=None)
    parser.add_argument("--training.learning_rate", dest="lr", type=float, default=None)
    return parser.parse_args()


def load_cfg(args) -> Dict:
    with open(args.config) as f:
        cfg = Dict(yaml.safe_load(f))
    if args.model_type: cfg.model.type = args.model_type
    if args.epochs:     cfg.training.epochs = args.epochs
    if args.batch_size: cfg.training.batch_size = args.batch_size
    if args.lr:         cfg.training.learning_rate = args.lr
    return cfg


# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    preds, labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += len(y)
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(y.cpu().numpy())
    return total_loss / total, correct / total, np.array(preds), np.array(labels)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_cfg(args)
    set_seed(cfg.training.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.training.device == "cuda" else "cpu"
    )
    print(f"Device : {device}")
    print(f"Model  : {cfg.model.type.upper()}")

    mode        = dataset_mode(cfg.model.type)
    test_actors = list(cfg.data.test_actors)
    all_actors  = list(range(1, 25))
    train_val_actors = [a for a in all_actors if a not in test_actors]

    full_ds = RAVDESSDataset(
        cfg.data.data_dir, actor_ids=train_val_actors, mode=mode,
        sample_rate=cfg.data.sample_rate, duration=cfg.data.duration,
        n_mfcc=cfg.data.n_mfcc, n_mels=cfg.data.n_mels,
        n_fft=cfg.data.n_fft,   hop_length=cfg.data.hop_length,
    )
    test_ds = RAVDESSDataset(
        cfg.data.data_dir, actor_ids=test_actors, mode=mode,
        sample_rate=cfg.data.sample_rate, duration=cfg.data.duration,
        n_mfcc=cfg.data.n_mfcc, n_mels=cfg.data.n_mels,
        n_fft=cfg.data.n_fft,   hop_length=cfg.data.hop_length,
    )

    val_n   = int(0.15 * len(full_ds))
    train_n = len(full_ds) - val_n
    train_ds, val_ds = random_split(
        full_ds, [train_n, val_n],
        generator=torch.Generator().manual_seed(cfg.training.seed),
    )
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    num_workers = 0 if os.name == "nt" else 2   # Windows: 0 workers
    train_loader = DataLoader(train_ds, cfg.training.batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   cfg.training.batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  cfg.training.batch_size, shuffle=False, num_workers=num_workers)

    model = build_model(cfg).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    weights   = full_ds.class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, factor=0.5, verbose=True)

    os.makedirs(cfg.output.model_dir,   exist_ok=True)
    os.makedirs(cfg.output.results_dir, exist_ok=True)

    name = cfg.model.type.lower()
    ckpt = os.path.join(cfg.output.model_dir, f"best_{name}.pth")

    best_val_acc  = 0.0
    patience_cnt  = 0
    tr_losses, vl_losses = [], []
    tr_accs,   vl_accs   = [], []

    for epoch in range(1, cfg.training.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(vl_loss)

        tr_losses.append(tr_loss); vl_losses.append(vl_loss)
        tr_accs.append(tr_acc);    vl_accs.append(vl_acc)

        flag = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            patience_cnt = 0
            torch.save(model.state_dict(), ckpt)
            flag = " ✓ saved"
        else:
            patience_cnt += 1

        print(
            f"Epoch {epoch:03d}/{cfg.training.epochs}"
            f"  train loss={tr_loss:.4f} acc={tr_acc:.4f}"
            f"  val   loss={vl_loss:.4f} acc={vl_acc:.4f}{flag}"
        )
        if patience_cnt >= cfg.training.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    # ── Save training curves ────────────────────────────────────────────────
    plot_training_curves(
        tr_losses, vl_losses, tr_accs, vl_accs,
        save_path=os.path.join(cfg.output.results_dir, f"curves_{name}.png"),
    )

    # ── Test evaluation ─────────────────────────────────────────────────────
    model.load_state_dict(torch.load(ckpt, map_location=device))
    _, test_acc, test_preds, test_labels = eval_epoch(model, test_loader, criterion, device)

    metrics = compute_metrics(test_labels, test_preds)
    print(f"\n{'='*50}")
    print(f"TEST RESULTS  [{name.upper()}]")
    print(f"  Accuracy       : {metrics['accuracy']:.4f}")
    print(f"  F1 macro       : {metrics['f1_macro']:.4f}")
    print(f"  F1 weighted    : {metrics['f1_weighted']:.4f}")
    print_report(test_labels, test_preds)

    plot_confusion_matrix(
        test_labels, test_preds,
        title=f"Confusion Matrix — {name.upper()}",
        save_path=os.path.join(cfg.output.results_dir, f"cm_{name}.png"),
    )

    with open(os.path.join(cfg.output.results_dir, f"metrics_{name}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Results saved → {cfg.output.results_dir}/")


if __name__ == "__main__":
    main()
