"""
Evaluate a trained model on the held-out test set.

Usage:
    python test.py --config config/default.yaml --model.type cnn
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
from torch.utils.data import DataLoader

from data_classes.ravdess_dataset import RAVDESSDataset
from utils import (
    build_model,
    compute_metrics,
    dataset_mode,
    plot_confusion_matrix,
    print_report,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--model.type", dest="model_type", default=None)
    return p.parse_args()


def load_cfg(args) -> Dict:
    with open(args.config) as f:
        cfg = Dict(yaml.safe_load(f))
    if args.model_type:
        cfg.model.type = args.model_type
    return cfg


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(y.numpy())
    return np.array(preds), np.array(labels)


def main():
    args = parse_args()
    cfg  = load_cfg(args)
    set_seed(cfg.training.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.training.device == "cuda" else "cpu"
    )
    name = cfg.model.type.lower()
    ckpt = os.path.join(cfg.output.model_dir, f"best_{name}.pth")

    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}  (run train.py first)")

    mode = dataset_mode(name)
    test_ds = RAVDESSDataset(
        cfg.data.data_dir,
        actor_ids=list(cfg.data.test_actors),
        mode=mode,
        sample_rate=cfg.data.sample_rate,
        duration=cfg.data.duration,
        n_mfcc=cfg.data.n_mfcc,
        n_mels=cfg.data.n_mels,
        n_fft=cfg.data.n_fft,
        hop_length=cfg.data.hop_length,
    )
    num_workers = 0 if os.name == "nt" else 2
    loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, num_workers=num_workers)

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    preds, labels = evaluate(model, loader, device)
    metrics = compute_metrics(labels, preds)

    os.makedirs(cfg.output.results_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"TEST RESULTS  [{name.upper()}]")
    print(f"  Accuracy       : {metrics['accuracy']:.4f}")
    print(f"  F1 macro       : {metrics['f1_macro']:.4f}")
    print(f"  F1 weighted    : {metrics['f1_weighted']:.4f}")
    print_report(labels, preds)

    plot_confusion_matrix(
        labels, preds,
        title=f"Confusion Matrix — {name.upper()}",
        save_path=os.path.join(cfg.output.results_dir, f"cm_{name}.png"),
    )
    with open(os.path.join(cfg.output.results_dir, f"metrics_{name}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved → {cfg.output.results_dir}/")


if __name__ == "__main__":
    main()
