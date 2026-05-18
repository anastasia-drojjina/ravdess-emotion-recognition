"""Shared utilities: seeding, metrics, plotting."""
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

EMOTION_NAMES = ["neutral", "calm", "happy", "sad", "angry", "fearful"]


# ── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def print_report(y_true, y_pred):
    print(classification_report(y_true, y_pred, target_names=EMOTION_NAMES, zero_division=0))


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, title: str = "", save_path: str = None):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=EMOTION_NAMES, yticklabels=EMOTION_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path: str = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.plot(train_losses, label="Train")
    ax1.plot(val_losses,   label="Val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(train_accs, label="Train")
    ax2.plot(val_accs,   label="Val")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


# ── Model factory (shared between train.py / test.py) ────────────────────────

def build_model(cfg):
    from model_classes.cnn_model import CNNEmotionClassifier
    from model_classes.rnn_model import SequenceEmotionClassifier
    from model_classes.baseline_model import MLPEmotionClassifier

    t = cfg.model.type.lower()
    n = cfg.data.n_classes

    if t == "cnn":
        return CNNEmotionClassifier(n_classes=n, n_mels=cfg.data.n_mels, dropout=cfg.model.cnn.dropout)
    if t in ("lstm", "gru", "rnn"):
        return SequenceEmotionClassifier(
            model_type=t,
            input_size=cfg.data.n_mfcc,
            hidden_size=cfg.model.rnn.hidden_size,
            num_layers=cfg.model.rnn.num_layers,
            n_classes=n,
            dropout=cfg.model.rnn.dropout,
            bidirectional=cfg.model.rnn.bidirectional,
        )
    if t == "mlp":
        return MLPEmotionClassifier(
            input_size=cfg.data.n_mfcc,
            hidden_sizes=list(cfg.model.mlp.hidden_sizes),
            n_classes=n,
            dropout=cfg.model.mlp.dropout,
        )
    raise ValueError(f"Unknown model type: {t}")


def dataset_mode(model_type: str) -> str:
    return "melspec" if model_type.lower() == "cnn" else "mfcc"
