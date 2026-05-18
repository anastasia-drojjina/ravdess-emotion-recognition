"""
Pre-compute and save MFCC features for all clips to speed up
baseline (sklearn) experiments in the notebooks.

Output: results/mfcc_features.npz
  X_train, y_train, X_test, y_test
  (each sample = mean + std of MFCC over time → 80-dim vector)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
_site = os.path.join(os.path.dirname(__file__), "..", "site-packages")
if os.path.isdir(_site) and _site not in sys.path:
    sys.path.insert(0, _site)

import numpy as np
import yaml
from addict import Dict
from tqdm import tqdm

from data_classes.ravdess_dataset import RAVDESSDataset


def mfcc_stat_features(dataset: RAVDESSDataset) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in tqdm(range(len(dataset)), desc="Extracting"):
        mfcc, label = dataset[i]
        mfcc = mfcc.numpy()           # (T, n_mfcc)
        feat = np.concatenate([mfcc.mean(axis=0), mfcc.std(axis=0)])
        X.append(feat)
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


if __name__ == "__main__":
    with open("config/default.yaml") as f:
        cfg = Dict(yaml.safe_load(f))

    test_actors      = list(cfg.data.test_actors)
    all_actors       = list(range(1, 25))
    train_val_actors = [a for a in all_actors if a not in test_actors]

    common = dict(
        mode="mfcc",
        sample_rate=cfg.data.sample_rate,
        duration=cfg.data.duration,
        n_mfcc=cfg.data.n_mfcc,
        n_fft=cfg.data.n_fft,
        hop_length=cfg.data.hop_length,
    )
    train_ds = RAVDESSDataset(cfg.data.data_dir, actor_ids=train_val_actors, **common)
    test_ds  = RAVDESSDataset(cfg.data.data_dir, actor_ids=test_actors,      **common)

    print(f"Train samples: {len(train_ds)}  Test samples: {len(test_ds)}")

    X_train, y_train = mfcc_stat_features(train_ds)
    X_test,  y_test  = mfcc_stat_features(test_ds)

    os.makedirs("results", exist_ok=True)
    np.savez("results/mfcc_features.npz",
             X_train=X_train, y_train=y_train,
             X_test=X_test,   y_test=y_test)
    print("Saved → results/mfcc_features.npz")
    print(f"Feature dim: {X_train.shape[1]}")
