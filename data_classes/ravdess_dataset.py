"""
RAVDESS Dataset loader — supports both speech (vocal_channel=01) and
song (vocal_channel=02) subsets.

Filename format: [Modality]-[Vocal_channel]-[Emotion]-[Intensity]-[Statement]-[Repetition]-[Actor].wav
Emotions used: 01=neutral 02=calm 03=happy 04=sad 05=angry 06=fearful
"""
import os
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

EMOTION_MAP = {
    "01": 0,  # neutral
    "02": 1,  # calm
    "03": 2,  # happy
    "04": 3,  # sad
    "05": 4,  # angry
    "06": 5,  # fearful
}

EMOTION_NAMES = ["neutral", "calm", "happy", "sad", "angry", "fearful"]


class RAVDESSDataset(Dataset):
    """
    mode='mfcc'    → returns (time_steps, n_mfcc)  tensors  [for RNN / MLP]
    mode='melspec' → returns (1, n_mels, time_frames) tensors [for CNN]
    """

    def __init__(
        self,
        data_dir: str,
        actor_ids=None,
        mode: str = "mfcc",
        sample_rate: int = 22050,
        duration: float = 4.0,
        n_mfcc: int = 40,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
    ):
        self.mode = mode
        self.sr = sample_rate
        self.duration = duration
        self.max_samples = int(sample_rate * duration)
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.files: list[str] = []
        self.labels: list[int] = []
        self._collect(data_dir, actor_ids)

    # ------------------------------------------------------------------
    def _collect(self, data_dir, actor_ids):
        for wav in Path(data_dir).rglob("*.wav"):
            parts = wav.stem.split("-")
            if len(parts) != 7:
                continue
            _, _, emotion, _, _, _, actor = parts
            if emotion not in EMOTION_MAP:
                continue
            if actor_ids is not None and int(actor) not in actor_ids:
                continue
            self.files.append(str(wav))
            self.labels.append(EMOTION_MAP[emotion])

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.files)

    def _load(self, path: str) -> np.ndarray:
        y, _ = librosa.load(path, sr=self.sr, duration=self.duration)
        if len(y) < self.max_samples:
            y = np.pad(y, (0, self.max_samples - len(y)))
        else:
            y = y[: self.max_samples]
        return y

    def __getitem__(self, idx):
        y = self._load(self.files[idx])
        label = self.labels[idx]

        if self.mode == "mfcc":
            feat = librosa.feature.mfcc(
                y=y, sr=self.sr, n_mfcc=self.n_mfcc,
                n_fft=self.n_fft, hop_length=self.hop_length,
            ).T.astype(np.float32)                        # (T, n_mfcc)
            feat = (feat - feat.mean()) / (feat.std() + 1e-8)
            return torch.from_numpy(feat), label

        elif self.mode == "melspec":
            mel = librosa.feature.melspectrogram(
                y=y, sr=self.sr, n_mels=self.n_mels,
                n_fft=self.n_fft, hop_length=self.hop_length,
            )
            mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
            # Normalize to [0, 1]
            lo, hi = mel_db.min(), mel_db.max()
            mel_db = (mel_db - lo) / (hi - lo + 1e-8)
            return torch.from_numpy(mel_db[np.newaxis]), label   # (1, n_mels, T)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    # ------------------------------------------------------------------
    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for CrossEntropyLoss."""
        counts = np.bincount(self.labels, minlength=len(EMOTION_MAP))
        weights = 1.0 / (counts + 1e-8)
        return torch.tensor(weights / weights.sum(), dtype=torch.float32)
