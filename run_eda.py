"""
Generates all EDA figures (equivalent to notebook 01_EDA.ipynb).
Run: python run_eda.py
Output: results/eda_*.png
"""
import os, sys, io
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
_c_pkgs = r"C:\Users\Anastasia\ravdess-pkgs"
_site = _c_pkgs if os.path.isdir(_c_pkgs) else os.path.join(os.path.dirname(os.path.abspath(__file__)), "site-packages")
if os.path.isdir(_site) and _site not in sys.path:
    sys.path.insert(0, _site)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import librosa
import librosa.display
from pathlib import Path
import yaml
from addict import Dict

plt.rcParams["figure.dpi"] = 120
sns.set_style("whitegrid")
os.makedirs("results", exist_ok=True)

with open("config/default.yaml") as f:
    cfg = Dict(yaml.safe_load(f))

DATA_DIR = Path(cfg.data.data_dir)
EMOTION_MAP = {"01":"neutral","02":"calm","03":"happy","04":"sad","05":"angry","06":"fearful"}
EMOTION_NAMES = list(EMOTION_MAP.values())
SR = cfg.data.sample_rate

print("Loading file list...")
records = []
for wav in DATA_DIR.rglob("*.wav"):
    parts = wav.stem.split("-")
    if len(parts) != 7:
        continue
    _, _, emotion, intensity, _, _, actor = parts
    if emotion not in EMOTION_MAP:
        continue
    records.append({
        "file": str(wav),
        "emotion": EMOTION_MAP[emotion],
        "intensity": "normal" if intensity == "01" else "strong",
        "actor": int(actor),
        "gender": "female" if int(actor) % 2 == 0 else "male",
    })

df = pd.DataFrame(records)
print(f"Total clips: {len(df)}")

# ── 1. Class distribution ────────────────────────────────────────────────────
print("Generating class distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
counts = df["emotion"].value_counts().reindex(EMOTION_NAMES)
axes[0].bar(EMOTION_NAMES, counts, color=sns.color_palette("Set2", 6))
axes[0].set_title("Clips per Emotion")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=30)
ct = df.groupby(["emotion", "gender"]).size().unstack()
ct.reindex(EMOTION_NAMES).plot(kind="bar", ax=axes[1], color=["#FF9999","#66B2FF"])
axes[1].set_title("Clips per Emotion x Gender")
axes[1].tick_params(axis="x", rotation=30)
axes[1].legend(title="Gender")
plt.tight_layout()
plt.savefig("results/eda_class_distribution.png", dpi=150)
plt.close()
print("  Saved: results/eda_class_distribution.png")

# ── 2. Waveforms ─────────────────────────────────────────────────────────────
print("Generating waveforms...")
fig, axes = plt.subplots(2, 3, figsize=(15, 6))
axes = axes.flatten()
for i, (emo_id, emo_name) in enumerate(EMOTION_MAP.items()):
    sample = df[df.emotion == emo_name].iloc[0]
    y, _ = librosa.load(sample.file, sr=SR, duration=4.0)
    t = np.linspace(0, len(y)/SR, len(y))
    axes[i].plot(t, y, alpha=0.8, linewidth=0.5)
    axes[i].set_title(f"{emo_name.capitalize()}")
    axes[i].set_xlabel("Time (s)")
plt.suptitle("Waveforms per Emotion", fontsize=14)
plt.tight_layout()
plt.savefig("results/eda_waveforms.png", dpi=150)
plt.close()
print("  Saved: results/eda_waveforms.png")

# ── 3. Mel-spectrograms ───────────────────────────────────────────────────────
print("Generating mel-spectrograms...")
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
axes = axes.flatten()
for i, (emo_id, emo_name) in enumerate(EMOTION_MAP.items()):
    sample = df[df.emotion == emo_name].iloc[0]
    y, _ = librosa.load(sample.file, sr=SR, duration=4.0)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=128, n_fft=2048, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(mel_db, sr=SR, hop_length=512,
                                    x_axis="time", y_axis="mel", ax=axes[i])
    axes[i].set_title(f"{emo_name.capitalize()}")
    fig.colorbar(img, ax=axes[i], format="%+2.0f dB")
plt.suptitle("Mel-Spectrograms per Emotion", fontsize=14)
plt.tight_layout()
plt.savefig("results/eda_mel_spectrograms.png", dpi=150)
plt.close()
print("  Saved: results/eda_mel_spectrograms.png")

# ── 4. MFCC ───────────────────────────────────────────────────────────────────
print("Generating MFCC plots...")
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
axes = axes.flatten()
for i, (emo_id, emo_name) in enumerate(EMOTION_MAP.items()):
    sample = df[df.emotion == emo_name].iloc[0]
    y, _ = librosa.load(sample.file, sr=SR, duration=4.0)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=40, n_fft=2048, hop_length=512)
    img = librosa.display.specshow(mfcc, sr=SR, hop_length=512, x_axis="time", ax=axes[i])
    axes[i].set_title(f"MFCC - {emo_name.capitalize()}")
    fig.colorbar(img, ax=axes[i])
plt.suptitle("MFCCs per Emotion", fontsize=14)
plt.tight_layout()
plt.savefig("results/eda_mfcc.png", dpi=150)
plt.close()
print("  Saved: results/eda_mfcc.png")

# ── 5. Average MFCC per emotion ───────────────────────────────────────────────
print("Computing average MFCCs (this takes ~2 min)...")
from tqdm import tqdm
emo_mfccs = {e: [] for e in EMOTION_NAMES}
for _, row in tqdm(df.iterrows(), total=len(df)):
    y, _ = librosa.load(row.file, sr=SR, duration=4.0)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=40, n_fft=2048, hop_length=512)
    emo_mfccs[row.emotion].append(mfcc.mean(axis=1))

fig, ax = plt.subplots(figsize=(12, 5))
for emo, color in zip(EMOTION_NAMES, sns.color_palette("Set2", 6)):
    means = np.stack(emo_mfccs[emo]).mean(axis=0)
    ax.plot(range(40), means, label=emo, color=color, linewidth=2)
ax.set_xlabel("MFCC Coefficient")
ax.set_ylabel("Mean Value")
ax.set_title("Average MFCC Coefficients per Emotion")
ax.legend(ncol=3)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/eda_mfcc_means.png", dpi=150)
plt.close()
print("  Saved: results/eda_mfcc_means.png")

print("\nAll EDA figures saved to results/")
