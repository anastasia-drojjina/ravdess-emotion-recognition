# RAVDESS Vocal Emotion Recognition

**Dataset:** RAVDESS — Ryerson Audio-Visual Database of Emotional Speech and Song  
**Task:** Multi-class classification of vocal emotion from audio recordings  
**Emotions:** neutral · calm · happy · sad · angry · fearful (6 classes)  
**Modality:** Song subset (Audio_Song_Actors_01-24, 24 actors, ~1056 clips)

---

## Project Structure

```
ravdess-emotion-recognition/
├── config/
│   └── default.yaml          # all hyperparameters
├── data_classes/
│   └── ravdess_dataset.py    # PyTorch Dataset (MFCC / mel-spectrogram modes)
├── model_classes/
│   ├── cnn_model.py          # 4-block CNN on mel-spectrograms
│   ├── rnn_model.py          # LSTM / GRU / RNN on MFCC sequences
│   └── baseline_model.py     # MLP with mean-pooled MFCC
├── extract_representations/
│   └── extract_features.py   # pre-compute MFCC stat features (for sklearn)
├── notebooks/
│   ├── 01_EDA.ipynb          # Exploratory Data Analysis
│   ├── 02_baseline.ipynb     # SVM / Random Forest baseline
│   ├── 03_CNN.ipynb          # CNN training & analysis
│   ├── 04_LSTM.ipynb         # LSTM / GRU training & analysis
│   └── 05_evaluation.ipynb   # Model comparison + SHAP interpretability
├── utils.py                  # metrics, plotting, model factory
├── train.py                  # training script
├── test.py                   # evaluation script
├── prepare.sh / prepare.bat  # install dependencies
└── run_all.sh / run_all.bat  # train all models sequentially
```

---

## Setup & Reproducibility

### 1. Install dependencies

```bash
# Linux / macOS
bash prepare.sh

# Windows
prepare.bat
```

### 2. Dataset

Download **Audio_Song_Actors_01-24** from Zenodo:  
https://doi.org/10.5281/zenodo.1188976

Place the extracted folder next to the project folder (default path in `config/default.yaml`):

```
Machine Learning/
├── Audio_Song_Actors_01-24/   ← dataset here
│   ├── Actor_01/
│   │   └── 03-02-01-01-01-01-01.wav
│   └── ...
└── ravdess-emotion-recognition/
    └── ...
```

Alternatively, edit `data.data_dir` in `config/default.yaml`.

### 3. Extract features for baseline notebooks

```bash
python extract_representations/extract_features.py
```

### 4. Train a single model

```bash
python train.py --config config/default.yaml "--model.type" cnn
python train.py --config config/default.yaml "--model.type" lstm
python train.py --config config/default.yaml "--model.type" gru
python train.py --config config/default.yaml "--model.type" mlp
```

### 5. Train all models

```bash
# Linux / macOS
bash run_all.sh

# Windows
run_all.bat
```

### 6. Evaluate saved model

```bash
python test.py --config config/default.yaml "--model.type" cnn
```

---

## Models

| Model | Input | Architecture |
|-------|-------|-------------|
| **MLP** | Mean+Std of MFCC (80-dim) | 2 hidden layers, BatchNorm, Dropout |
| **CNN** | Mel-spectrogram (1×128×T) | 4 conv blocks + Global Avg Pooling |
| **LSTM** | MFCC sequence (T×40) | 2-layer Bidirectional LSTM |
| **GRU** | MFCC sequence (T×40) | 2-layer Bidirectional GRU |

---

## Experimental Design

- **Speaker-independent split:** Actors 21–24 reserved for test; actors 1–20 for train/val (85/15).
- **Class-weighted loss:** compensates for neutral class imbalance (fewer samples).
- **Early stopping:** patience = 12 epochs.
- **Evaluation metrics:** Accuracy, F1 macro, F1 weighted, per-class report.

---

## Results

Results and figures are saved in `results/` after training:

| File | Content |
|------|---------|
| `metrics_<model>.json` | Accuracy, F1 macro/weighted |
| `cm_<model>.png` | Normalized confusion matrix |
| `curves_<model>.png` | Train/val loss and accuracy |

See `notebooks/05_evaluation.ipynb` for full comparison and SHAP analysis.

---

## Key Findings (to be filled after experiments)

> Run the notebooks and fill this section before the presentation.

---

## Requirements

- Python 3.10+
- PyTorch ≥ 2.0
- librosa ≥ 0.10
- scikit-learn ≥ 1.3
- shap ≥ 0.42

See `requirements.txt` for the full list.
