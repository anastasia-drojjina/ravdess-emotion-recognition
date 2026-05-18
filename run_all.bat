@echo off
REM run_all.bat — train and evaluate all model variants (Windows)
echo ============================================================
echo  RAVDESS Emotion Recognition — Training all models
echo ============================================================

echo.
echo [1/4] MLP baseline
python train.py --config config/default.yaml "--model.type" mlp

echo.
echo [2/4] CNN (mel-spectrogram)
python train.py --config config/default.yaml "--model.type" cnn

echo.
echo [3/4] LSTM (bidirectional)
python train.py --config config/default.yaml "--model.type" lstm

echo.
echo [4/4] GRU (bidirectional)
python train.py --config config/default.yaml "--model.type" gru

echo.
echo ============================================================
echo  All models trained. Results in results/
echo ============================================================
