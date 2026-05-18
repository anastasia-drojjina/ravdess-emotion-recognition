@echo off
REM prepare.bat — install dependencies (Windows)
echo Installing requirements...
pip install -r requirements.txt
echo.
echo Setup complete. Run:
echo   python train.py --config config/default.yaml "--model.type" cnn
echo   python test.py  --config config/default.yaml "--model.type" cnn
