#!/usr/bin/env bash
# ─────────────────────────────────────────────────────
# prepare.sh — install dependencies
# Usage: bash prepare.sh
# ─────────────────────────────────────────────────────
set -e
echo "Installing requirements..."
pip install -r requirements.txt
echo ""
echo "Setup complete. You can now run:"
echo "  python train.py --config config/default.yaml --model.type cnn"
echo "  python test.py  --config config/default.yaml --model.type cnn"
