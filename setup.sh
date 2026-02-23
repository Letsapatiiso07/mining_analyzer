#!/usr/bin/env bash
# setup.sh – One-command environment setup for SA Mining Analyzer
# Usage:  bash setup.sh
# Creates a virtualenv, installs all deps, and runs the full pipeline.

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"

echo "============================================================"
echo "  SA Open-Pit Mining Site Analyzer – Setup"
echo "============================================================"

# Check Python version
PYTHON=$(which python3.10 2>/dev/null || which python3 2>/dev/null || which python 2>/dev/null)
if [ -z "$PYTHON" ]; then
    echo "Python 3.10+ not found. Please install it first."
    exit 1
fi
PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python $PY_VER found at $PYTHON"

# Create virtualenv if not present
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"
echo "Virtual environment activated"

# Upgrade pip
pip install --upgrade pip wheel setuptools --quiet

# Install requirements
echo "Installing Python dependencies (this may take a few minutes)..."
pip install -r "$PROJECT_DIR/requirements.txt"

echo ""
echo "============================================================"
echo "  Running data pipeline..."
echo "============================================================"
cd "$PROJECT_DIR"

python scripts/01_download_and_process.py
python scripts/02_features.py
python scripts/03_train.py
python scripts/04_predict.py

echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  Launch the dashboard with:"
echo "    source venv/bin/activate"
echo "    streamlit run app.py"
echo "============================================================"