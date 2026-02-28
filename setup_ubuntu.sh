#!/bin/bash
set -e

RUST_TOOLCHAIN="${RUST_TOOLCHAIN:-nightly-2025-09-02}"

echo "=== ClozeMaster Ubuntu Setup ==="
echo "Rust toolchain: $RUST_TOOLCHAIN"
echo ""

# 1. System packages
echo "[1/6] Installing system packages..."
sudo apt-get update
sudo apt-get install -y git curl wget build-essential

# 2. Rust compiler (rustup)
echo "[2/6] Installing Rust..."
if ! command -v rustup &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "rustup already installed"
fi
rustup install "$RUST_TOOLCHAIN"
rustup default "$RUST_TOOLCHAIN"
echo "Using: $(rustc --version)"

# 3. Miniconda + Python 3.8
echo "[3/6] Installing Miniconda..."
if ! command -v conda &> /dev/null; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    rm /tmp/miniconda.sh
else
    eval "$(conda shell.bash hook)"
    echo "Conda already installed"
fi

echo "[4/6] Creating conda environment (py38)..."
# Accept Conda ToS for default channels (required since late 2025)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

if ! conda env list | grep -q "^py38 "; then
    conda create -n py38 python=3.8 -y
fi
conda activate py38

# 4. PyTorch + Python dependencies
echo "[5/6] Installing Python packages..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install tokenizers pandas
pip install -r requirements.txt

# 5. Download Incoder-1B model
echo "[6/6] Downloading Incoder-1B model..."
if [ ! -d "model/Incoder1b" ]; then
    pip install huggingface-hub
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='facebook/incoder-1B', local_dir='model/Incoder1b')
"
else
    echo "Model already exists at model/Incoder1b"
fi

# 6. Create required directories
mkdir -p temp log dataset/history_codes target_dataset

echo ""
echo "=== Setup complete ==="
echo "Activate env:  conda activate py38"
echo "Run:           python main.py"
echo ""
echo "NOTE: Place your .rs files in dataset/history_codes/ (nested dirs OK)."
echo "To use a different Rust toolchain: RUST_TOOLCHAIN=nightly-2025-01-01 bash setup_ubuntu.sh"
