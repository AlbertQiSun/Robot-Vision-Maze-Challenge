#!/usr/bin/env bash
# ===========================================================================
#  Visual Navigation Player — One-Command Setup
#  Works on: macOS (Intel/Apple Silicon), Linux (with/without CUDA), WSL2
#
#  Usage:
#    chmod +x setup.sh
#    ./setup.sh              # auto-detects CUDA
#    ./setup.sh --cpu        # force CPU-only PyTorch
#    ./setup.sh --cuda 12.1  # force specific CUDA version
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="game"
PYTHON_VER="3.10"
FORCE_CPU=false
CUDA_VER=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu) FORCE_CPU=true; shift ;;
        --cuda) CUDA_VER="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo " Visual Navigation Player Setup"
echo "========================================"

# ---- 1. Check conda ----
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Install Miniconda or Anaconda first."
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# ---- 2. Detect OS & GPU ----
OS="$(uname -s)"
ARCH="$(uname -m)"
echo "[INFO] OS=$OS  ARCH=$ARCH"

HAS_CUDA=false
if command -v nvidia-smi &>/dev/null; then
    HAS_CUDA=true
    DETECTED_CUDA="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -1 || true)"
    echo "[INFO] NVIDIA GPU detected (driver: ${DETECTED_CUDA:-unknown})"
else
    echo "[INFO] No NVIDIA GPU detected"
fi

if [[ "$OS" == "Darwin" ]]; then
    echo "[INFO] macOS detected — will use MPS (Metal) for PyTorch if Apple Silicon"
    FORCE_CPU=false  # PyTorch handles MPS automatically
fi

# ---- 3. Create / update conda environment ----
echo ""
echo "[STEP 1/5] Creating conda environment '${ENV_NAME}' (Python ${PYTHON_VER})..."

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  Environment '${ENV_NAME}' exists. Updating..."
    conda env update -n "$ENV_NAME" -f environment.yaml --prune 2>&1 | tail -3
else
    conda env create -f environment.yaml 2>&1 | tail -5
fi

# ---- 4. Install PyTorch with correct backend ----
echo ""
echo "[STEP 2/5] Installing PyTorch..."

# Activate environment for pip installs
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

if [[ "$FORCE_CPU" == true ]]; then
    echo "  Installing PyTorch (CPU-only)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
elif [[ "$HAS_CUDA" == true ]]; then
    if [[ -n "$CUDA_VER" ]]; then
        CU_TAG="cu${CUDA_VER//./}"
    else
        # Auto-detect CUDA version
        NVCC_VER="$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || true)"
        if [[ -n "$NVCC_VER" ]]; then
            CU_TAG="cu${NVCC_VER//./}"
        else
            CU_TAG="cu121"  # safe default
        fi
    fi
    echo "  Installing PyTorch (CUDA ${CU_TAG})..."
    pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CU_TAG}" -q
else
    echo "  Installing PyTorch (default — CPU or MPS)..."
    pip install torch torchvision -q
fi

# ---- 5. Install remaining pip dependencies ----
echo ""
echo "[STEP 3/5] Installing additional dependencies..."
pip install faiss-cpu>=1.7.4 kornia>=0.7.0 kornia-rs>=0.1.0 tqdm>=4.65.0 natsort networkx -q

# Install vis-nav-game from test PyPI
pip install --extra-index-url https://test.pypi.org/simple/ vis-nav-game -q 2>/dev/null || \
    echo "  [WARN] vis-nav-game install may have issues — will work if already in conda env"

# ---- 6. Set up Git LFS ----
echo ""
echo "[STEP 4/5] Setting up Git LFS for model weights..."
if command -v git-lfs &>/dev/null || command -v git lfs &>/dev/null; then
    git lfs install 2>/dev/null || true
    echo "  Git LFS initialized."
else
    echo "  [WARN] Git LFS not found. Install it: https://git-lfs.com"
    echo "         Model weights in models/ won't be pulled without LFS."
fi

# ---- 7. Clone game engine (optional, for multi-maze training) ----
echo ""
echo "[STEP 5/5] Checking for game engine (vis_nav_game_public)..."
if [[ ! -d "vis_nav_game_public" ]]; then
    echo "  Cloning vis_nav_game_public for multi-maze generation..."
    git clone https://github.com/ai4ce/vis_nav_game_public.git 2>&1 | tail -2
else
    echo "  vis_nav_game_public already exists."
fi

# ---- 8. Create required directories ----
mkdir -p models cache training_data

# ---- Done ----
echo ""
echo "========================================"
echo " Setup complete!"
echo "========================================"
echo ""
echo " To activate:  conda activate ${ENV_NAME}"
echo ""
echo " Quick check:"
echo "   python -c \"import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')\""
echo ""
echo " Run baseline:  python source/baseline.py"
echo " Run autonomous player:  python source/player.py"
echo ""
