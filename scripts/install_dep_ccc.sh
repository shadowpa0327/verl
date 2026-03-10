#!/bin/bash
# Install verl dependencies without installing the verl package itself
# Equivalent to the Dockerfile but as a standalone bash script
# Uses uv for venv creation and package installation

set -e  # Exit on error

# ----------
# Configuration
# ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(dirname "$SCRIPT_DIR")"

PYTHON_VERSION="3.11"
PYTHON_VERSION_SHORT="311"
TORCH_VERSION="2.8"
FLASHINFER_VERSION="0.3.1"
VLLM_VERSION="0.11.0"
VENV_DIR="${VENV_DIR:-.venv}"

# Proxy settings (set these if needed)
# export https_proxy="http://your-proxy:port"

# ----------
# Check uv installation
# ----------
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Install it with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "=== Installing verl dependencies using uv ==="

# ----------
# Create venv
# ----------
if [ ! -d "${VENV_DIR}" ]; then
    echo ">>> Creating virtual environment at ${VENV_DIR} with Python ${PYTHON_VERSION}..."
    uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}"
else
    echo ">>> Using existing virtual environment at ${VENV_DIR}"
fi

# ----------
# PyTorch
# ----------
echo ">>> Installing PyTorch 2.8.0+cu128..."
uv pip install --python "${VENV_DIR}/bin/python" \
    "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0" \
    --index-url https://download.pytorch.org/whl/cu128

# -------------------
# Flash Attention 2
# -------------------
echo ">>> Installing Flash Attention 2..."
uv pip install --python "${VENV_DIR}/bin/python" ninja==1.13.0 psutil

# Install prebuilt wheel
VERSION="2.8.3"
ABI_FLAG=$("${VENV_DIR}/bin/python" -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
FILE="flash_attn-${VERSION}+cu12torch${TORCH_VERSION}cxx11abi${ABI_FLAG}-cp${PYTHON_VERSION_SHORT}-cp${PYTHON_VERSION_SHORT}-linux_x86_64.whl"
REPO="Dao-AILab/flash-attention"
URL="https://github.com/${REPO}/releases/download/v${VERSION}/${FILE}"

echo ">>> Downloading flash-attn from: ${URL}"
uv pip install --python "${VENV_DIR}/bin/python" "${URL}"

# ------------
# FlashInfer
# ------------
echo ">>> Installing FlashInfer ${FLASHINFER_VERSION}..."
uv pip install --python "${VENV_DIR}/bin/python" \
    "flashinfer-python==${FLASHINFER_VERSION}" --prerelease=allow

# ---------------
# Miscellaneous
# ---------------
echo ">>> Installing miscellaneous dependencies..."
uv pip install --python "${VENV_DIR}/bin/python" \
    transformers==4.57.1 \
    hydra-core \
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
    "numpy<2.0.0" \
    pytest \
    pybind11 \
    codetiming \
    torchdata \
    datasets \
    peft \
    qwen_vl_utils \
    mathruler \
    pylatexenc \
    cupy-cuda12x

# Additional dependencies from setup.py
echo ">>> Installing additional dependencies from setup.py..."
uv pip install --python "${VENV_DIR}/bin/python" \
    accelerate \
    dill \
    pandas \
    "pyarrow>=19.0.0" \
    "ray[default]>=2.41.0" \
    wandb \
    "packaging>=20.0" \
    tensorboard

# ------
# vLLM
# ------
echo ">>> Installing vLLM from GitHub fork..."
# Install from suffix_decode branch with precompiled wheel for faster installation
export VLLM_USE_PRECOMPILED=1
#export VLLM_PRECOMPILED_WHEEL_LOCATION="https://wheels.vllm.ai/${VLLM_VERSION}/vllm-${VLLM_VERSION}-cp38-abi3-manylinux1_x86_64.whl"
uv pip install vllm==0.11.0

# -----
# SGLang
# -----

uv pip install "sglang[all]==0.5.2" --no-cache-dir

# ----------------
# verl
# ----------------
echo ">>> Installing verl in editable mode..."
uv pip install --python "${VENV_DIR}/bin/python" --no-deps -e .

# ----------
# Epilogue
# ----------
echo ""
echo "=== Dependency installation complete ==="
echo ""
echo "Activate the environment with:"
echo "  source ${VENV_DIR}/bin/activate"