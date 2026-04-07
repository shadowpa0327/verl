#!/bin/bash
# Install verl dependencies without installing the verl package itself
# Equivalent to the Dockerfile but as a standalone bash script
# Uses uv for venv creation and package installation
#
# Aligned with vllm018 Docker image:
#   - Python 3.12, CUDA 12.9, PyTorch 2.10, vLLM 0.18.0
#   - Supports TorchSpec KV connector (MooncakeHiddenStatesConnector)

set -e  # Exit on error

# ----------
# Configuration
# ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(dirname "$SCRIPT_DIR")"

PYTHON_VERSION="3.12"
PYTHON_VERSION_SHORT="312"
TORCH_VERSION="2.10"
VLLM_VERSION="0.18.0"
TRANSFORMERS_VERSION="5.3.0"
FLASH_ATTN_VERSION="2.8.3"
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
echo "    Python: ${PYTHON_VERSION}"
echo "    PyTorch: ${TORCH_VERSION}"
echo "    vLLM: ${VLLM_VERSION}"
echo "    Transformers: ${TRANSFORMERS_VERSION}"
echo ""

# ----------
# Create venv
# ----------
if [ ! -d "${VENV_DIR}" ]; then
    echo ">>> Creating virtual environment at ${VENV_DIR} with Python ${PYTHON_VERSION}..."
    uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}"
else
    echo ">>> Using existing virtual environment at ${VENV_DIR}"
fi

PIP="uv pip install --python ${VENV_DIR}/bin/python"

# ----------
# PyTorch (cu129, matching Docker image)
# ----------
echo ">>> Installing PyTorch ${TORCH_VERSION}.0+cu129..."
$PIP \
    "torch==${TORCH_VERSION}.0" "torchvision==0.25.0" "torchaudio==${TORCH_VERSION}.0" \
    --index-url https://download.pytorch.org/whl/cu129

# -------------------
# Flash Attention 2
# -------------------
echo ">>> Installing Flash Attention 2..."
$PIP ninja==1.13.0 psutil pybind11 wheel

# Try prebuilt wheel first, fall back to source build
VERSION="${FLASH_ATTN_VERSION}"
ABI_FLAG=$("${VENV_DIR}/bin/python" -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
FILE="flash_attn-${VERSION}+cu12torch${TORCH_VERSION}cxx11abi${ABI_FLAG}-cp${PYTHON_VERSION_SHORT}-cp${PYTHON_VERSION_SHORT}-linux_x86_64.whl"
REPO="Dao-AILab/flash-attention"
URL="https://github.com/${REPO}/releases/download/v${VERSION}/${FILE}"

echo ">>> Trying prebuilt flash-attn wheel: ${URL}"
if ! $PIP "${URL}" 2>/dev/null; then
    echo ">>> Prebuilt wheel not found, building from source..."
    FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=32 $PIP --no-build-isolation "flash_attn==${VERSION}"
fi

# ------
# vLLM
# ------
echo ">>> Installing vLLM ${VLLM_VERSION}..."
$PIP "vllm==${VLLM_VERSION}"

# ---------------
# Transformers
# ---------------
echo ">>> Installing transformers ${TRANSFORMERS_VERSION}..."
$PIP "transformers==${TRANSFORMERS_VERSION}"

# ---------------
# Miscellaneous (from Docker image + verl deps)
# ---------------
echo ">>> Installing miscellaneous dependencies..."
$PIP \
    hydra-core \
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
    pytest \
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
$PIP \
    accelerate \
    dill \
    pandas \
    "pyarrow>=19.0.0" \
    "ray[default]>=2.41.0" \
    wandb \
    "packaging>=20.0" \
    tensorboard

# ---------------
# TRL (no-deps, matching Docker)
# ---------------
echo ">>> Installing trl..."
$PIP --no-deps "trl==0.27.0"

# ---------------
# Mooncake (for drafter co-training)
# ---------------
echo ">>> Installing mooncake..."
$PIP mooncake || echo "WARNING: mooncake install failed (may need system RDMA libs). Drafter co-training requires this."

# ----------------
# verl (editable)
# ----------------
echo ">>> Installing verl in editable mode..."
$PIP --no-deps -e "${VERL_ROOT}"

# ----------
# Epilogue
# ----------
echo ""
echo "=== Dependency installation complete ==="
echo ""
echo "Key versions:"
echo "  Python:       ${PYTHON_VERSION}"
echo "  PyTorch:      ${TORCH_VERSION}.0+cu129"
echo "  vLLM:         ${VLLM_VERSION}"
echo "  Transformers: ${TRANSFORMERS_VERSION}"
echo "  Flash Attn:   ${FLASH_ATTN_VERSION}"
echo ""
echo "Activate the environment with:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Test drafter pipeline:"
echo "  python scripts/test_vllm_hs_collector.py --stage 1"
