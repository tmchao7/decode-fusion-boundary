#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${NANOVLLM_VENV:-.venv}"
PYTHON_VERSION="${NANOVLLM_PYTHON_VERSION:-3.11}"
TORCH_VERSION="${NANOVLLM_TORCH_VERSION:-2.8.0}"
TORCH_CUDA_INDEX="${NANOVLLM_TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/${ENV_DIR}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi was not found. Load/install the NVIDIA driver first." >&2
  exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
  cat >&2 <<'MSG'
ERROR: nvcc was not found on PATH.
Load the server CUDA toolkit/module before running this script. PyTorch's
wheel includes the CUDA runtime, but Megakernel compilation still needs nvcc.
MSG
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv was not found; installing it under ~/.local/bin..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is still unavailable after installation." >&2
  exit 1
fi

cd "${ROOT_DIR}"
uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}"
PYTHON="${VENV_DIR}/bin/python"

uv pip install --python "${PYTHON}" --upgrade pip setuptools wheel
uv pip install --python "${PYTHON}" ninja packaging cmake
uv pip install --python "${PYTHON}" \
  "torch==${TORCH_VERSION}" \
  --index-url "${TORCH_CUDA_INDEX}"

# FlashAttention is installed separately because it may need a local CUDA build.
grep -v '^flash-attn' requirements.server.txt > /tmp/nano-vllm-requirements-no-flash.txt
uv pip install --python "${PYTHON}" \
  --requirement /tmp/nano-vllm-requirements-no-flash.txt
uv pip install --python "${PYTHON}" \
  "flash-attn>=2.7,<3" \
  --no-build-isolation

uv pip install --python "${PYTHON}" --editable "${ROOT_DIR}" --no-deps

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
"${PYTHON}" -m tools.verify_server_env

cat <<MSG

Environment ready: ${VENV_DIR}
Activate it with:
  source ${VENV_DIR}/bin/activate
MSG
