#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${NANOVLLM_CONDA_ENV:-nano-vllm-server}"
TORCH_VERSION="${NANOVLLM_TORCH_VERSION:-2.8.0}"
TORCH_CUDA_INDEX="${NANOVLLM_TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda was not found. Install Miniconda first." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi was not found. Load/install the NVIDIA driver before continuing." >&2
  exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
  cat >&2 <<'MSG'
ERROR: nvcc was not found on PATH.
The PyTorch wheel provides the CUDA runtime, but building the Megakernel
extension requires a CUDA toolkit/nvcc installation. Load the server's CUDA
module or add nvcc to PATH, then rerun this script.
MSG
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env update --name "${ENV_NAME}" --file "${ROOT_DIR}/environment.server.yml" --prune
else
  conda env create --file "${ROOT_DIR}/environment.server.yml" --name "${ENV_NAME}"
fi

conda_run=(conda run --no-capture-output --name "${ENV_NAME}")
"${conda_run[@]}" python -m pip install --upgrade pip
"${conda_run[@]}" python -m pip install \
  "torch==${TORCH_VERSION}" \
  --index-url "${TORCH_CUDA_INDEX}"
"${conda_run[@]}" python -m pip install \
  --requirement "${ROOT_DIR}/requirements.server.txt"
"${conda_run[@]}" python -m pip install \
  --editable "${ROOT_DIR}" \
  --no-deps
"${conda_run[@]}" python "${ROOT_DIR}/tools/verify_server_env.py"

echo
echo "Environment ready: ${ENV_NAME}"
echo "Activate it with: conda activate ${ENV_NAME}"
