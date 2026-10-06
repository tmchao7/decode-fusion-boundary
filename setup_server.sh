#!/usr/bin/env bash
set -euo pipefail

# Create the GPU environment from an already-installed Python.  The script is
# deliberately explicit about the installer because restricted servers often
# have conda + pip working while uv's resolver cannot reach public PyPI.
#
# Overridable environment variables:
#   NANOVLLM_INSTALLER       conda or uv (default: conda when available)
#   NANOVLLM_VENV             environment directory (default: .venv)
#   NANOVLLM_PYTHON_VERSION   preferred Python minor (default: 3.11)
#   NANOVLLM_PYTHON_BIN       exact interpreter to use (uv mode only)
#   NANOVLLM_TORCH_VERSION    torch version (default: 2.8.0)
#   NANOVLLM_TORCH_CUDA_INDEX CUDA wheel index (default: official cu128)
#   NANOVLLM_PYPI_INDEX_URL   package mirror (default: Tuna)
#   NANOVLLM_FLASH_ATTN_WHEEL optional prebuilt flash-attn wheel URL/path
#   NANOVLLM_VERIFY_MODEL     optional model path for the final verifier
#   NANOVLLM_BUILD_EXTENSION  1 to build the taskgraph extension (default: 1)

ENV_DIR="${NANOVLLM_VENV:-.venv}"
PYTHON_VERSION="${NANOVLLM_PYTHON_VERSION:-3.11}"
PYTHON_BIN="${NANOVLLM_PYTHON_BIN:-}"
TORCH_VERSION="${NANOVLLM_TORCH_VERSION:-2.8.0}"
TORCH_CUDA_INDEX="${NANOVLLM_TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"
PYPI_INDEX_URL="${NANOVLLM_PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
FLASH_ATTN_WHEEL="${NANOVLLM_FLASH_ATTN_WHEEL:-}"
VERIFY_MODEL="${NANOVLLM_VERIFY_MODEL:-}"
BUILD_EXTENSION="${NANOVLLM_BUILD_EXTENSION:-1}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/${ENV_DIR}"
INSTALLER="${NANOVLLM_INSTALLER:-}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi was not found. Load/install the NVIDIA driver first." >&2
  exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
  cat >&2 <<'MSG'
ERROR: nvcc was not found on PATH.
Load the server CUDA toolkit/module before running this script. PyTorch's
wheel includes the CUDA runtime, but Megakernel compilation needs nvcc.
MSG
  exit 1
fi

if [ -z "${INSTALLER}" ]; then
  if command -v conda >/dev/null 2>&1; then
    INSTALLER=conda
  else
    INSTALLER=uv
  fi
fi
case "${INSTALLER}" in
  conda|uv) ;;
  *) echo "ERROR: NANOVLLM_INSTALLER must be conda or uv (got ${INSTALLER})" >&2; exit 1 ;;
esac

# Use an interpreter already installed on the server. This avoids downloading
# a managed CPython through uv on restricted networks.
if [ "${INSTALLER}" = uv ] && [ -z "${PYTHON_BIN}" ]; then
  for candidate in "python${PYTHON_VERSION}" python3.12 python3.11 python3.10 python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      candidate_path="$(command -v "${candidate}")"
      if "${candidate_path}" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)' >/dev/null 2>&1; then
        PYTHON_BIN="${candidate_path}"
        break
      fi
    fi
  done
fi

if [ "${INSTALLER}" = uv ]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv was not found; installing it under ~/.local/bin..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
  fi
  command -v uv >/dev/null 2>&1 || { echo "ERROR: uv is unavailable" >&2; exit 1; }
  if [ -z "${PYTHON_BIN}" ]; then
    echo "ERROR: no system Python 3.10, 3.11, or 3.12 was found." >&2
    echo "Set NANOVLLM_PYTHON_BIN=/path/to/python3.11 and rerun." >&2
    exit 1
  fi
  export UV_PYTHON_DOWNLOADS=0
  export UV_SYSTEM_CERTS="${UV_SYSTEM_CERTS:-true}"
  export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
  export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
  if [ ! -x "${VENV_DIR}/bin/python" ]; then
    uv venv "${VENV_DIR}" --python "${PYTHON_BIN}" --no-managed-python --no-python-downloads
  fi
  PYTHON="${VENV_DIR}/bin/python"
else
  command -v conda >/dev/null 2>&1 || { echo "ERROR: conda is unavailable" >&2; exit 1; }
  if [ ! -x "${VENV_DIR}/bin/python" ]; then
    conda create --prefix "${VENV_DIR}" "python=${PYTHON_VERSION}" -y
  fi
  PYTHON="${VENV_DIR}/bin/python"
fi

if ! "${PYTHON}" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)'; then
  echo "ERROR: ${PYTHON} is not Python 3.10-3.12" >&2
  exit 1
fi

if [ "${INSTALLER}" = uv ]; then
  pip_install() {
    uv pip install --python "${PYTHON}" --index-url "${PYPI_INDEX_URL}" "$@"
  }
else
  pip_install() {
    "${PYTHON}" -m pip install --index-url "${PYPI_INDEX_URL}" "$@"
  }
fi

cd "${ROOT_DIR}"
echo "Installer: ${INSTALLER}"
echo "Python: ${PYTHON}"
echo "Python package index: ${PYPI_INDEX_URL}"
echo "PyTorch CUDA fallback index: ${TORCH_CUDA_INDEX}"

pip_install --upgrade pip setuptools wheel
pip_install ninja packaging
if [ "${INSTALLER}" = uv ]; then
  uv pip install --python "${PYTHON}" \
    "torch==${TORCH_VERSION}" \
    --index-url "${PYPI_INDEX_URL}" \
    --extra-index-url "${TORCH_CUDA_INDEX}"
else
  "${PYTHON}" -m pip install \
    --index-url "${PYPI_INDEX_URL}" \
    --extra-index-url "${TORCH_CUDA_INDEX}" \
    "torch==${TORCH_VERSION}"
fi

tmp_requirements="$(mktemp)"
trap 'rm -f "${tmp_requirements}"' EXIT
grep -v '^flash-attn' requirements.server.txt > "${tmp_requirements}"
pip_install --requirement "${tmp_requirements}"
if [ -n "${FLASH_ATTN_WHEEL}" ]; then
  pip_install --no-deps "${FLASH_ATTN_WHEEL}"
else
  pip_install "flash-attn>=2.7,<3" --no-build-isolation
fi
pip_install --editable "${ROOT_DIR}" --no-deps

"${PYTHON}" -m pip check
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
verify_args=(--require-megakernel)
if [ -n "${VERIFY_MODEL}" ]; then
  verify_args+=(--model "${VERIFY_MODEL}")
fi
"${PYTHON}" -m tools.verify_server_env "${verify_args[@]}"
if [ "${BUILD_EXTENSION}" = 1 ]; then
  "${PYTHON}" -m nanovllm.backends.megakernel.build
fi

cat <<MSG

Environment ready: ${VENV_DIR}
Installer: ${INSTALLER}
Activate it with:
  source ${VENV_DIR}/bin/activate
MSG
