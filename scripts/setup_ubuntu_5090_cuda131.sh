#!/usr/bin/env bash
set -euo pipefail

# Quick setup for Ubuntu + RTX 5090 + CUDA 12.8 (no viewer).
#
# Optional environment variables:
#   ENV_NAME=dpvo5090
#   PYTHON_VERSION=3.10
#   CUDA_HOME=/usr/local/cuda-12.8
#   TORCH_VERSION=2.9.1+cu128
#   TORCHVISION_VERSION=0.24.1+cu128
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
#   TORCH_INSTALL_POLICY=auto # auto|fail|always
#   FORCE_TORCH_REINSTALL=0   # set to 1 to force reinstall torch packages
#   PIP_NO_CACHE=0            # set to 1 to pass --no-cache-dir to pip
#   TORCH_CUDA_ARCH_LIST=12.0
#   SKIP_APT=0                # set to 1 to skip apt dependencies
#   SKIP_DATA=0               # set to 1 to skip model/data download
#   RUN_DEMO=0                # set to 1 to run demo after setup

ENV_NAME="${ENV_NAME:-dpvo5090}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
EXPECTED_CUDA_VERSION="${EXPECTED_CUDA_VERSION:-12.8}"
EXPECTED_TORCH_CUDA_TAG="${EXPECTED_TORCH_CUDA_TAG:-cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.9.1+cu128}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.24.1+cu128}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_INSTALL_POLICY="${TORCH_INSTALL_POLICY:-auto}"
FORCE_TORCH_REINSTALL="${FORCE_TORCH_REINSTALL:-0}"
PIP_NO_CACHE="${PIP_NO_CACHE:-0}"
# Resolve CUDA path with sensible defaults for CUDA 12.8 hosts.
if [[ -n "${CUDA_HOME:-}" ]]; then
  CUDA_HOME="${CUDA_HOME}"
elif [[ -d /usr/local/cuda-12.8 ]]; then
  CUDA_HOME="/usr/local/cuda-12.8"
elif [[ -d /usr/local/cuda ]]; then
  CUDA_HOME="/usr/local/cuda"
elif command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
else
  CUDA_HOME="/usr/local/cuda-12.8"
fi
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
SKIP_APT="${SKIP_APT:-0}"
SKIP_DATA="${SKIP_DATA:-0}"
RUN_DEMO="${RUN_DEMO:-0}"

# Python probes in this script read these values from os.environ.
export EXPECTED_CUDA_VERSION
export TORCH_VERSION
export TORCHVISION_VERSION

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "[1/7] System checks"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | sed -n '1,20p'
else
  echo "ERROR: nvidia-smi not found. Run this script on the Ubuntu 5090 machine."
  exit 1
fi

if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | sed -n '1,20p'
else
  echo "WARNING: nvcc not found in PATH. Will rely on CUDA_HOME=${CUDA_HOME}."
fi

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "ERROR: nvcc not found at ${CUDA_HOME}/bin/nvcc"
  echo "Set CUDA_HOME to your toolkit path, e.g. CUDA_HOME=/usr/local/cuda-12.8"
  exit 1
fi
echo "Using CUDA_HOME=${CUDA_HOME}"
DETECTED_CUDA_VERSION="$("${CUDA_HOME}/bin/nvcc" --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n1)"
if [[ -z "${DETECTED_CUDA_VERSION}" ]]; then
  echo "ERROR: Could not parse CUDA toolkit version from ${CUDA_HOME}/bin/nvcc --version"
  exit 1
fi
if [[ "${DETECTED_CUDA_VERSION}" != "${EXPECTED_CUDA_VERSION}" ]]; then
  echo "ERROR: CUDA toolkit mismatch. Expected ${EXPECTED_CUDA_VERSION}, found ${DETECTED_CUDA_VERSION}."
  echo "Set CUDA_HOME to a CUDA ${EXPECTED_CUDA_VERSION} toolkit path, e.g. /usr/local/cuda-12.8"
  exit 1
fi
echo "Detected CUDA toolkit version: ${DETECTED_CUDA_VERSION}"

if [[ "${SKIP_APT}" != "1" ]] && command -v apt-get >/dev/null 2>&1; then
  echo "[2/7] Installing system dependencies"
  APT_INSTALL="apt-get install -y build-essential git wget unzip ninja-build"
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update
    sudo bash -lc "${APT_INSTALL}"
  else
    apt-get update
    bash -lc "${APT_INSTALL}"
  fi
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is required but not found."
  exit 1
fi

echo "[3/7] Creating conda env: ${ENV_NAME}"
eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi
conda activate "${ENV_NAME}"

echo "[4/7] Installing Python dependencies"
if [[ "${TORCH_INSTALL_POLICY}" != "auto" ]] && [[ "${TORCH_INSTALL_POLICY}" != "fail" ]] && [[ "${TORCH_INSTALL_POLICY}" != "always" ]]; then
  echo "ERROR: TORCH_INSTALL_POLICY must be one of: auto|fail|always"
  exit 1
fi

for required_var in EXPECTED_CUDA_VERSION TORCH_VERSION TORCHVISION_VERSION; do
  if [[ -z "${!required_var}" ]]; then
    echo "ERROR: required variable ${required_var} is empty."
    exit 1
  fi
done

python -m pip install -U pip setuptools wheel

torch_needs_install="0"
torch_force_reinstall="0"
torch_install_reason=""

if [[ "${FORCE_TORCH_REINSTALL}" == "1" ]]; then
  torch_needs_install="1"
  torch_force_reinstall="1"
  torch_install_reason="FORCE_TORCH_REINSTALL=1"
elif [[ "${TORCH_INSTALL_POLICY}" == "always" ]]; then
  torch_needs_install="1"
  torch_force_reinstall="1"
  torch_install_reason="TORCH_INSTALL_POLICY=always"
else
  if python - <<'PY'
import os
import sys

req_torch = os.environ["TORCH_VERSION"]
req_torchvision = os.environ["TORCHVISION_VERSION"]
req_cuda = os.environ["EXPECTED_CUDA_VERSION"]

try:
    import torch
except Exception:
    print("torch is not installed.", file=sys.stderr)
    sys.exit(10)

try:
    import torchvision
except Exception:
    print("torchvision is not installed.", file=sys.stderr)
    sys.exit(10)

cur_torch = torch.__version__
cur_torchvision = torchvision.__version__
cur_cuda = torch.version.cuda

mismatches = []
if cur_torch != req_torch:
    mismatches.append(f"torch version: expected {req_torch}, found {cur_torch}")
if cur_torchvision != req_torchvision:
    mismatches.append(f"torchvision version: expected {req_torchvision}, found {cur_torchvision}")
if cur_cuda != req_cuda:
    mismatches.append(f"torch CUDA runtime: expected {req_cuda}, found {cur_cuda}")

if mismatches:
    print("Detected torch installation mismatch:", file=sys.stderr)
    for msg in mismatches:
        print(f"  - {msg}", file=sys.stderr)
    sys.exit(11)

print(f"torch/torchvision already satisfied: {cur_torch}, {cur_torchvision} (cuda {cur_cuda})")
PY
  then
    :
  else
    probe_rc=$?
    if [[ "${TORCH_INSTALL_POLICY}" == "fail" ]]; then
      echo "ERROR: torch/torchvision missing or mismatched and TORCH_INSTALL_POLICY=fail."
      echo "Set TORCH_INSTALL_POLICY=auto (default) or FORCE_TORCH_REINSTALL=1 to repair automatically."
      exit 1
    fi
    torch_needs_install="1"
    if [[ "${probe_rc}" == "11" ]]; then
      torch_force_reinstall="1"
      torch_install_reason="existing torch installation mismatch"
    elif [[ "${probe_rc}" == "10" ]]; then
      torch_install_reason="missing torch/torchvision"
    else
      echo "ERROR: torch probe failed with unexpected exit code ${probe_rc}."
      echo "Please inspect the previous error output before retrying."
      exit 1
    fi
  fi
fi

if [[ "${torch_needs_install}" == "1" ]]; then
  echo "Installing torch packages (${torch_install_reason})"
  pip_args=(install --upgrade)
  if [[ "${PIP_NO_CACHE}" == "1" ]]; then
    pip_args+=(--no-cache-dir)
  fi
  if [[ "${torch_force_reinstall}" == "1" ]]; then
    pip_args+=(--force-reinstall)
  fi
  pip_args+=(
    "torch==${TORCH_VERSION}"
    "torchvision==${TORCHVISION_VERSION}"
    --index-url "${TORCH_INDEX_URL}"
  )
  python -m pip "${pip_args[@]}"
fi

python -m pip install numpy==1.26.4 numba tqdm einops yacs opencv-python matplotlib scipy plyfile evo

echo "[5/7] Building DPVO extensions"
mkdir -p thirdparty
if [[ ! -d thirdparty/eigen-3.4.0 ]]; then
  wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
  unzip -q -o eigen-3.4.0.zip -d thirdparty
fi
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST
export EXPECTED_CUDA_VERSION
export EXPECTED_TORCH_CUDA_TAG
python - <<'PY'
import os
import sys
import torch

expected_cuda = os.environ["EXPECTED_CUDA_VERSION"]
expected_tag = os.environ["EXPECTED_TORCH_CUDA_TAG"]
torch_cuda = torch.version.cuda
torch_ver = torch.__version__
if torch_cuda != expected_cuda:
    print(f"ERROR: torch.version.cuda={torch_cuda} does not match expected toolkit {expected_cuda}", file=sys.stderr)
    print("Reinstall torch/torchvision with matching CUDA wheels before building DPVO.", file=sys.stderr)
    sys.exit(1)
if expected_tag not in torch_ver:
    print(f"ERROR: torch build tag mismatch. Expected '{expected_tag}' in torch.__version__={torch_ver}", file=sys.stderr)
    print("Reinstall torch/torchvision with TORCH_INSTALL_POLICY=auto or FORCE_TORCH_REINSTALL=1.", file=sys.stderr)
    sys.exit(1)
print("torch available for build:", torch_ver, "cuda:", torch_cuda)
PY
# setup.py imports torch extensions, so we must build in the active env.
python -m pip install -v --no-build-isolation .

if [[ "${SKIP_DATA}" != "1" ]]; then
  echo "[6/7] Downloading model and sample data"
  ./download_models_and_data.sh
fi

echo "[7/7] Smoke test"
python - <<'PY'
import torch
import cuda_corr
import cuda_ba
import lietorch_backends

print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
print("extensions: ok")
PY

if [[ "${RUN_DEMO}" == "1" ]]; then
  if [[ ! -f movies/IMG_0492.MOV ]]; then
    echo "ERROR: movies/IMG_0492.MOV not found. Re-run with SKIP_DATA=0."
    exit 1
  fi
  python demo.py --imagedir=movies/IMG_0492.MOV --calib=calib/iphone.txt --stride=5 --plot
fi

echo "Setup complete."
