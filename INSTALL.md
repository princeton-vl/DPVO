# DPVO Installation Guide

This guide documents the installation process for DPVO on modern NVIDIA GPUs (RTX 40/50 series) with CUDA 12.8 and PyTorch 2.8.

## Environment

- **GPU**: NVIDIA RTX 5090 (Blackwell architecture, compute capability 12.0)
- **CUDA**: 12.8
- **Python**: 3.12
- **PyTorch**: 2.8.0

## Prerequisites

- NVIDIA Driver compatible with CUDA 12.8
- Conda (Miniconda or Anaconda)
- C++ compiler (g++)
- CUDA Toolkit 12.8

---

## Installation Steps

### 1. Create Conda Environment

```bash
conda create -n dpvo python=3.12
conda activate dpvo
```

### 2. Install Python Packages

```bash
# PyTorch with CUDA 12.8 support
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# PyTorch Scatter (for sparse operations)
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

# Other dependencies
pip install tensorboard numba tqdm einops pypose kornia numpy plyfile evo opencv-python yacs
```

### 3. Build CUDA Extensions

Set the CUDA architecture for your GPU and build:

```bash
# For RTX 5090 (Blackwell, sm_120)
export TORCH_CUDA_ARCH_LIST="12.0"

# For RTX 4090 (Ada Lovelace, sm_89)
# export TORCH_CUDA_ARCH_LIST="8.9"

# For RTX 3090 (Ampere, sm_86)
# export TORCH_CUDA_ARCH_LIST="8.6"

# Build and install
pip install --no-build-isolation .
```

### 4. Install Pangolin (Required for DPViewer)

DPViewer requires the Pangolin library for 3D visualization.

> **⚠️ CRITICAL: ABI Compatibility**
>
> Pangolin **MUST** be built with `-D_GLIBCXX_USE_CXX11_ABI=1` to match PyTorch 2.8.0.
> If you have an existing Pangolin installation, you must rebuild it with this flag.
> Otherwise, you will get `undefined symbol` errors when importing DPViewer.

```bash
# Install dependencies
sudo apt-get install libglew-dev libpython3-dev libeigen3-dev libgl1-mesa-dev \
    libwayland-dev libxkbcommon-dev wayland-protocols libepoxy-dev

# Clone Pangolin (or use existing clone)
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin

# IMPORTANT: Clean any previous build
rm -rf build
mkdir build && cd build

# Build with CXX11 ABI=1 (MUST match PyTorch 2.8.0)
cmake .. -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"
make -j$(nproc)
sudo make install
sudo ldconfig
cd ../..
```

**Verify Pangolin ABI** (should show `std::__cxx11::basic_string`):
```bash
nm -DC /usr/local/lib/libpango_display.so | grep "CreateWindowAndBind"
# Expected output (ABI=1, correct):
#   pangolin::CreateWindowAndBind(std::__cxx11::basic_string<char, ...>, ...)
# Wrong output (ABI=0, needs rebuild):
#   pangolin::CreateWindowAndBind(std::string, ...)
```

### 5. Install DPViewer (Optional - for visualization)

```bash
# Clean any previous build
rm -rf DPViewer/build DPViewer/*.egg-info

# Install
pip install --no-build-isolation ./DPViewer
```

**Verify DPViewer installation:**
```bash
python -c "from dpviewerx import Viewer; print('DPViewer OK')"
```

### 6. Install Classical Backend (Optional - for large loop closure)

The classical backend uses DBoW2 for closing very large loops.

**Step 1. Install OpenCV C++ API:**

```bash
sudo apt-get install -y libopencv-dev
```

**Step 2. Build and Install DBoW2:**

```bash
cd DBoW2
mkdir -p build && cd build
cmake ..
make -j$(nproc)
sudo make install
cd ../..
```

**Step 3. Install DPRetrieval:**

```bash
pip install --no-build-isolation ./DPRetrieval/
```

---

## What Gets Installed

The `pip install .` command builds and installs:

1. **dpvo** - Main Python package
2. **cuda_corr** - CUDA extension for correlation operations
3. **cuda_ba** - CUDA extension for bundle adjustment
4. **lietorch_backends** - CUDA extension for Lie group operations (SE3, SO3, Sim3)

---

## Verification

After installation, verify by running:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
python -c "import dpvo; print('DPVO imported successfully')"
```

---

## Running Demo

```bash
# With visualization
python demo.py --imagedir=<path_to_images> --calib=<path_to_calibration> --stride=1 --viz

# Without visualization (if DPViewer not installed)
python demo.py --imagedir=<path_to_images> --calib=<path_to_calibration> --stride=1 --plot
```

---

## Troubleshooting

### CUDA Architecture Reference

| GPU Series | Architecture | Compute Capability |
|------------|--------------|-------------------|
| RTX 5090/5080 | Blackwell | 12.0 |
| RTX 4090/4080 | Ada Lovelace | 8.9 |
| RTX 3090/3080 | Ampere | 8.6 |
| RTX 2080 Ti | Turing | 7.5 |

To check your GPU's compute capability:
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

---

### Why `--no-build-isolation` is Required

When building CUDA extensions with `pip install`, pip normally creates an isolated virtual environment for the build process. This causes problems because:

1. **PyTorch dependency**: The CUDA extensions need to link against PyTorch's CUDA libraries during compilation
2. **Build isolation**: In an isolated environment, pip installs only packages listed in `build-requires`, but the extensions need the *exact* PyTorch version you installed
3. **Header files**: The extensions include PyTorch header files (`<torch/extension.h>`) which must match your installed version

**Without `--no-build-isolation`:**
```
ModuleNotFoundError: No module named 'torch'
```

**With `--no-build-isolation`:**
- pip uses your current conda environment directly
- The build can access your installed PyTorch
- Headers and libraries match correctly

---

### Understanding C++ ABI Compatibility

#### What is the C++ ABI?

The C++ ABI (Application Binary Interface) defines how C++ code is compiled at the binary level, including:
- How function names are encoded (name mangling)
- How `std::string` and other STL types are represented in memory
- How exceptions are handled

GCC introduced a new ABI in GCC 5.1 (2015) with the flag `_GLIBCXX_USE_CXX11_ABI`:
- **ABI=0** (old): Pre-C++11 std::string implementation
- **ABI=1** (new): C++11 compliant std::string with small string optimization

#### Why Does This Matter?

All C++ libraries that share objects (like `std::string`) must use the **same ABI**. If they don't:
```
undefined symbol: _ZN8pangolin19CreateWindowAndBindENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE...
```

This error shows Pangolin was built with ABI=0 (uses `std::string`) but DPViewer expects ABI=1 (uses `std::__cxx11::basic_string`).

#### Check PyTorch's ABI

```bash
python -c "import torch; print('CXX11_ABI:', torch._C._GLIBCXX_USE_CXX11_ABI)"
```

PyTorch 2.8.0 uses **ABI=1** (True), so all linked libraries must also use ABI=1.

#### Fixing ABI Mismatches

**Step-by-step fix for Pangolin ABI mismatch:**

```bash
# 1. Check current Pangolin ABI
nm -DC /usr/local/lib/libpango_display.so | grep "CreateWindowAndBind"
# If it shows "std::string" instead of "std::__cxx11::basic_string", rebuild is needed

# 2. Go to Pangolin directory and clean
cd ~/Pangolin
rm -rf build

# 3. Rebuild with correct ABI
mkdir build && cd build
cmake .. -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"
make -j$(nproc)
sudo make install
sudo ldconfig

# 4. Verify the fix
nm -DC /usr/local/lib/libpango_display.so | grep "CreateWindowAndBind"
# Should now show: std::__cxx11::basic_string<char, ...>

# 5. Rebuild DPViewer
cd ~/park/SOLUTION/DPVO
pip uninstall dpviewer -y
rm -rf DPViewer/build DPViewer/*.egg-info
pip install --no-build-isolation ./DPViewer

# 6. Verify DPViewer
python -c "from dpviewerx import Viewer; print('DPViewer OK')"
```

**For DPViewer** (already configured in CMakeLists.txt):
```cmake
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
```

---

### Error: `cannot convert 'at::DeprecatedTypeProperties' to 'c10::ScalarType'`

This error occurs with PyTorch 2.x due to deprecated API.

**Cause**: PyTorch 2.x deprecated the `.type()` method which returned `at::DeprecatedTypeProperties`. The new API uses `.scalar_type()` which returns `at::ScalarType` directly.

**Fix**: Change all `.type()` calls to `.scalar_type()`:

```cpp
// Before (PyTorch 1.x)
AT_DISPATCH_FLOATING_TYPES(tensor.type(), "kernel_name", ([&] { ... }));

// After (PyTorch 2.x)
AT_DISPATCH_FLOATING_TYPES(tensor.scalar_type(), "kernel_name", ([&] { ... }));
```

**Files modified:**

| File | Changes |
|------|---------|
| `dpvo/altcorr/correlation_kernel.cu` | 4 fixes |
| `dpvo/lietorch/src/lietorch_gpu.cu` | 19 fixes |
| `dpvo/lietorch/src/lietorch_cpu.cpp` | 19 fixes |
| `dpvo/lietorch/include/dispatch.h` | Updated macro |

---

### Error: `nvcc fatal: Unsupported gpu architecture 'compute_XX'`

Set the correct CUDA architecture for your GPU:
```bash
export TORCH_CUDA_ARCH_LIST="12.0"  # Adjust for your GPU
pip install --no-build-isolation .
```

---

### Error: `CUDA error: no kernel image is available for execution on the device`

This means the CUDA code was compiled for a different GPU architecture than your device.

**Check your GPU:**
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

**Check PyTorch supported architectures:**
```bash
python -c "import torch; print(torch.cuda.get_arch_list())"
```

**Rebuild with correct architecture:**
```bash
export TORCH_CUDA_ARCH_LIST="12.0"  # Match your GPU
pip uninstall dpvo lietorch cuda_corr cuda_ba -y
pip install --no-build-isolation .
```

---

### Error: `Unknown CMake command "python_add_library"`

This occurs in DPViewer when pybind11 can't find CMake's Python module properly.

**Fix applied in DPViewer/CMakeLists.txt:**
```cmake
# Use find_package(Python ...) not find_package(Python3 ...)
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development)
```

---

### Error: CMake finds wrong Python version

CMake may find system Python instead of conda Python.

**Check which Python CMake finds:**
```
-- Found Python: /usr/bin/python3.10 (wrong!)
-- Found Python: /home/user/miniconda3/envs/dpvo/bin/python3.12 (correct!)
```

**Fix applied in DPViewer/setup.py:**
```python
cmake_args = [
    "-DPython_EXECUTABLE={}".format(sys.executable),
    "-DPython3_EXECUTABLE={}".format(sys.executable),
    # ...
]
```

---

### Error: `libpango_windowing.so: cannot open shared object file`

Pangolin libraries are not in the library path.

**Fix:**
```bash
sudo ldconfig
# Or set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

---

### FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated

This is a warning, not an error. The code works but uses deprecated API.

**Location**: `dpvo/net.py:187`

**Fix** (optional):
```python
# Before
from torch.cuda.amp import autocast

# After
from torch.amp import autocast
# Use: autocast('cuda', enabled=True)
```

---

## Code Modifications Summary

### PyTorch 2.8 Compatibility

All `.type()` calls changed to `.scalar_type()` for PyTorch 2.x compatibility.

### DPViewer CMake Fixes

**DPViewer/CMakeLists.txt:**
```cmake
# Python detection for pybind11
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development)

# Match PyTorch's ABI
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
```

**DPViewer/setup.py:**
```python
# Ensure CMake uses conda Python
cmake_args = [
    "-DPython_EXECUTABLE={}".format(sys.executable),
    "-DPython3_EXECUTABLE={}".format(sys.executable),
    # ...
]
```

---

## Quick Reference

```bash
# Full installation sequence
conda create -n dpvo python=3.12
conda activate dpvo

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
pip install tensorboard numba tqdm einops pypose kornia numpy plyfile evo opencv-python yacs

export TORCH_CUDA_ARCH_LIST="12.0"
pip install --no-build-isolation .

# Optional: Pangolin + DPViewer
sudo apt-get install libglew-dev libpython3-dev libeigen3-dev libgl1-mesa-dev \
    libwayland-dev libxkbcommon-dev wayland-protocols libepoxy-dev

git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"
make -j$(nproc) && sudo make install && sudo ldconfig
cd ../..

# Verify Pangolin ABI (should show std::__cxx11::basic_string)
nm -DC /usr/local/lib/libpango_display.so | grep "CreateWindowAndBind"

rm -rf DPViewer/build DPViewer/*.egg-info
pip install --no-build-isolation ./DPViewer

# Verify DPViewer
python -c "from dpviewerx import Viewer; print('DPViewer OK')"
```
