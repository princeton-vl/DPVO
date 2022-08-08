cmake_minimum_required(VERSION 3.8)

project(dpviewer)
project(viewer LANGUAGES CUDA CXX)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CXX_FLAGS}")


include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})


list(APPEND CMAKE_PREFIX_PATH ${TORCH_PATH})

find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

include_directories(
    ${EIGEN_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS} 
) 

add_subdirectory(pybind11)
add_subdirectory(dpviewer)


