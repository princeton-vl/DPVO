cmake_minimum_required(VERSION 3.4...3.18)
project(dpretrieval)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(DBoW2 REQUIRED)
include_directories(${DBoW2_INCLUDE_DIRS})


add_subdirectory(pybind11)
pybind11_add_module(dpretrieval src/main.cpp)
target_link_libraries(dpretrieval PRIVATE ${OpenCV_LIBS} ${DBoW2_LIBS} )

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(dpretrieval
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})