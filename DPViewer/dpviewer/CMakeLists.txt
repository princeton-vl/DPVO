
# optional libraries
find_package(Pangolin REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

pybind11_add_module(dpviewerx viewer.cpp viewer_cuda.cu)
include_directories(${Pangolin_INCLUDE_DIRS})

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(dpviewerx PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_link_libraries(dpviewerx PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} ${Pangolin_LIBRARIES})

