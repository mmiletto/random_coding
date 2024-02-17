cmake_minimum_required(VERSION 3.26.4)  # CMake version check

project(hello_world CXX CUDA)

enable_language(CUDA)
enable_language(CXX)
find_package(CUDAToolkit REQUIRED)

set (CMAKE_CXX_STANDARD 23)             # Enable c++23 standard

message("CUDAToolkit_INCLUDE_DIR ${CUDAToolkit_INCLUDE_DIR}")
message("CUDA_CUDART ${CUDA_CUDART}")


add_executable(
    hello_world
        main.cu
        BLAS/axpy.h
        BLAS/axpy.cu
        main.cu
)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURE "89")