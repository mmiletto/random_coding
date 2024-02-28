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

set_target_properties(hello_world PROPERTIES CUDA_ARCHITECTURES "50;72;86;89")

target_compile_options(hello_world PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        --use_fast_math
        -Xptxas="-v" # show register usage for each kernel on each target arch version
        >)


