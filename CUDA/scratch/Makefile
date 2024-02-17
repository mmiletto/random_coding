CUDA_NVCC = nvcc
CUDA_FLAGS = -arch=sm_86

all: hello_world

hello_world: ../main.cu BLAS/axpy.h BLAS/axpy.cu
	$(CUDA_NVCC) $(CUDA_FLAGS) -o hello_world main.cu BLAS/axpy.cu

debug_hello_world: ../main.cu BLAS/axpy.h BLAS/axpy.cu
	$(CUDA_NVCC) $(CUDA_FLAGS) -G -o hello_world main.cu BLAS/axpy.cu

clean: hello_world
	rm hello_world