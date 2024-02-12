#include <vector>
#include "BLAS/axpy.h"

int main(int argc, char const *argv[])
{
	// Problem variables
	std::vector<float> a, x;
	float y = 1.0;
	size_t N = 10000;
	// device side
	float* dev_a, dev_x;

	// Problem setup
	a.resize(N);
	x.resize(N);


	// copy to device
	cudaMalloc((void **)dev_a, N * sizeof(float));
	cudaMalloc((void **)dev_x, N * sizeof(float));

	cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_x, x, N * sizeof(float), cudaMemcpyHostToDevice);


	// call kernel
	size_t n_threads = 512;
	size_t n_blocks = std::ceil(N / n_threads);
	axpy<<<n_blocks, n_threads>>>(dev_a, dev_x, y, N);

	//copy back
	cudaMemcpy(dev_x, x, N * sizeof(float), cudaMemcpyDeviceToHost);


	return 0;
}