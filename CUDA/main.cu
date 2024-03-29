#include <vector>
#include <iostream>
#include <cmath>

#include "BLAS/axpy.h"
#include <cassert>

int main() {
    // Problem variables
    std::vector<float> a;
    std::vector<float> x;
    float y = 1.0;
    size_t N = 1000;

    // device side
    float *dev_a;
    float *dev_x;

    // Problem setup
    a.resize(N);
    x.resize(N);
    for (int i = 0; i < N; i++) {
        a[i] = 1;
        x[i] = (float) i;
    }

    cudaSetDevice(0);
    // copy to device
    cudaMalloc((void **) &dev_a, N * sizeof(float));
    cudaMalloc((void **) &dev_x, N * sizeof(float));

    cudaMemcpy(dev_a, a.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x, x.data(), N * sizeof(float), cudaMemcpyHostToDevice);


    // call kernel
    size_t n_threads = 512;
    size_t n_blocks = std::ceil(float(N) / n_threads);
    axpy<<<n_blocks, n_threads>>>(dev_a, dev_x, y, N);

	//copy back
	cudaMemcpy(x.data(), dev_a, N * sizeof(float), cudaMemcpyDeviceToHost);

    int i = 0;
    for (auto& value : x)
    {
        assert(value == 1 * i + 1);
        i++;
    }


	return 0;
}