#include <cuda.h>

__global__ void axpy(float *a, float* x, float y, const size_t N)
