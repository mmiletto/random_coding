#include "axpy.h"

/*
	AXPY Operation A = A * X plus Y
	Scalar multiplication and vector addition.
*/
__global__ void axpy(float *a, float* x, float y, const size_t N)
{
	const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		a[i] = a[i] * x[i] + y;
	}
}