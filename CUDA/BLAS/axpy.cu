#include <axpy.h>

/*
	AXPY Operation = A * X plus Y
	Scalar multiplication and vector addition.
*/
__global__ void axpy(float *a, float* x, float y, const size_t N)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		y[i] = a[i] * x[i] + y;
	}
}