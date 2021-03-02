#include "matrix.hcu"

matrix::matrix(float *A, uint m, uint n)
{
    this.m = m;
    this.n = n;
    uint sz = m * n * sizeof(float);
    cudaMalloc(&dA, sz);
    cudaMemcpy(dA, A, sz, cudaMemcpyDeviceToHost);
}

__device__ inline float matrix::get(uint i, uint j)
{
    return dA[i * k + j];
}
