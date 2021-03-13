#pragma once
#ifndef __CUDACC__
#define __device__
#define __host__
#define __global__
#endif
typedef unsigned int uint;
__global__ void convert_diagonal(float *D, float *DD, uint m, uint n);
__host__ void convert_diagonal_major(float *D, float *DD, uint m, uint n);