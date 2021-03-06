#pragma once
#ifndef __CUDACC__
#define __device__
#define __host__
#endif

typedef unsigned int uint;

__device__ float softmin(float a, float b, float c, const float gamma);
__host__ void sgemm_cublas(const float *A, const float *B, float *C,
                           const uint m, const uint k, const uint n,
                           const float alpha);
__host__ void sq_euclid_dist(const float *X, const float *Y, float *D,
                             const uint m, const uint n, const uint k);
__host__ float softdtw_cuda_naive(float *D, float *R, uint m, uint n,
                                  float gamma);
__host__ void softdtw_grad_cuda_naive(float *D, float *R, float *E, uint m,
                                      uint n, float gamma);
