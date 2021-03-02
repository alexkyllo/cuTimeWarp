/** CUDA implementation of Soft DTW
 *  @file soft_dtw.cu
 */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

typedef unsigned int uint;

/** Take the softmin of 3n elements
 * @param a The first element
 * @param b The second element
 * @param c The third element
 * @param gamma The smoothing factor
 */
__device__ float softmin(float a, float b, float c, const float gamma)
{
    a /= -gamma;
    b /= -gamma;
    c /= -gamma;
    float max_of = max(max(a, b), c);
    float sum = exp(a - max_of) + exp(b - max_of) + exp(c - max_of);

    return -gamma * (log(sum) + max_of);
}

__host__ void sgemm_cublas(const float *A, const float *B, float *C,
                           const uint m, const uint k, const uint n,
                           const float alpha)
{
    const float beta = 0.0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    // call cuBLAS to multiply transposed matrices
    // (input is row-major but cublas expects column major
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha, A, m, B, k,
                &beta, C, m);
    cublasDestroy(handle);
}

__global__ void sq_euclid_norm(const uint m, const uint n, const float *X,
                               float *XX)
{
    // TODO
    uint i = blockIdx.y * blockDim.y + threadIdx.y;
    uint j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m && j < n)
    {
        float x = X[i * n + j];
        XX[i] += x * x;
    }
}

__global__ void euclid_dist(const uint m, const uint n, const float *XX,
                            const float *YY, const float *XY, float *D)
{
    uint i = blockIdx.y * blockDim.y + threadIdx.y;
    uint j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m && j < n)
    {
        D[i * n + j] = XX[i] + YY[j] - (XY[i * n + j]);
    }
}

__host__ void sq_euclid_dist(const float *X, const float *Y, float *D,
                             const uint m, const uint n, const uint k)
{
    // TODO: This needs testing
    float *dX;
    float *dY;
    float *XX; // = new float[m]{0};
    float *YY; // = new float[n]{0};
    float *XY; // = new float[m * n]{0};
    cudaMalloc(&dX, m * sizeof(float));
    cudaMalloc(&dY, n * sizeof(float));
    cudaMalloc(&XX, m * sizeof(float));
    cudaMalloc(&YY, n * sizeof(float));
    cudaMalloc(&XY, m * n * sizeof(float));
    cudaMemset(XX, 0, m * sizeof(float));
    cudaMemset(YY, 0, n * sizeof(float));
    cudaMemset(XY, 0, m * n * sizeof(float));

    uint block_size = min(m, 1024);
    uint grid_size = (m + block_size - 1) / block_size;
    // compute squared euclidean norm of X
    sq_euclid_norm<<<grid_size, block_size>>>(m, k, X, XX);
    block_size = min(m, 1024);
    grid_size = (m + block_size - 1) / block_size;
    sq_euclid_norm<<<block_size, grid_size>>>(n, k, Y, YY);

    // compute (2*X)*YT
    // gemm_blas<T>(X, Y, XY, m, k, n, 2.0);
    sgemm_cublas(X, Y, XY, m, k, n, 2.0);
    // cublasHandle_t handle;
    // cublasCreate(&handle);
    // const float alpha = 2.0;
    // const float beta = 0.0;
    // // call cuBLAS to multiply transposed matrices
    // // (input is row-major but cublas expects column major
    // cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha, X, m, Y,
    // k,
    //             &beta, XY, m);

    // compute x^2 + y^2 - 2xy
    euclid_dist<<<block_size, grid_size>>>(m, n, XX, YY, XY, D);
    cudaFree(XX);
    cudaFree(YY);
    cudaFree(XY);
}
