/** CUDA implementation of Soft DTW
 *  @file timewarp.cu
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

void sgemm_cublas(const float *A, const float *B, float *C, const uint m,
                  const uint k, const uint n, const float alpha)
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

__host__ void sq_euclidean_distance(const float *X, const float *Y, float *D,
                                    const uint m, const uint n, const uint k)
{
    // TODO, work in progress
    float *XX; // = new float[m]{0};
    float *YY; // = new float[n]{0};
    float *XY; // = new float[m * n]{0};
    cudaMalloc(&XX, m * sizeof(float));
    cudaMalloc(&YY, n * sizeof(float));
    cudaMalloc(&XY, m * n * sizeof(float));
    cudaMemset(XX, 0, m * sizeof(float));
    cudaMemset(YY, 0, n * sizeof(float));
    cudaMemset(XY, 0, m * n * sizeof(float));

    // compute squared euclidean norm of X
    // TODO: write this as a device function
    for (uint i = 0; i < m; i++)
    {
        for (uint j = 0; j < k; j++)
        {
            float x = X[i * k + j];
            XX[i] += x * x;
        }
    }

    // compute squared euclidean norm of Y
    for (uint i = 0; i < n; i++)
    {
        for (uint j = 0; j < k; j++)
        {
            float y = Y[i * k + j];
            YY[i] += y * y;
        }
    }

    // compute (2*X)*YT
    // gemm_blas<T>(X, Y, XY, m, k, n, 2.0);
    sgemm_cublas(X, Y, XY, m, k, n, 2.0);

    // compute x^2 + y^2 - 2xy
    // TODO: write this as a device function
    for (uint i = 0; i < m; i++)
    {
        for (uint j = 0; j < n; j++)
        {
            D[i * n + j] = XX[i] + YY[j] - (XY[i * n + j]);
        }
    }

    delete[] XX;
    delete[] YY;
    delete[] XY;
}
