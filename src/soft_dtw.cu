/** CUDA implementation of Soft DTW
 *  @file soft_dtw.cu
 */
#include <cblas.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

typedef unsigned int uint;

#define cudaErrchk(ans)                                                        \
    {                                                                          \
        GPUAssert((ans), __FILE__, __LINE__);                                  \
    }
inline void GPUAssert(cudaError_t code, const char *file, int line,
                      bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
        {
            exit(code);
        }
    }
}

/** Take the softmin of 3 elements
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
    // Doesn't work, need to get the transpositions right
    // Try computing BT * AT = C and then 2.0 * A using cublasSscal
    const float beta = 0.0;
    const float alpha1 = 1.0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    // call cuBLAS to multiply transposed matrices
    float *tempA;
    cudaMalloc(&tempA, m * k * sizeof(float));
    cudaMemcpy(tempA, A, m * k * sizeof(float), cudaMemcpyDeviceToDevice);
    cublasSscal(handle, m * k, &alpha, tempA, 1);
    // (input is row-major but cublas expects column major
    cublasSgemm(handle,      // cublas handle
                CUBLAS_OP_T, // transpose first matrix
                CUBLAS_OP_N, // tranpose second matrix
                n,           // rows in first matrix
                m,           // columns in second matrix
                k,           // columns in first matrix
                &alpha1,     // scalar for first matrix
                B,           // first matrix
                k,           // stride of first matrix
                tempA,       // second matrix
                k,           // stride of second matrix
                &beta,       // scalar for C
                C,           // result matrix
                n            // stride of result matrix
    );
    cudaFree(tempA);
    cublasDestroy(handle);
}

__global__ void sq_euclid_norm(const uint m, const uint k, const float *X,
                               float *XX)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m)
    {
        for (uint j = 0; j < k; j++)
        {
            float x = X[i * k + j];
            XX[i] += x * x;
        }
    }
}

__global__ void euclid_dist(const uint m, const uint n, const float *XX,
                            const float *YY, const float *XY, float *D)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m)
    {
        for (uint j = 0; j < n; j++)
        {
            D[i * n + j] = XX[i] + YY[j] - (XY[i * n + j]);
            printf("XY[%d]) = %.2f\n", i * n + j, XY[i * n + j]);
        }
    }
}

void gemm_blas(const float *A, const float *B, float *C, uint m, uint k, uint n,
               float alpha)
{
    cblas_sgemm(CblasRowMajor, // Row-major striding
                CblasNoTrans,  // Do not transpose A
                CblasTrans,    // Transpose B
                m,             // Rows in A
                n,             // Columns in B
                k,             // Columns in A
                alpha,         // Alpha (scalar used to scale A*B)
                A,             // Input Matrix A
                k,             // LDA stride of matrix A
                B,             // Input Matrix B
                k,             // LDA stride of matrix B
                0.0,           // Beta (scalar used to scale matrix C)
                C,             // Result Matrix C
                n);            // LDA stride of matrix C
}

__host__ void sq_euclid_dist(const float *X, const float *Y, float *D,
                             const uint m, const uint n, const uint k)
{
    // TODO: This needs testing
    // TODO: change this to work on device arrays only
    float *dX;
    float *dY;
    float *dD;
    float *XX; // = new float[m]{0};
    float *YY; // = new float[n]{0};
    float *XY; // = new float[m * n]{0};
    size_t size_m = m * sizeof(float);
    size_t size_n = n * sizeof(float);
    size_t size_mn = n * size_m;
    size_t size_mk = k * size_m;
    size_t size_nk = k * size_n;
    cudaMalloc(&dD, size_mn);
    cudaMalloc(&dX, size_mk);
    cudaMalloc(&dY, size_nk);
    cudaMalloc(&XX, size_m);
    cudaMalloc(&YY, size_n);
    cudaMalloc(&XY, size_mn);
    cudaMemset(XX, 0, size_m);
    cudaMemset(YY, 0, size_n);
    cudaMemset(XY, 0, size_mn);
    cudaMemset(dD, 0, size_mn);
    cudaMemcpy(dX, X, size_mk, cudaMemcpyHostToDevice);
    cudaMemcpy(dY, Y, size_nk, cudaMemcpyHostToDevice);

    uint block_size = min(m, 1024);
    uint grid_size = (m + block_size - 1) / block_size;
    // compute squared euclidean norm of X
    sq_euclid_norm<<<grid_size, block_size>>>(m, k, dX, XX);
    block_size = min(n, 1024);
    grid_size = (n + block_size - 1) / block_size;
    sq_euclid_norm<<<block_size, grid_size>>>(n, k, dY, YY);

    // // compute (2*X)*YT
    // // gemm_blas<T>(X, Y, XY, m, k, n, 2.0);
    sgemm_cublas(dX, dY, XY, m, k, n, 2.0);
    // workaround for now, doing the matrix multiplication on CPU
    // so we can focus on the DTW algorithm on GPU
    // float *hXY = new float[m * n]{0};
    // gemm_blas(X, Y, hXY, m, k, n, 2.0);
    // cudaMemcpy(XY, hXY, size_mn, cudaMemcpyHostToDevice);
    //
    block_size = min(m, 1024);
    grid_size = (m + block_size - 1) / block_size;
    euclid_dist<<<block_size, grid_size>>>(m, n, XX, YY, XY, dD);
    cudaErrchk(cudaMemcpy(D, dD, size_mn, cudaMemcpyDeviceToHost));
    cudaFree(dD);
    cudaFree(XX);
    cudaFree(YY);
    cudaFree(XY);
}
