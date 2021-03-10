#include "euclid_dist.cuh"
/** A wrapper for cublasSgemm that works on row-major matrices by transposing
 *  A, B and C should be __device__ arrays
 *  @param A Input matrix of dimensions m * k
 *  @param B Input matrix of dimensions k * n
 *  @param C Result matrix of dimensions m * n
 *  @param m Height of matrix A and matrix C
 *  @param k Width of matrix A and height of matrix B
 *  @param n Width of matrix B and matrix C
 *  @param alpha A scalar to elementwise multiply A by
 */
__host__ void sgemm_cublas(const float *A, const float *B, float *C,
                           const uint m, const uint k, const uint n,
                           const float alpha)
{
    const float beta = 0.0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    // call cuBLAS to multiply transposed matrices B^T * A
    // (input is row-major but cublas expects column major)
    cublasSgemm(handle,      // cublas handle
                CUBLAS_OP_T, // transpose first matrix
                CUBLAS_OP_N, // tranpose second matrix
                n,           // rows in first matrix
                m,           // columns in second matrix
                k,           // columns in first matrix
                &alpha,      // scalar for first matrix
                B,           // first matrix
                k,           // stride of first matrix
                A,           // second matrix
                k,           // stride of second matrix
                &beta,       // scalar for C
                C,           // result matrix
                n            // stride of result matrix
    );
    cublasDestroy(handle);
}

/** CUDA kernel to compute the squared euclidean norm of matrix X
 *  @param m Height (rows) of matrix X
 *  @param k Width (columns) of matrix X
 *  @param XX a length m vector for the result
 */
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

/** CUDA kernel to compute the euclidean distance between two Euclidean norm
 * vectors XX and YY, i.e. X*X + Y*Y - 2X*Y
 *  @param m The length of vectors in X
 *  @param n The length of vectors in Y
 *  @param XX Squared Euclidean norm of X
 *  @param YY Squared Euclidean norm of Y
 *  @param XY 2 * X * Y^T (matrix multiplication result)
 *  @param D The result euclidean distance matrix with dimensions (m x n)
 */
__global__ void euclid_dist(const uint m, const uint n, const float *XX,
                            const float *YY, const float *XY, float *D)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m)
    {
        for (uint j = 0; j < n; j++)
        {
            D[i * n + j] = XX[i] + YY[j] - (XY[i * n + j]);
        }
    }
}

/** Host function to compute the Squared Euclidean distance between two sets of
 *  column vectors (e.g. two multivariate time series)
 *  X and Y by using the euclidian norms, i.e. X*X + Y*Y - 2X*Y
 *  Inputs X, Y, D should be __device__ arrays.
 *  @param X A set of vectors of length (row count) m
 *  @param Y A set of vectors of length (row count) n
 *  @param D A result array for the distance matrix of dimension (m x n)
 *  @param m The length of vectors in X
 *  @param n The length of vectors in Y
 *  @param k The number of vectors in X and Y (columns)
 */
__host__ void sq_euclid_dist(const float *X, const float *Y, float *D,
                             const uint m, const uint n, const uint k)
{
    float *XX;
    float *YY;
    float *XY;
    size_t size_m = m * sizeof(float);
    size_t size_n = n * sizeof(float);
    size_t size_mn = n * size_m;
    cudaMalloc(&XX, size_m);
    cudaMalloc(&YY, size_n);
    cudaMalloc(&XY, size_mn);
    cudaMemset(XX, 0, size_m);
    cudaMemset(YY, 0, size_n);
    cudaMemset(XY, 0, size_mn);
    cudaMemset(D, 0, size_mn);

    uint block_size = min(m, 1024);
    uint grid_size = (m + block_size - 1) / block_size;
    // compute squared euclidean norm of X
    sq_euclid_norm<<<grid_size, block_size>>>(m, k, X, XX);
    block_size = min(n, 1024);
    grid_size = (n + block_size - 1) / block_size;
    sq_euclid_norm<<<block_size, grid_size>>>(n, k, Y, YY);

    // compute (2*X)*YT
    sgemm_cublas(X, Y, XY, m, k, n, 2.0);

    block_size = min(m, 1024);
    grid_size = (m + block_size - 1) / block_size;
    euclid_dist<<<block_size, grid_size>>>(m, n, XX, YY, XY, D);
    cudaFree(XX);
    cudaFree(YY);
    cudaFree(XY);
}

/** Host function to compute all pairwise squared Euclidean distances between
 *  two sets of time series so that we can compute Soft-DTW
 *  on many distance matrices in parallel.
 *  Inputs X, Y, D should be __device__ arrays.
 *  @param X A set of nX vectors of length (row count) m x k (column count)
 *  @param Y A set of nY vectors of length (row count) n x k (column count)
 *  @param D A result array for the distance matrix of dimension (m x n)
 *  @param nX The number of time series in batch X
 *  @param nY The number of time series in batch Y
 *  @param m The length of vectors in X
 *  @param n The length of vectors in Y
 *  @param k The number of vectors in X and Y (columns)
 */
__host__ void sq_euclid_dist_multi(const float *X, const float *Y, float *D,
                                   const uint nX, const uint nY, const uint m,
                                   const uint n, const uint k)
{
    // TODO work in progress, needs testing, probably going to be slow
    // Maybe rather than computing this for all pairs and then softdtw,
    // it would be faster to compute parallelize across the pairs once and
    // compute the distance matrix and softdtw for each pair independently?
    float *XX; // nX x m
    float *YY; // nY x n
    float *XY; // (nX x nY) x m x n
    size_t size_mx = nX * m * sizeof(float);
    size_t size_ny = nY * n * sizeof(float);
    size_t size_mnxy = nX * m * size_ny;
    cudaMalloc(&XX, size_mx);
    cudaMalloc(&YY, size_ny);
    cudaMalloc(&XY, size_mnxy);
    cudaMemset(XX, 0, size_mx);
    cudaMemset(YY, 0, size_ny);
    cudaMemset(XY, 0, size_mnxy);
    cudaMemset(D, 0, size_mnxy);

    uint block_size_m = min(m, 1024);
    uint grid_size_m = (m + block_size_m - 1) / block_size_m;
    uint block_size_n = min(n, 1024);
    uint grid_size_n = (n + block_size_n - 1) / block_size_n;
    // compute squared euclidean norm of X
    // is a loop the best way to do this or can we write one kernel to compute
    // multiple norms in parallel?
    // Need to use cudaStreamCreate to run kernels in the loop in parallel?
    for (uint i = 0; i < m; i++)
    {
        sq_euclid_norm<<<grid_size_m, block_size_m>>>(m, k, &X[i * (m * k)],
                                                      &XX[i * m]);
    }
    for (uint i = 0; i < n; i++)
    {
        sq_euclid_norm<<<block_size_n, grid_size_n>>>(n, k, &Y[i * (n * k)],
                                                      &YY[i * n]);
    }
    cudaDeviceSynchronize();
    const float beta = 0.0;
    const float alpha = 2.0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    // Compute 2*X*Y^T for each X and Y
    for (uint i = 0; i < nX; i++)
    {
        for (uint j = 0; j < nY; j++)
        {
            // call cuBLAS to multiply transposed matrices B^T * A
            // (input is row-major but cublas expects column major)
            cublasSgemm(handle,                    // cublas handle
                        CUBLAS_OP_T,               // transpose first matrix
                        CUBLAS_OP_N,               // tranpose second matrix
                        n,                         // rows in first matrix
                        m,                         // columns in second matrix
                        k,                         // columns in first matrix
                        &alpha,                    // scalar for first matrix
                        &Y[j * (n * k)],           // first matrix
                        k,                         // stride of first matrix
                        &X[i * (m * k)],           // second matrix
                        k,                         // stride of second matrix
                        &beta,                     // scalar for C
                        &XY[(i * nX + j) * m * n], // result matrix
                        n                          // stride of result matrix
            );
            // compute XX + YY - 2XY for each pair of X and Y
            euclid_dist<<<block_size_m, grid_size_m>>>(
                m, n, &XX[i * m], &YY[j * n], &XY[(i * nX + j) * m * n],
                &D[(i * nX + j) * m * n]);
        }
    }
    cublasDestroy(handle);
    cudaFree(XX);
    cudaFree(YY);
    cudaFree(XY);
}