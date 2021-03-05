/** CUDA implementation of Soft DTW
 *  @file soft_dtw.cu
 */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>

typedef unsigned int uint;

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

/** CUDA kernel to compute the euclidean distance between two sets of vectors
 *  X and Y by using the euclidian norms, i.e. X*X + Y*Y - 2X*Y
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

/** Host function for retrieving the number of SMs on the GPU device
 *  Useful for limiting the # of threadblocks to the # of SMs in a kernel launch
 *  @param device_num The device number, default 0
 *  @return the SM count
 */
__host__ uint get_device_sm_count(uint device_num = 0)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_num);
    return deviceProp.multiProcessorCount;
}

/** Kernel function for computing "naive" Soft DTW on pairwise Euclidean
 * distance matrix for multivariate time series with CUDA. Input D should be a
 * __device__ array.
 * This naive version only works for sequence lengths <= 1024 i.e. can fit in
 * a single threadblock.
 * Assumes omly a single threadblock in the kernel launch.
 * Each thread can process one anti-diagonal.
 * @param D The pairwise squared Euclidean distance array of two time series
 * @param R An m+2 x n+2 array that will be filled with the alignments
 * @param cost The total path cost will be written to this address
 * @param m Length of first time series
 * @param n Length of second time series
 * @param gamma SoftDTW smoothing parameter
 */
__global__ void softdtw_naive_kernel(float *D, float *R, float *cost, uint m,
                                     uint n, float gamma)
{
    const uint ii = threadIdx.x;
    // block size = max(m, n) (length of longest diagonal)
    const uint bx = blockDim.x;
    // number of antidiagonals is 2 * max(m,n) - 1
    const uint passes = 2 * bx - 1;

    for (uint p = 0; p < passes; p++)
    {
        uint jj = max(0, min(p - ii, n - 1));
        uint i = ii + 1;
        uint j = jj + 1;

        if (ii + jj == p && (ii < m && jj < n))
        {
            float cost = D[(i - 1) * n + j - 1];
            float r1 = R[(i - 1) * (n + 2) + j];
            float r2 = R[i * (n + 2) + j - 1];
            float r3 = R[(i - 1) * (n + 2) + j - 1];
            double prev_min = softmin(r1, r2, r3, gamma);
            R[i * (n + 2) + j] = cost + prev_min;
        }
        __syncthreads();
    }
    if (ii == 0)
    {
        *cost = R[m * (n + 2) + n];
    }
}

/** Kernel function for computing tiled Soft DTW on pairwise Euclidean distance
 * matrix for multivariate time series with CUDA. Input D should be a
 * __device__ array.
 * This naive version only works for sequence length <= 1024.
 * @param D The pairwise squared Euclidean distance array of two time series
 * @param R An m+2 x n+2 array that will be filled with the alignments
 * @param cost The total path cost will be written to this address
 * @param m Length of first time series
 * @param n Length of second time series
 * @param gamma SoftDTW smoothing parameter
 */
__global__ void softdtw_tiled_kernel(float *D, float *R, float *cost, uint m,
                                     uint n, float gamma)
{
    // TODO
    // Divide R into tiles
    // Each tile depends on the tiles to its top, left, and top-left
    // Assign one thread to spin on the signal variable for this tile
    // Process the tile diagonally from upper left to lower right
    // using a loop counter to keep track of fully processed diagonals
    // and while loop and syncthreads to spin on it
    // Write to the signal variables to signal the next tiles
}

/** Kernel to fill a matrix with infinity except for index 0 = 0.0
 *  to initialize the DTW cost matrix
 */
__global__ void fill_matrix_inf(float *A, int width, int height, float val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < width * height; i += gridDim.x * blockDim.x)
    {
        A[i] = val;
    }
    if (idx == 0)
    {
        A[0] = 0.0;
    }
}

/** Host function for computing Soft DTW on pairwise Euclidean distance matrix
 * for multivariate time series with CUDA.
 * Input D should be a __device__ array.
 * Only a single block is used. m and n must each be no longer than 1024.
 * @param D The pairwise squared Euclidean distance array of two time series
 * @param m Length of first time series
 * @param n Length of second time series
 * @param gamma SoftDTW smoothing parameter
 */
__host__ float softdtw_cuda_naive(float *D, uint m, uint n, float gamma)
{
    float *R;
    size_t m2n2 = (m + 2) * (n + 2);
    size_t sz_R = m2n2 * sizeof(float);
    cudaMalloc(&R, sz_R);
    // Launch a kernel to fill matrix R with infinity
    const int inf_tpb = 256;
    int inf_blocks = (m2n2 + inf_tpb - 1) / m2n2;
    fill_matrix_inf<<<inf_blocks, inf_tpb>>>(R, m + 2, n + 2, std::numeric_limits<float>::infinity());

    dim3 B = dim3(1);
    dim3 TPB = dim3(max(m, n));
    float path_cost;
    float *d_path_cost;
    cudaMalloc(&d_path_cost, sizeof(float));
    // Launch the kernel
    softdtw_naive_kernel<<<B, TPB>>>(D, R, d_path_cost, m, n, gamma);
    // Copy the path cost back to host
    cudaMemcpy(&path_cost, d_path_cost, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_path_cost);
    cudaFree(R);
    
    return path_cost;
}

// TODO: Soft DTW gradient

// TODO: Barycenter computation (average time series under SoftDTW geometry)
// through gradient descent with SoftDTW as loss function

// TODO: 1-nearest neighbor classification function