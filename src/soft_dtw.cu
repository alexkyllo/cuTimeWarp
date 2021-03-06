/** CUDA implementation of Soft DTW. Host programs for calling the CUDA kernels.
 *  @file soft_dtw.cu
 *  @author Alex Kyllo and Afrooz Rahmati
 *  @date 2021-03
 */
#include "kernels/euclid_dist.cuh"
#include "kernels/helper_functions.cuh"
#include "kernels/soft_dtw_diagonal_major.cuh"
#include "kernels/soft_dtw_naive.cuh"
#include "kernels/soft_dtw_naive_multi.cuh"
#include "kernels/soft_dtw_stencil.cuh"
#include "kernels/soft_dtw_tiled.cuh"
#include "kernels/soft_dtw_tiled_multi.cuh"
#include <cassert>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>

typedef unsigned int uint;

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
    cudaErrchk(cudaMalloc(&XX, size_m));
    cudaErrchk(cudaMalloc(&YY, size_n));
    cudaErrchk(cudaMalloc(&XY, size_mn));
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
    sq_euclid_norm<<<grid_size, block_size>>>(n, k, Y, YY);

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

    cudaErrchk(cudaMalloc(&XX, size_mx));
    cudaErrchk(cudaMalloc(&YY, size_ny));
    cudaErrchk(cudaMalloc(&XY, size_mnxy));
    cudaMemset(XX, 0, size_mx);
    cudaMemset(YY, 0, size_ny);
    cudaMemset(XY, 0, size_mnxy);
    cudaMemset(D, 0, size_mnxy);

    uint block_size_m = min(m, 1024);
    uint grid_size_m = (m + block_size_m - 1) / block_size_m;
    uint block_size_n = min(n, 1024);
    uint grid_size_n = (n + block_size_n - 1) / block_size_n;
    // compute squared euclidean norm of X
    // allocate extra streams to try and get these to run concurrently
    const int num_streams = min(max(nX, nY), 32);
    cudaStream_t streams[num_streams];
    for (uint i = 0; i < num_streams; i++)
        cudaStreamCreate(&streams[i]);
    for (uint i = 0; i < nX; i++)
    {
        uint stream_num = i % num_streams;
        sq_euclid_norm<<<grid_size_m, block_size_m, 0, streams[stream_num]>>>(
            m, k, &X[i * (m * k)], &XX[i * m]);
    }
    for (uint i = 0; i < nY; i++)
    {
        uint stream_num = i % num_streams;
        sq_euclid_norm<<<block_size_n, grid_size_n, 0, streams[stream_num]>>>(
            n, k, &Y[i * (n * k)], &YY[i * n]);
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
            uint stream_num = (i * nY + j) % num_streams;
            cublasSetStream(handle, streams[stream_num]);
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
                        &XY[(i * nY + j) * m * n], // result matrix
                        n                          // stride of result matrix
            );
            // compute XX + YY - 2XY for each pair of X and Y
            euclid_dist<<<block_size_m, grid_size_m, 0, streams[stream_num]>>>(
                m, n, &XX[i * m], &YY[j * n], &XY[(i * nY + j) * m * n],
                &D[(i * nY + j) * m * n]);
        }
    }
    for (uint i = 0; i < num_streams; i++)
        cudaStreamDestroy(streams[i]);
    cublasDestroy(handle);
    cudaFree(XX);
    cudaFree(YY);
    cudaFree(XY);
}

/** Host function for computing Soft DTW on pairwise Euclidean distance matrix
 * for multivariate time series with CUDA.
 * Input D should be a __device__ array.
 * Only a single block is used. m and n must each be no longer than 1024.
 * @param D The pairwise squared Euclidean distance array of two time series
 * @param R An m+2 x n+2 device array that will be filled with alignment values.
 * @param nD The number of distance matrices in D and its leading dimension
 * @param m Length of first time series
 * @param n Length of second time series
 * @param gamma SoftDTW smoothing parameter
 */
__host__ float softdtw_cuda_naive(float *D, float *R, uint m, uint n,
                                  float gamma)
{
    assert(min(m, n) < 1025);
    size_t m2n2 = (m + 2) * (n + 2);
    // Launch a kernel to fill matrix R with infinity
    const int inf_tpb = 256;
    int inf_blocks = (m2n2 + inf_tpb - 1) / m2n2;
    fill_matrix_inf<<<inf_blocks, inf_tpb>>>(
        R, m + 2, n + 2, std::numeric_limits<float>::infinity());

    dim3 B = dim3(1);
    // length of the longest antidiagonal is the smaller of the two dims
    dim3 TPB = dim3(min(m, n));
    float path_cost;
    float *d_path_cost;
    cudaErrchk(cudaMalloc(&d_path_cost, sizeof(float)));
    // Launch the kernel
    softdtw_naive_kernel<<<B, TPB>>>(D, R, d_path_cost, m, n, gamma);
    // Copy the path cost back to host
    cudaMemcpy(&path_cost, d_path_cost, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_path_cost);

    return path_cost;
}

/** Host function for computing SoftDTW gradient by backpropagation
 * @param D The pairwise squared Euclidean distance array of two time series
 * @param R An m+2 x n+2 array of alignment values.
 * @param E An m x n array that will be filled with the gradient values.
 * @param m Length of first time series
 * @param n Length of second time series
 * @param gamma SoftDTW smoothing parameter
 */
__host__ void softdtw_grad_cuda_naive(float *D, float *R, float *E, uint m,
                                      uint n, float gamma)
{
    // Allocate larger temporary device arrays for D and E
    float *D_;
    cudaErrchk(cudaMalloc(&D_, (m + 1) * (n + 1) * sizeof(float)));
    cudaMemset(D_, 0, (m + 1) * (n + 1) * sizeof(float));
    // Copy each row of D to D_
    for (uint i = 0; i < m; i++)
    {
        cudaMemcpy(&D_[i * (n + 1)], &D[i * n], n * sizeof(float),
                   cudaMemcpyDeviceToDevice);
    }
    float *E_;
    cudaErrchk(cudaMalloc(&E_, (m + 2) * (n + 2) * sizeof(float)));
    cudaMemset(E_, 0, (m + 2) * (n + 2) * sizeof(float));

    // D_ is m+1 x n+1
    // R and E_ are m+2 x n+2
    // fill the last row and column of D with 0
    // fill the last row and column of R with -inf
    float neg_inf = -INFINITY;
    for (uint i = 1; i < m + 1; i++)
    {
        cudaMemset(&D_[(i - 1) * (n + 1) + n], 0, sizeof(float));
        cudaMemcpy(&R[i * (n + 2) + n + 1], &neg_inf, sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    for (uint j = 1; j < n + 1; j++)
    {
        cudaMemset(&D_[m * (n + 1) + (j - 1)], 0, sizeof(float));
        cudaMemcpy(&R[(m + 1) * (n + 2) + j], &neg_inf, sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    // Set last element of E to 1
    float one = 1.0;
    cudaMemcpy(&E_[(m + 1) * (n + 2) + n + 1], &one, sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemcpy(&R[(m + 1) * (n + 2) + n + 1], &R[m * (n + 2) + n],
               sizeof(float), cudaMemcpyDeviceToDevice);

    // Set last element of D to 0
    cudaMemset(&D[m * (n + 1) + n], 0, sizeof(float));

    dim3 B = dim3(1);
    dim3 TPB = dim3(min(m, n));
    softdtw_grad_naive_kernel<<<B, TPB>>>(D_, R, E_, m, n, gamma);

    // Copy E_ back to E without the first and last row and column
    for (uint i = 0; i < m; i++)
    {
        cudaMemcpy(&E[i * n], &E_[(i + 1) * (n + 2) + 1], n * sizeof(float),
                   cudaMemcpyDeviceToDevice);
    }
    cudaFree(D_);
    cudaFree(E_);
}

/** Host function for computing soft DTW tiled and shared memory version
 *  we consider the tile with as square, otherwise need to define tile width and
 * height
 * @param da The device array for the first time series data
 * @param db The device array for the second time series data
 * @param D_tiled The device array for the distance matrix
 * @param tile_width the width of our tiled for taking the advantage of shared
 * memory
 * @param total_tiles_waves the total number of tiles
 * @param total_tiles_columns the total number of tiles in one column
 * @param total_tiles_rows the total number of tiles in one row
 * @param min_tiles the minimum number of possible tiles
 *
 */
__host__ void soft_dtw_tiled(float *da, float *db, float *D_, uint tile_width,
                             uint total_tiles_waves, uint total_tiles_columns,
                             uint total_tiles_rows, uint min_tiles, float gamma)
{

    // start wave front process
    // Populate dependency managed by loop
    for (int waveId = 0; waveId < total_tiles_waves; waveId++)
    {
        int wave_Len = waveId + 1;
        if (wave_Len > min_tiles)
            wave_Len = min(min_tiles, total_tiles_waves - waveId);

        // call kernel
        // for none squeare block size, we need to pass the min for
        // threadsPerBlock value
        dim3 blockPerGrid(wave_Len);
        dim3 threadPerBlock(tile_width, tile_width);

        softdtw_global_tiled<<<blockPerGrid, threadPerBlock>>>(
            da, db, D_, waveId, total_tiles_rows, total_tiles_columns,
            tile_width, gamma);

        cudaDeviceSynchronize();
    }
    // copy back data to host
    // cudaMemcpy(D, D_, mn_size, cudaMemcpyDeviceToDevice);
}

/** Host function for computing soft DTW tiled and shared memory version for
 * multivariate we consider the tile with as square, otherwise need to define
 * tile width and height
 * @param da The device array for the first time series data
 * @param db The device array for the second time series data
 * @param D_tiled The device array for the distance matrix
 * @param tile_width the width of our tiled for taking the advantage of shared
 * memory
 * @param tile_height
 * @param total_tiles_waves the total number of tiles
 * @param total_tiles_columns the total number of tiles in one column
 * @param total_tiles_rows the total number of tiles in one row
 * @param min_tiles the minimum number of possible tiles
 * @param nX The number of time series in batch X
 * @param nY The number of time series in batch Y
 *
 */
__host__ void soft_dtw_tiled_multi(float *da, float *db, uint nX, uint nY,
                                   float *D_, uint tile_width, uint tile_height,
                                   uint total_tiles_waves,
                                   uint total_tiles_columns,
                                   uint total_tiles_rows, uint min_tiles,
                                   float gamma, uint m, uint n)
{

    const int num_streams = min(max(nX, nY), 32);
    cudaStream_t streams[num_streams];
    for (uint i = 0; i < num_streams; i++)
        cudaStreamCreate(&streams[i]);

    // compute squared euclidean norm of X
    // allocate extra streams to try and get these to run concurrently

    // start wave front process
    // Populate dependency managed by loop
    for (int waveId = 0; waveId < total_tiles_waves; waveId++)
    {
        int wave_Len = waveId + 1;
        if (wave_Len > min_tiles)
            wave_Len = min(min_tiles, total_tiles_waves - waveId);

        // call kernel
        // for none squeare block size, we need to pass the min for
        // threadsPerBlock value
        dim3 blockPerGrid(wave_Len);
        dim3 threadPerBlock(min(tile_width, tile_height));

        for (uint i = 0; i < nX; i++)
        {
            for (uint j = 0; j < nY; j++)
            {
                uint stream_num = (i * nY + j) % num_streams;
                softdtw_global_tiled_multi<<<blockPerGrid, threadPerBlock, 0,
                                             streams[stream_num]>>>(
                    &da[i * m], &db[j * n], &D_[(i * nY + j) * m * n], waveId,
                    total_tiles_rows, total_tiles_columns, tile_width,
                    tile_height, gamma);
            }
        }
        cudaDeviceSynchronize();

        // call kernel
        // for none squeare block size, we need to pass the min for
        // threadsPerBlock value
        blockPerGrid = dim3(wave_Len);
        threadPerBlock = dim3(min(tile_width, tile_height));

        for (uint i = 0; i < nX; i++)
        {
            for (uint j = 0; j < nY; j++)
            {
                uint stream_num = (i * nY + j) % num_streams;
                softdtw_global_tiled_multi<<<blockPerGrid, threadPerBlock, 0,
                                             streams[stream_num]>>>(
                    &da[i * m], &db[j * n], &D_[(i * nY + j) * m * n], waveId,
                    total_tiles_rows, total_tiles_columns, tile_width,
                    tile_height, gamma);
            }
        }
        cudaDeviceSynchronize();
    }

    for (uint i = 0; i < num_streams; i++)
        cudaStreamDestroy(streams[i]);

    // copy back data to host
    // cudaMemcpy(D, D_, mn_size, cudaMemcpyDeviceToDevice);
}

/** Host function for computing Soft DTW on pairwise Euclidean distance matrix
 * for multivariate time series with CUDA.
 * Input D should be a __device__ array of dimension (nD x m x n).
 * Each threadblock computes DTW for a pair of time series
 * m and n must each be no longer than 1024.
 * @param D The pairwise squared Euclidean distance array of two time series
 * @param R An (nD x (m+2) x (n+2)) device array to fill with alignment values.
 * @param costs A length nD array that will be filled with the pairwise costs
 * @param nD The number of distance matrices in D and its leading dimension
 * @param m Length of first time series
 * @param n Length of second time series
 * @param gamma SoftDTW smoothing parameter
 * @param bandwidth Maximum warping distance from the diagonal to consider for
 * optimal path calculation (Sakoe-Chiba band). Default = 0 = unlimited.
 */
__host__ void softdtw_cuda_naive_multi(float *D, float *R, float *costs,
                                       uint nD, uint m, uint n, float gamma,
                                       uint bandwidth)
{
    assert(min(m, n) < 1025);
    size_t m2n2 = nD * (m + 2) * (n + 2);
    // Launch a kernel to fill matrix R with infinity
    const size_t inf_tpb = 256;
    size_t inf_blocks = (m2n2 + inf_tpb - 1) / inf_tpb;

    fill_matrix_inf<<<inf_blocks, inf_tpb>>>(
        R, (m + 2) * (n + 2), nD, std::numeric_limits<float>::infinity());
    cudaErrchk(cudaDeviceSynchronize());
    dim3 B = dim3(nD);
    uint min_mn = min(m, n);
    uint threads = min_mn;
    assert(threads < 1025);
    dim3 TPB = dim3(threads);
    float *d_path_cost;
    cudaErrchk(cudaMalloc(&d_path_cost, nD * sizeof(float)));
    // Launch the kernel
    softdtw_naive_kernel_multi<<<B, TPB>>>(D, R, d_path_cost, nD, m, n, gamma,
                                           bandwidth);
    // Copy the path cost back to host
    cudaMemcpy(costs, d_path_cost, nD * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_path_cost);
}

/** Host function for computing Soft DTW on pairwise Euclidean distance matrix
 * for multivariate time series with CUDA.
 * Input D should be a __device__ array of dimension (nD x m x n).
 * Each threadblock computes DTW for a pair of time series
 * m and n must each be no longer than 1024.
 * @param D The pairwise squared Euclidean distance array of two time series
 * @param R An (nD x (m+2) x (n+2)) device array to fill with alignment values.
 * @param costs A length nD array that will be filled with the pairwise costs
 * @param nD The number of distance matrices in D and its leading dimension
 * @param m Length of first time series
 * @param n Length of second time series
 * @param gamma SoftDTW smoothing parameter
 * @param bandwidth Maximum warping distance from the diagonal to consider for
 * optimal path calculation (Sakoe-Chiba band). Default = 0 = unlimited.
 */
__host__ void softdtw_cuda_stencil(float *D, float *R, float *costs, uint nD,
                                   uint m, uint n, float gamma, uint bandwidth)
{
    assert(min(m, n) < 1025);
    size_t m2n2 = nD * (m + 2) * (n + 2);
    // Launch a kernel to fill matrix R with infinity
    const int inf_tpb = 256;
    int inf_blocks = (m2n2 + inf_tpb - 1) / m2n2;
    fill_matrix_inf<<<inf_blocks, inf_tpb>>>(
        R, (m + 2) * (n + 2), nD, std::numeric_limits<float>::infinity());

    dim3 B = dim3(nD);
    dim3 TPB = dim3(min(m, n) + 2);
    uint SMEM = nD * (min(m, n) + 2) * 3 * sizeof(float);
    float *d_path_cost;
    cudaErrchk(cudaMalloc(&d_path_cost, nD * sizeof(float)));
    // Launch the kernel
    softdtw_stencil<<<B, TPB, SMEM>>>(D, R, d_path_cost, nD, m, n, gamma,
                                      bandwidth);
    // Copy the path cost back to host
    cudaMemcpy(costs, d_path_cost, nD * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_path_cost);
}

/** Host function for converting a matrix from row major to antidiagonal-major
 * layout.
 *  @param D The input matrix of dimension m x n
 *  @param DD The output matrix of dimension (m+n-1) x min(m,n)
 *  @param m The height of the input matrix (rows)
 *  @param n The width of the input matrix (columns)
 */
__host__ void convert_diagonal_major(float *D, float *DD, uint m, uint n)
{
    uint T = m * n;
    uint TPB = min(T, 1024);
    uint B = (T + TPB - 1) / TPB;
    convert_diagonal<<<B, TPB>>>(D, DD, m, n);
    cudaErrchk(cudaDeviceSynchronize());
}

/** Host function for converting a 3D tensor of m x n matrices from row major to
 * antidiagonal-major layout.
 *  @param D The input matrix of dimension m x n
 *  @param DD The output matrix of dimension (m+n-1) x min(m,n)
 *  @param nD The number of mxn matrices in D
 *  @param m The height of the input matrix (rows)
 *  @param n The width of the input matrix (columns)
 */
__host__ void convert_diagonal_major_multi(float *D, float *DD, uint nD, uint m,
                                           uint n)
{
    uint T = m * n;
    uint TPB = min(T, 1024);
    uint B = ((int)T + (int)TPB - 1) / (int)TPB;

    // run concurrently with streams
    const int num_streams = min((int)nD, 32);
    cudaStream_t streams[num_streams];

    for (uint i = 0; i < num_streams; i++)
        cudaStreamCreate(&streams[i]);

    for (uint i = 0; i < nD; i++)
    {
        uint stream_num = i % num_streams;
        convert_diagonal<<<B, TPB, 0, streams[stream_num]>>>(
            &D[i * m * n], &DD[i * (m + n - 1) * min(m, n)], m, n);
        cudaErrchk(cudaDeviceSynchronize());
    }

    for (uint i = 0; i < num_streams; i++)
        cudaStreamDestroy(streams[i]);
    cudaErrchk(cudaDeviceSynchronize());
}

__host__ float softdtw_cuda_diagonal(float *DD, float *RD, uint m, uint n,
                                     float gamma)
{
    // Launch a kernel to fill matrix R with infinity
    const int inf_tpb = 1024;
    size_t nRD = (std::min(m, n) + 2) * (m + n + 3);
    const uint inf_blocks = (nRD + inf_tpb - 1) / nRD;
    fill_matrix_inf<<<inf_blocks, inf_tpb>>>(
        RD, (min(m, n) + 2) * (m + n + 3), 1,
        std::numeric_limits<float>::infinity());

    dim3 B = dim3(1);
    // length of the longest antidiagonal is the smaller of the two dims
    dim3 TPB = dim3(min(m, n));
    float path_cost = 0;
    float *d_path_cost = 0;
    cudaErrchk(cudaMalloc(&d_path_cost, sizeof(float)));
    // Launch the kernel
    softdtw_diagonal_kernel<<<B, TPB>>>(DD, RD, d_path_cost, m, n, gamma);
    // Copy the path cost back to host
    cudaErrchk(cudaMemcpy(&path_cost, d_path_cost, sizeof(float),
                          cudaMemcpyDeviceToHost));

    return path_cost;
}

__host__ void softdtw_cuda_diagonal_multi(float *DD, float *RD, float *costs,
                                          uint nD, uint m, uint n, float gamma)
{
    assert(min(m, n) < 1025);
    // Launch a kernel to fill matrix R with infinity
    const int inf_tpb = 1024;

    size_t nRD = (std::min(m, n) + 2) * (m + n + 3);
    const uint inf_blocks = (nRD + inf_tpb - 1) / nRD;
    fill_matrix_inf<<<inf_blocks, inf_tpb>>>(
        RD, (min(m, n) + 2) * (m + n + 3), nD,
        std::numeric_limits<float>::infinity());

    dim3 B = dim3(nD);
    // length of the longest antidiagonal is the smaller of the two dims
    dim3 TPB = dim3(min(m, n));
    float *d_path_cost;
    cudaErrchk(cudaMalloc(&d_path_cost, nD * sizeof(float)));
    // Launch the kernel
    softdtw_diagonal_kernel_multi<<<B, TPB>>>(DD, RD, d_path_cost, nD, m, n,
                                              gamma);
    // Copy the path cost back to host
    cudaErrchk(cudaMemcpy(costs, d_path_cost, nD * sizeof(float),
                          cudaMemcpyDeviceToHost));
    cudaFree(d_path_cost);
}
