/** CUDA implementation of Soft DTW
 *  @file soft_dtw.cu
 */
#include "kernels/euclid_dist.cuh"
#include "kernels/helper_functions.cuh"
#include "kernels/soft_dtw_naive.cuh"
#include "kernels/soft_dtw_naive_multi.cuh"
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>

typedef unsigned int uint;

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
    size_t m2n2 = (m + 2) * (n + 2);
    // Launch a kernel to fill matrix R with infinity
    const int inf_tpb = 256;
    int inf_blocks = (m2n2 + inf_tpb - 1) / m2n2;
    fill_matrix_inf<<<inf_blocks, inf_tpb>>>(
        R, m + 2, n + 2, std::numeric_limits<float>::infinity());

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

    return path_cost;
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
 */
__host__ void softdtw_cuda_naive_multi(float *D, float *R, float *costs,
                                       uint nD, uint m, uint n, float gamma)
{
    size_t m2n2 = nD * (m + 2) * (n + 2);
    // Launch a kernel to fill matrix R with infinity
    const int inf_tpb = 256;
    int inf_blocks = (m2n2 + inf_tpb - 1) / m2n2;
    fill_matrix_inf<<<inf_blocks, inf_tpb>>>(
        R, (m + 2) * (n + 2), nD, std::numeric_limits<float>::infinity());

    dim3 B = dim3(nD);
    dim3 TPB = dim3(max(m, n));
    float *d_path_cost;
    cudaMalloc(&d_path_cost, nD * sizeof(float));
    // Launch the kernel
    softdtw_naive_kernel_multi<<<B, TPB>>>(D, R, d_path_cost, nD, m, n, gamma);
    // Copy the path cost back to host
    cudaMemcpy(costs, d_path_cost, nD * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_path_cost);
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
    cudaMalloc(&D_, (m + 1) * (n + 1) * sizeof(float));
    cudaMemset(D_, 0, (m + 1) * (n + 1) * sizeof(float));
    // Copy each row of D to D_
    for (uint i = 0; i < m; i++)
    {
        cudaMemcpy(&D_[i * (n + 1)], &D[i * n], n * sizeof(float),
                   cudaMemcpyDeviceToDevice);
    }
    float *E_;
    cudaMalloc(&E_, (m + 2) * (n + 2) * sizeof(float));
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
    dim3 TPB = dim3(max(m, n));
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

// TODO: Barycenter computation (average time series under SoftDTW geometry)
// through gradient descent with SoftDTW as loss function
// TODO: 1-nearest neighbor classification function
