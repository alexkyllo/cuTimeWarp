#include "helper_functions.cuh"
#include "soft_dtw_naive_multi.cuh"

/** Kernel function for computing "naive" Soft DTW on pairwise Euclidean
 * distance matrix for multivariate time series with CUDA.
 * Input D should be a __device__ array.
 * This naive version only works for sequence lengths <= 1024 i.e. can fit in
 * a single threadblock.
 * Each threadblock computes DTW for a pair of time series
 * Each thread can process one anti-diagonal.
 * @param D The pairwise squared Euclidean distance array of two time series
 * @param R An m+2 x n+2 array that will be filled with the alignments
 * @param cost The total path costs will be written to this array of length nD
 * @param nD The number of distance matrices in D and its leading dimension
 * @param m Length of first time series
 * @param n Length of second time series
 * @param gamma SoftDTW smoothing parameter
 */
__global__ void softdtw_naive_kernel_multi(float *D, float *R, float *cost,
                                           uint nD, uint m, uint n, float gamma)
{
    const uint tx = threadIdx.x;
    const uint bx = blockIdx.x;
    uint bD = bx * m * n;
    uint bD2 = bx * (m + 2) * (n + 2);

    // block size = max(m, n) (length of longest diagonal)
    // number of antidiagonals is 2 * max(m,n) - 1
    const uint passes = 2 * blockDim.x - 1;

    for (uint p = 0; p < passes; p++)
    {
        uint jj = max(0, min(p - tx, n - 1));
        uint i = tx + 1;
        uint j = jj + 1;

        if (tx + jj == p && (tx < m && jj < n))
        {
            float c = D[bD + (i - 1) * n + j - 1];
            float r1 = R[bD2 + (i - 1) * (n + 2) + j];
            float r2 = R[bD2 + i * (n + 2) + j - 1];
            float r3 = R[bD2 + (i - 1) * (n + 2) + j - 1];
            double prev_min = softmin(r1, r2, r3, gamma);
            R[bD2 + i * (n + 2) + j] = c + prev_min;
        }
        __syncthreads();
    }
    if (tx == 0)
    {
        cost[bx] = R[bD2 + m * (n + 2) + n];
    }
}