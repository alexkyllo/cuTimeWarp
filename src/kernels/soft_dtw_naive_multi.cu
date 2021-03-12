#include "helper_functions.cuh"
#include "soft_dtw_naive_multi.cuh"
#include <stdio.h>
/** Kernel function for computing "naive" Soft DTW on pairwise Euclidean
 * distance matrix for multivariate time series with CUDA.
 * Input D should be a __device__ array.
 * This naive version only works for sequence lengths <= 1024 i.e. can fit in
 * a single threadblock.
 * Each threadblock computes DTW for a pair of time series
 * Each thread can process one anti-diagonal.
 * @param D A 3D tensor of pairwise squared Euclidean distance matrices
 * between time series
 * @param R An m+2 x n+2 array that will be filled with the alignments
 * @param cost The total path costs will be written to this array of length nD
 * @param nD The number of distance matrices in D and its leading dimension
 * @param m Length of first time series
 * @param n Length of second time series
 * @param gamma SoftDTW smoothing parameter
 * @param bandwidth Maximum warping distance from the diagonal to consider for
 * optimal path calculation (Sakoe-Chiba band). Default = 0 = unlimited.
 */
__global__ void softdtw_naive_kernel_multi(float *D, float *R, float *cost,
                                           uint nD, uint m, uint n, float gamma,
                                           uint bandwidth)
{
    const uint tx = threadIdx.x;
    const uint bx = blockIdx.x;
    uint bD = bx * m * n;
    uint bD2 = bx * (m + 2) * (n + 2);

    // block size = max(m, n) (length of longest diagonal)
    // number of antidiagonals is 2 * max(m,n) - 1
    const uint passes = 2 * blockDim.x - 2;

    for (uint p = 0; p < passes; p++)
    {
        uint jj = max(0, min(p - tx, n - 1));
        uint i = tx + 1;
        uint j = jj + 1;
        int width = abs((int)m - (int)n) + bandwidth;
        int lower = max(1, i - bandwidth);
        int upper = min(max((int)m, (int)n), (int)i + width) + 1;
        bool is_in_lower = m > n ? i >= lower : j >= lower;
        bool is_in_upper = m > n ? i < upper : j < upper;
        bool is_in_band = bandwidth == 0 || (is_in_lower && is_in_upper);
        if (tx + jj == p && is_in_band && tx < m && jj < n)
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

// TODO: write a backward gradient kernel for multi time series
