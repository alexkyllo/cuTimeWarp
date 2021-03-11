#include "helper_functions.cuh"
#include "soft_dtw_stencil.cuh"
#include <stdio.h>

/** Kernel function for computing Soft DTW on pairwise Euclidean
 * distance matrix for multivariate time series with CUDA.
 * Uses a shared memory stencil for caching the previous diagonal
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
 */
__global__ void softdtw_stencil(float *D, float *R, float *cost, uint nD,
                                uint m, uint n, float gamma)
{
    // dynamic shared memory diagonal buffer array for caching the previous
    // diagonals.
    // length is (max(m,n) + 2) * 3 because it needs to store three
    // diagonals of R and the longest diagonal is (max(m,n)+2)
    // there should be max(m+2,n+2) threads
    extern __shared__ float stencil[];
    const uint tx = threadIdx.x;
    const uint bx = blockIdx.x;
    uint bD = bx * m * n;
    uint bD2 = bx * (m + 2) * (n + 2);

    // block size = max(m+2, n+2) (length of longest diagonal of R)
    // number of antidiagonals is 2 * max(m,n) - 1
    const uint passes = 2 * max(m, n); // 2 * blockDim.x - 1;

    // each pass is one diagonal of the distance matrix
    for (uint p = 0; p < passes; p++)
    {
        uint pp = p;
        uint jj = max(0, min(pp - tx, n + 1));
        uint i = tx + 1;
        uint j = jj + 1;

        // calculate index offsets into the shared memory array for each
        // diagonal, using mod to rotate them.
        uint cur_idx = (pp + 2) % 3 * (blockDim.x);
        uint prev_idx = (pp + 1) % 3 * (blockDim.x);
        uint prev2_idx = pp % 3 * (blockDim.x);
        bool is_wave = tx + jj == pp && tx < m + 2 && jj < n + 2;
        if (is_wave)
        {
            // load a diagonal into shared memory
            if (p == 0 && tx == 0)
            {
                stencil[prev2_idx] = 0;
            }
            stencil[prev2_idx + jj] = R[bD2 + tx * (n + 2) + jj];
        }
        // synchronize to make sure shared mem is done loading
        __syncthreads();
        // check if this thread is on the current diagonal and in-bounds
        pp = p - 2;
        jj = max(0, min(pp - tx, n));
        i = tx + 1;
        j = jj + 1;
        cur_idx = (pp + 2) % 3 * (blockDim.x);
        prev_idx = (pp + 1) % 3 * (blockDim.x);
        prev2_idx = pp % 3 * (blockDim.x);
        is_wave = tx + jj == pp && (tx < m + 1 && jj < n + 1);
        if (is_wave)
        {
            float c = D[bD + (i - 1) * n + j - 1];
            // read the elements of R from the stencil
            float r1 = stencil[prev_idx + i];
            float r2 = stencil[prev_idx + i - 1];
            float r3 = stencil[prev2_idx + i - 1];
            double prev_min = softmin(r1, r2, r3, gamma);
            // write the current element of R back to the stencil
            stencil[cur_idx + i] = c + prev_min;
        }
        // make sure the diagonal is finished before proceeding to the next
        __syncthreads();

        // after a diagonal is no longer used, write that portion of R in
        // shared memory back to global memory
        if (is_wave)
        {
            R[bD2 + tx * (n + 2) + jj] = stencil[prev2_idx + tx];
        }
        // R[m,n] is the best path total cost, the last thread should
        // write this from the stencil back to the cost array in global memory
        if (p == passes - 1 && tx + jj == pp && tx < m + 1 && jj < n + 1)
        {
            cost[bx] = stencil[prev2_idx + tx];
        }
    }
}
