#include "helper_functions.cuh"
#include "soft_dtw_naive.cuh"

/** Kernel function for computing "naive" Soft DTW on pairwise Euclidean
 * distance matrix for multivariate time series with CUDA. Input D should be a
 * __device__ array.
 * This naive version only works for sequence lengths <= 1024 i.e. can fit in
 * a single threadblock.
 * Assumes only a single threadblock in the kernel launch.
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
    const uint tx = threadIdx.x;
    // block size = max(m, n) (length of longest diagonal)
    // number of antidiagonals is 2 * max(m,n) - 1
    const uint passes = 2 * blockDim.x - 2;

    for (uint p = 0; p < passes; p++)
    {
        uint jj = max(0, min(p - tx, n - 1));
        uint i = tx + 1;
        uint j = jj + 1;

        if (tx + jj == p && (tx < m && jj < n))
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
    if (tx == 0)
    {
        *cost = R[m * (n + 2) + n];
    }
}

/** Kernel function for computing "naive" SoftDTW gradient by backpropagation
 * This naive version only works for sequence lengths <= 1024 i.e. can fit in
 * a single threadblock.
 * Assumes only a single threadblock in the kernel launch.
 * Each thread can process one anti-diagonal.
 * @param D The pairwise squared Euclidean distance array of two time series
 * @param R An m+2 x n+2 array that will be filled with the alignment
 * values.
 * @param E An m+2 x n+2 array that will be filled with the gradient values.
 * @param m Length of first time series
 * @param n Length of second time series
 * @param gamma SoftDTW smoothing parameter
 */
__global__ void softdtw_grad_naive_kernel(float *D, float *R, float *E, uint m,
                                          uint n, float gamma)
{
    const uint tx = threadIdx.x;
    const uint passes = 2 * blockDim.x - 1;

    for (uint p = 0; p < passes; p++)
    {
        uint backward_p = passes - p - 1;
        uint jj = max(0, min(backward_p - tx, n - 1));
        uint i = tx + 1;
        uint j = jj + 1;
        if (tx + jj == backward_p && (tx < m && jj < n))
        {
            if (isinf(R[i * (n + 2) + j]))
            {
                R[i * (n + 2) + j] = -INFINITY;
            }
            float r0 = R[i * (n + 2) + j];
            float a =
                exp((R[(i + 1) * (n + 2) + j] - r0 - D[i * (n + 1) + (j - 1)]) /
                    gamma);
            float b =
                exp((R[i * (n + 2) + (j + 1)] - r0 - D[(i - 1) * (n + 1) + j]) /
                    gamma);
            float c =
                exp((R[(i + 1) * (n + 2) + j + 1] - r0 - D[i * (n + 1) + j]) /
                    gamma);
            E[i * (n + 2) + j] = E[(i + 1) * (n + 2) + j] * a +
                                 E[i * (n + 2) + j + 1] * b +
                                 E[(i + 1) * (n + 2) + j + 1] * c;
        }
        __syncthreads();
    }
}