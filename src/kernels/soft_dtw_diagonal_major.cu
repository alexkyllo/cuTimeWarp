#include "helper_functions.cuh"
#include "soft_dtw_diagonal_major.cuh"
#include <stdio.h>
void print_diag(const char *X, const uint m, const uint n)
{
    for (uint k = 0; k < m + n - 1; k++)
    {
        for (uint j = 0; j <= k; j++)
        {
            uint i = k - j;
            if (i < m && j < n)
            {
                // std::cout << X[i * n + j] << " ";
            }
        }
        // std::cout << "\n";
    }
}

__global__ void convert_diagonal(float *D, float *DD, uint m, uint n)
{
    const uint tx = blockIdx.x * blockDim.x + threadIdx.x;
    uint j = tx % n;
    uint i = (tx - j) / n;
    uint dest_i = i + j;
    uint dest_j = j - max(0, (int)dest_i - (int)m + 1);
    DD[dest_i * m + dest_j] = D[i * n + j];
}

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

__host__ void convert_diagonal_major(float *D, float *DD, uint m, uint n)
{
    uint T = m * n;
    uint TPB = min(T, 1024);
    uint B = (T + TPB - 1) / TPB;
    convert_diagonal<<<B, TPB>>>(D, DD, m, n);
    cudaErrchk(cudaDeviceSynchronize());
}

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
__global__ void softdtw_diagonal_kernel(float *D, float *R, float *cost, uint m,
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
