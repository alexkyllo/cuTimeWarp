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
                std::cout << X[i * n + j] << " ";
            }
        }
        std::cout << "\n";
    }
}

__global__ void convert_diagonal(float *D, float *DD, uint m, uint n)
{
    const uint tx = blockIdx.x * blockDim.x + threadIdx.x;
    uint j = tx % n;
    uint i = (tx - j) / n;
    // new i is the antidiagonal ordinal, sum of i and j
    uint dest_i = i + j;
    // new j = j if in upper left half, else j-dist from leading antidiagonal
    uint dest_j = j - max(0, (int)dest_i - (int)m + 1);
    DD[dest_i * m + dest_j] = D[i * n + j];
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
 * This version assumes D is a diagonal-major array where m and n are the
 * dimensions of the original row-major array. m x n becomes (m+n-1) x min(m,n).
 * It also assumes R is a diagonal-major array where (m+2) and (n+2) are the
 * dimensions of the original row-major array.
 * (m+2) x (n+2) becomes (m+n+3) x min(m+2,n+2)
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
__global__ void _softdtw_diagonal_kernel(float *D, float *R, float *cost,
                                         uint m, uint n, float gamma)
{
    const uint tx = threadIdx.x;
    const uint bx = blockDim.x;
    // block size = min(m, n) (length of longest diagonal)
    // number of antidiagonals is m+n-1
    const uint passes = m + n - 1;

    for (uint p = 0; p < passes; p++)
    {
        // uint jj = max(0, min(p - tx, n - 1));
        // uint dest_i = tx + jj;
        // uint dest_j = jj - max(0, (int)dest_i - (int)(m + 3));
        // uint i = dest_i + 1;
        // uint j = dest_j + 1;

        uint jj = max(0, min(p - tx, n - 1));
        uint old_i = tx + 1;
        uint old_j = jj + 1;

        uint i = old_i + old_j;
        uint j = old_j - max(0, (int)i - (int)(m + 3));

        if (tx + jj == p && (tx < m && jj < n))
        {
            float cost = D[(i - 2) * bx + j - 1];     // 1,0
            float r1 = R[(i - 1) * (bx + 2) + j];     // 1,1
            float r2 = R[(i - 2) * (bx + 2) + j - 1]; // 2, 0
            float r3 = R[(i - 1) * (bx + 2) + j - 1];
            double prev_min = softmin(r1, r2, r3, gamma);
            R[i * (bx + 2) + j] = cost + prev_min;
            if (tx == 0)
            {
                printf("pass %d tid %d reading %.2f from D[%d, %d]\n", p, tx,
                       cost, i - 2, j - 1);
                printf("pass %d tid %d reading %.2f from R[%d, %d]\n", p, tx,
                       r1, i - 1, j);
                printf("pass %d tid %d reading %.2f from R[%d, %d]\n", p, tx,
                       r2, i - 2, j - 1);
                printf("pass %d tid %d reading %.2f from R[%d, %d]\n", p, tx,
                       r3, i - 1, j - 1);
                printf(
                    "pass %d tx %d jj %d i %d j %d writing %.2f to R[%d, %d]\n",
                    p, tx, jj, i, j, cost + prev_min, i, j);
            }
        }
        __syncthreads();
    }
    if (tx == 0)
    {
        *cost = R[m * (bx + 2) + n];
    }
}

__global__ void softdtw_diagonal_kernel(float *D, float *R, float *cost, uint m,
                                        uint n, float gamma)
{
    const uint tx = threadIdx.x;
    const uint bx = blockDim.x;
    // block size = min(m, n) (length of longest diagonal)
    // number of antidiagonals is m+n-1
    // D is now (m+n-1) x min(m,n)
    // R is now (m+n+3) x min(m+1,n+1)
    const uint passes = m + n - 1;

    for (uint p = 0; p < passes; p++)
    {
        uint ii = max(0, (int)p - (int)tx);
        uint past_mid = max(0, (int)p - (int)bx + 1);
        uint i = ii + 1 - past_mid;
        uint j = tx + 1 + past_mid;

        if (tx + ii <= p && j <= n)
        {
            // convert i,j to diagonal-major coordinates
            // new j = j if in upper left half, else j-dist from leading
            // antidiagonal
            uint di = (i - 1) + (j - 1);
            uint dj = j - 1 - past_mid;
            uint ri = i + j;
            uint rj = j - past_mid;
            uint r1j = rj - 1;
            uint r2j = rj - 1;
            uint r3j = rj;

            // If we are past the antidiagonal, need to increment the previous
            // cell references
            if (p >= bx)
            {
                r1j++;
                r2j++;
                r3j++;
            }
            if (p > bx)
            {
                r1j++;
            }

            float cost = D[di * bx + dj];
            float r1 = R[di * (bx + 2) + r1j];
            float r2 = R[(ri - 1) * (bx + 2) + r2j];
            float r3 = R[(ri - 1) * (bx + 2) + r3j];
            double prev_min = softmin(r1, r2, r3, gamma);
            R[ri * (bx + 2) + rj] = cost + prev_min;
        }
        __syncthreads();
        if (tx == 0)
        {
            *cost = R[(m + n) * (bx + 2) + 1];
        }
    }
}