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

/** Kernel function for converting a matrix from row major to antidiagonal-major
 * layout.
 *  @param D The input matrix of dimension m x n
 *  @param DD The output matrix of dimension (m+n-1) x min(m,n)
 *  @param m The height of the input matrix (rows)
 *  @param n The width of the input matrix (columns)
 */
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
    uint B = (T + TPB - 1) / TPB;

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
    }

    for (uint i = 0; i < num_streams; i++)
        cudaStreamDestroy(streams[i]);
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
__global__ void softdtw_diagonal_kernel(float *D, float *R, float *cost, uint m,
                                        uint n, float gamma)
{
    const uint tx = threadIdx.x;
    const uint bd = blockDim.x;
    // block size = min(m, n) (length of longest diagonal)
    // number of antidiagonals is m+n-1
    // D is now (m+n-1) x min(m,n)
    // R is now (m+n+3) x min(m+1,n+1)
    const uint passes = m + n - 1;

    for (uint p = 0; p < passes; p++)
    {
        uint ii = max(0, (int)p - (int)tx);
        uint past_mid = max(0, (int)p - (int)bd + 1);
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
            if (p >= bd)
            {
                r1j++;
                r2j++;
                r3j++;
            }
            if (p > bd)
            {
                r1j++;
            }

            float cost = D[di * bd + dj];
            float r1 = R[di * (bd + 2) + r1j];
            float r2 = R[(ri - 1) * (bd + 2) + r2j];
            float r3 = R[(ri - 1) * (bd + 2) + r3j];
            double prev_min = softmin(r1, r2, r3, gamma);
            R[ri * (bd + 2) + rj] = cost + prev_min;
        }
        __syncthreads();
    }
    if (tx == 0)
    {
        *cost = R[(m + n) * (bd + 2) + 1];
    }
}

__global__ void softdtw_diagonal_kernel_multi(float *D, float *R, float *cost,
                                              uint nD, uint m, uint n,
                                              float gamma)
{
    const uint tx = threadIdx.x;
    const uint bd = blockDim.x;
    const uint bx = blockIdx.x;
    const uint bD = bx * m * n;
    const uint bD2 = bx * (m + 2) * (n + 2);

    // block size = min(m, n) (length of longest diagonal)
    // number of antidiagonals is m+n-1

    // D is now (m+n-1) x min(m,n)
    // R is now (m+n+3) x min(m+1,n+1)
    const uint passes = m + n - 1;

    for (uint p = 0; p < passes; p++)
    {
        uint ii = max(0, (int)p - (int)tx);
        uint past_mid = max(0, (int)p - (int)bd + 1);
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
            if (p >= bd)
            {
                r1j++;
                r2j++;
                r3j++;
            }
            if (p > bd)
            {
                r1j++;
            }

            float cost = D[bD + di * bd + dj];
            float r1 = R[bD2 + di * (bd + 2) + r1j];
            float r2 = R[bD2 + (ri - 1) * (bd + 2) + r2j];
            float r3 = R[bD2 + (ri - 1) * (bd + 2) + r3j];
            double prev_min = softmin(r1, r2, r3, gamma);
            R[bD2 + ri * (bd + 2) + rj] = cost + prev_min;
        }
        __syncthreads();
    }
    if (tx == 0)
    {
        cost[bx] = R[bD2 + (m + n) * (bd + 2) + 1];
    }
}
