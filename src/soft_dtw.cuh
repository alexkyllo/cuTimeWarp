/** CUDA implementation of Soft DTW. Host programs for calling the CUDA kernels.
 *  @file soft_dtw.cuh
 *  @author Alex Kyllo and Afrooz Rahmati
 *  @date 2021-03
 */
#pragma once
#ifndef __CUDACC__
#define __device__
#define __host__
#endif

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
                           const float alpha);

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
                             const uint m, const uint n, const uint k);

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
                                   const uint n, const uint k);

__host__ float softdtw_cuda_naive(float *D, float *R, uint m, uint n,
                                  float gamma);
/** Host function for computing SoftDTW gradient by backpropagation
 * @param D The pairwise squared Euclidean distance array of two time series
 * @param R An m+2 x n+2 array of alignment values.
 * @param E An m x n array that will be filled with the gradient values.
 * @param m Length of first time series
 * @param n Length of second time series
 * @param gamma SoftDTW smoothing parameter
 */
__host__ void softdtw_grad_cuda_naive(float *D, float *R, float *E, uint m,
                                      uint n, float gamma);
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
                                       uint bandwidth = 0);

__host__ void soft_dtw_tiled(float *da, float *db, float *D_, uint tile_width,
                             uint total_tiles_waves, uint total_tiles_columns,
                             uint total_tiles_rows, uint min_tiles,
                             float gamma);

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
 *
 */
__host__ void soft_dtw_tiled_multi(float *da, float *db, uint nX, uint nY,
                                   float *D_, uint tile_width, uint tile_height,
                                   uint total_tiles_waves,
                                   uint total_tiles_columns,
                                   uint total_tiles_rows, uint min_tiles,
                                   float gamma, uint m, uint n);

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
                                   uint m, uint n, float gamma,
                                   uint bandwidth = 0);

/** Host function for converting a matrix from row major to antidiagonal-major
 * layout.
 *  @param D The input matrix of dimension m x n
 *  @param DD The output matrix of dimension (m+n-1) x min(m,n)
 *  @param m The height of the input matrix (rows)
 *  @param n The width of the input matrix (columns)
 */
__host__ void convert_diagonal_major(float *D, float *DD, uint m, uint n);

/** Host function for converting a 3D tensor of m x n matrices from row major to
 * antidiagonal-major layout.
 *  @param D The input matrix of dimension m x n
 *  @param DD The output matrix of dimension (m+n-1) x min(m,n)
 *  @param nD The number of mxn matrices in D
 *  @param m The height of the input matrix (rows)
 *  @param n The width of the input matrix (columns)
 */
__host__ void convert_diagonal_major_multi(float *D, float *DD, uint nD, uint m,
                                           uint n);

__host__ float softdtw_cuda_diagonal(float *DD, float *RD, uint m, uint n,
                                     float gamma);

__host__ void softdtw_cuda_diagonal_multi(float *DD, float *RD, float *costs,
                                          uint nD, uint m, uint n, float gamma);
