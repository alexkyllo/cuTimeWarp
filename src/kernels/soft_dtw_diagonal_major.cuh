/** Diagonal-major layout Soft-DTW kernels
 *  @file soft_dtw_diagonal_major.cuh
 *  @author Alex Kyllo
 *  @date 2021-03-14
 */
#pragma once
#ifndef __CUDACC__
#define __device__
#define __host__
#define __global__
#endif

typedef unsigned int uint;

/** Kernel function for converting a matrix from row major to antidiagonal-major
 * layout.
 *  @param D The input matrix of dimension m x n
 *  @param DD The output matrix of dimension (m+n-1) x min(m,n)
 *  @param m The height of the input matrix (rows)
 *  @param n The width of the input matrix (columns)
 */
__global__ void convert_diagonal(float *D, float *DD, uint m, uint n);

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
                                        uint n, float gamma);

/** Kernel function for computing "naive" Soft DTW on pairwise Euclidean
 * distance matrix for multiple distance matrices of multivariate time series
 * with CUDA. Input D should be a __device__ array.
 * This version assumes D is an array of diagonal-major arrays where m and n are
 * the dimensions of the original row-major array. m x n becomes (m+n-1) x
 * min(m,n). It also assumes R is a diagonal-major array where (m+2) and (n+2)
 * are the dimensions of the original row-major array. (m+2) x (n+2) becomes
 * (m+n+3) x min(m+2,n+2) This naive version only works for sequence lengths <=
 * 1024 i.e. can fit in a single threadblock. Assumes only a single threadblock
 * in the kernel launch. Each thread can process one anti-diagonal.
 * @param D The pairwise squared Euclidean distance array of two time series
 * @param R An m+2 x n+2 array that will be filled with the alignments
 * @param cost The total path cost will be written to this address
 * @param nD The number of distance matrices in D
 * @param m Length of first time series
 * @param n Length of second time series
 * @param gamma SoftDTW smoothing parameter
 */
__global__ void softdtw_diagonal_kernel_multi(float *D, float *R, float *cost,
                                              uint nD, uint m, uint n,
                                              float gamma);
