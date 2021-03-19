/** Naive Soft-DTW kernel for one pair of time series
 *  @file soft_dtw_naive.cuh
 *  @author Alex Kyllo
 *  @date 2021-03
 */

#pragma once
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
                                     uint n, float gamma);

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
                                          uint n, float gamma);