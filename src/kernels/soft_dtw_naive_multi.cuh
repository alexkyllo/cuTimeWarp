#pragma once
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
                                           uint bandwidth = 0);
