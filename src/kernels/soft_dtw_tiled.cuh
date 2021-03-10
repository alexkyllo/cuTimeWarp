#pragma once
// /** Kernel function for computing tiled Soft DTW on pairwise Euclidean distance
//  * matrix for multivariate time series with CUDA. Input D should be a
//  * __device__ array.
//  * This naive version only works for sequence length <= 1024.
//  * @param D The pairwise squared Euclidean distance array of two time series
//  * @param R An m+2 x n+2 array that will be filled with the alignments
//  * @param cost The total path cost will be written to this address
//  * @param m Length of first time series
//  * @param n Length of second time series
//  * @param gamma SoftDTW smoothing parameter
//  */
// __global__ void softdtw_tiled_kernel(float *D, float *R, float *cost, uint m,
//                                      uint n, float gamma);

/** Kernel function for computing tiled Soft DTW using wave front approach
 * matrix for multivariate time series with CUDA. Input a , b and D should be a
 * __device__ array.
 * @param a The first time series data 
 * @param b The second time series data 
 * @param m Length of first time series
 * @param n Length of second time series
 * @param D The pairwise distance array of two time series
 * @param waveId the ID for the wave
 * @param total_tiles_rows
 * @param total_tiles_columns
 * @param tile_width
 * @param tileRow
 * @param tileColumn
 * @param gamma SoftDTW smoothing parameter
 */

 __global__ void softdtw_global_tiled(float *a ,float *b ,float *D, int waveId, 
    uint total_tiles_rows, 
    uint total_tiles_columns , uint tile_width);

/** Kernel function for computing tiled Soft DTW using wave front approach
 * matrix for multivariate time series with CUDA. Input a , b and D should be a
 * __device__ array.
 * @param a The first time series data 
 * @param b The second time series data 
 * @param m Length of first time series
 * @param n Length of second time series
 * @param D The pairwise distance array of two time series
 * @param waveId the ID for the wave
 * @param total_tiles_rows
 * @param total_tiles_columns
 * @param tile_width
 * @param tileRow
 * @param tileColumn
 * @param gamma SoftDTW smoothing parameter
 */

 __global__ void softdtw_tiled_wavefront(float *a ,float *b ,float *D, int waveId, 
    uint total_tiles_rows, 
    uint total_tiles_columns , uint tile_width , uint tileRow , uint  tileColumn);   