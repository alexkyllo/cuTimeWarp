#include "soft_dtw_tiled.cuh"
/** Kernel function for computing tiled Soft DTW on pairwise Euclidean distance
 * matrix for multivariate time series with CUDA. Input D should be a
 * __device__ array.
 * This naive version only works for sequence length <= 1024.
 * @param D The pairwise squared Euclidean distance array of two time series
 * @param R An m+2 x n+2 array that will be filled with the alignments
 * @param cost The total path cost will be written to this address
 * @param m Length of first time series
 * @param n Length of second time series
 * @param gamma SoftDTW smoothing parameter
 */
__global__ void softdtw_tiled_kernel(float *D, float *R, float *cost, uint m,
                                     uint n, float gamma)
{
    // TODO
    // Divide R into tiles
    // Each tile depends on the tiles to its top, left, and top-left
    // Assign one thread to spin on the signal variable for this tile
    // Process the tile diagonally from upper left to lower right
    // using a loop counter to keep track of fully processed diagonals
    // and while loop and syncthreads to spin on it
    // Write to the signal variables to signal the next tiles
}