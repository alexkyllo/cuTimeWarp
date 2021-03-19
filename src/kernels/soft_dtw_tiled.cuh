/** Wavefront Tiled implementation of Soft-DTW
 * @file soft_dtw_tiled.cuh
 * @author Afrooz Rahmati
 * @date 2021-03
 */
#pragma once

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

__global__ void softdtw_global_tiled(float *a, float *b, float *D, int waveId,
                                     uint total_tiles_rows,
                                     uint total_tiles_columns, uint tile_width,
                                     float gamma);

/** Kernel function for computing tiled Soft DTW using wave front approach
 * matrix for multivariate time series with CUDA. Input a , b and D should be a
 * __device__ array.
 * @param a The first time series data
 * @param b The second time series data
 * @param m Length of first time series
 * @param n Length of second time series
 * @param D The pairwise distance array of two time series
 * @param total_tiles_rows
 * @param total_tiles_columns
 * @param tile_width
 * @param tileRow
 * @param tileColumn
 * @param gamma SoftDTW smoothing parameter
 */

__device__ void softdtw_tiled_wavefront(float *a, float *b, float *D,
                                        uint total_tiles_rows,
                                        uint total_tiles_columns,
                                        uint tile_width, uint tileRow,
                                        uint tileColumn, float gamma);
