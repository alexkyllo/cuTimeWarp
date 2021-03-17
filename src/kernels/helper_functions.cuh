#pragma once
#ifndef __CUDACC__
#define __device__
#define __host__
#define __global__
#endif
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
using namespace std;

typedef unsigned int uint;
/** Take the softmin of 3 elements
 * @param a The first element
 * @param b The second element
 * @param c The third element
 * @param gamma The smoothing factor
 */
__device__ float softmin(float a, float b, float c, const float gamma);

/** Host function for retrieving the number of SMs on the GPU device
 *  Useful for limiting the # of threadblocks to the # of SMs in a kernel launch
 *  @param device_num The device number, default 0
 *  @return the SM count
 */
__host__ uint get_device_sm_count(uint device_num = 0);

/** Kernel to fill a matrix with infinity except for index 0 = 0.0
 *  to initialize the DTW cost matrix
 */
__global__ void fill_matrix_inf(float *A, uint width, uint height, float val);

/** Check whether i,j are within the Sakoe-Chiba band
 *  @param m The length of the first time series
 *  @param n The length of the second time series
 *  @param i The cell row index
 *  @param j The cell column index
 *  @param bandwidth Maximum warping distance from the diagonal to consider for
 *  optimal path calculation (Sakoe-Chiba band). 0 = unlimited.
 */
__device__ bool check_sakoe_chiba_band(int m, int n, int i, int j,
                                       int bandwidth);

#define cudaErrchk(ans)                                                        \
    {                                                                          \
        GPUAssert((ans), __FILE__, __LINE__);                                  \
    }
inline void GPUAssert(cudaError_t code, const char *file, int line,
                      bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPU assert: %s %s %d\n", cudaGetErrorString(code),
                file, line);
        if (abort)
        {
            exit(code);
        }
    }
}
