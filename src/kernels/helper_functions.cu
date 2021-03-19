/** Helper functions shared across CUDA kernels
 *  @file helper_functions.cu
 *  @author Alex Kyllo
 *  @date 2021-03
 */
#include "helper_functions.cuh"

/** Host function for retrieving the number of SMs on the GPU device
 *  Useful for limiting the # of threadblocks to the # of SMs in a kernel launch
 *  @param device_num The device number, default 0
 *  @return the SM count
 */
__host__ uint get_device_sm_count(uint device_num)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_num);
    return deviceProp.multiProcessorCount;
}

/** Take the softmin of 3 elements
 * @param a The first element
 * @param b The second element
 * @param c The third element
 * @param gamma The smoothing factor
 */
__device__ float softmin(float a, float b, float c, const float gamma)
{
    float ng = -gamma;
    a /= ng;
    b /= ng;
    c /= ng;
    float max_of = max(max(a, b), c);
    float sum = exp(a - max_of) + exp(b - max_of) + exp(c - max_of);

    return ng * (log(sum) + max_of);
}

/** Check whether i,j are within the Sakoe-Chiba band
 *  @param m The length of the first time series
 *  @param n The length of the second time series
 *  @param i The cell row index
 *  @param j The cell column index
 *  @param bandwidth Maximum warping distance from the diagonal to consider for
 *  optimal path calculation (Sakoe-Chiba band). 0 = unlimited.
 */
__device__ bool check_sakoe_chiba_band(int m, int n, int i, int j,
                                       int bandwidth)
{
    if (bandwidth == 0)
    {
        return true;
    }
    int width = abs(m - n) + bandwidth;
    int lower = max(1, (m > n ? j : i) - bandwidth);
    int upper = min(max(m, n), (m > n ? j : i) + width) + 1;
    bool is_in_lower = (m > n ? i : j) >= lower;
    bool is_in_upper = (m > n ? i : j) < upper;
    return is_in_lower && is_in_upper;
}

/** Kernel to fill a matrix with infinity except for index 0 = 0.0
 *  to initialize the DTW cost matrix
 */
__global__ void fill_matrix_inf(float *A, uint width, uint height, float val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < width * height; i += gridDim.x * blockDim.x)
    {
        A[i] = val;
        if (i % width == 0)
            A[i] = 0.0;
    }
}
