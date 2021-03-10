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
    a /= -gamma;
    b /= -gamma;
    c /= -gamma;
    float max_of = max(max(a, b), c);
    float sum = exp(a - max_of) + exp(b - max_of) + exp(c - max_of);

    return -gamma * (log(sum) + max_of);
}

/** Kernel to fill a matrix with infinity except for index 0 = 0.0
 *  to initialize the DTW cost matrix
 */
__global__ void fill_matrix_inf(float *A, int width, int height, float val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < width * height; i += gridDim.x * blockDim.x)
    {
        A[i] = val;
        if (i % width == 0)
            A[i] = 0.0;
    }
}