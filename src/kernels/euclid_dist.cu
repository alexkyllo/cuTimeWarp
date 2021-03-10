#include "euclid_dist.cuh"

/** CUDA kernel to compute the squared euclidean norm of matrix X
 *  @param m Height (rows) of matrix X
 *  @param k Width (columns) of matrix X
 *  @param XX a length m vector for the result
 */
__global__ void sq_euclid_norm(const uint m, const uint k, const float *X,
                               float *XX)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m)
    {
        for (uint j = 0; j < k; j++)
        {
            float x = X[i * k + j];
            XX[i] += x * x;
        }
    }
}

/** CUDA kernel to compute the euclidean distance between two Euclidean norm
 * vectors XX and YY, i.e. X*X + Y*Y - 2X*Y
 *  @param m The length of vectors in X
 *  @param n The length of vectors in Y
 *  @param XX Squared Euclidean norm of X
 *  @param YY Squared Euclidean norm of Y
 *  @param XY 2 * X * Y^T (matrix multiplication result)
 *  @param D The result euclidean distance matrix with dimensions (m x n)
 */
__global__ void euclid_dist(const uint m, const uint n, const float *XX,
                            const float *YY, const float *XY, float *D)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m)
    {
        for (uint j = 0; j < n; j++)
        {
            D[i * n + j] = XX[i] + YY[j] - (XY[i * n + j]);
        }
    }
}
