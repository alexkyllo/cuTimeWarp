#pragma once
#include <cublas_v2.h>
typedef unsigned int uint;

/** A wrapper for cublasSgemm that works on row-major matrices by transposing
 *  A, B and C should be __device__ arrays
 *  @param A Input matrix of dimensions m * k
 *  @param B Input matrix of dimensions k * n
 *  @param C Result matrix of dimensions m * n
 *  @param m Height of matrix A and matrix C
 *  @param k Width of matrix A and height of matrix B
 *  @param n Width of matrix B and matrix C
 *  @param alpha A scalar to elementwise multiply A by
 */
__host__ void sgemm_cublas(const float *A, const float *B, float *C,
                           const uint m, const uint k, const uint n,
                           const float alpha);

/** CUDA kernel to compute the squared euclidean norm of matrix X
 *  @param m Height (rows) of matrix X
 *  @param k Width (columns) of matrix X
 *  @param XX a length m vector for the result
 */
__global__ void sq_euclid_norm(const uint m, const uint k, const float *X,
                               float *XX);

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
                            const float *YY, const float *XY, float *D);

/** Host function to compute the Squared Euclidean distance between two sets of
 *  column vectors (e.g. two multivariate time series)
 *  X and Y by using the euclidian norms, i.e. X*X + Y*Y - 2X*Y
 *  Inputs X, Y, D should be __device__ arrays.
 *  @param X A set of vectors of length (row count) m
 *  @param Y A set of vectors of length (row count) n
 *  @param D A result array for the distance matrix of dimension (m x n)
 *  @param m The length of vectors in X
 *  @param n The length of vectors in Y
 *  @param k The number of vectors in X and Y (columns)
 */
__host__ void sq_euclid_dist(const float *X, const float *Y, float *D,
                             const uint m, const uint n, const uint k);

/** Host function to compute all pairwise squared Euclidean distances between
 *  two sets of time series so that we can compute Soft-DTW
 *  on many distance matrices in parallel.
 *  Inputs X, Y, D should be __device__ arrays.
 *  @param X A set of nX vectors of length (row count) m x k (column count)
 *  @param Y A set of nY vectors of length (row count) n x k (column count)
 *  @param D A result array for the distance matrix of dimension (m x n)
 *  @param nX The number of time series in batch X
 *  @param nY The number of time series in batch Y
 *  @param m The length of vectors in X
 *  @param n The length of vectors in Y
 *  @param k The number of vectors in X and Y (columns)
 */
__host__ void sq_euclid_dist_multi(const float *X, const float *Y, float *D,
                                   const uint nX, const uint nY, const uint m,
                                   const uint n, const uint k);