/** Naive CPU implmentation of DTW as a baseline and reference.
 * @file soft_dtw_cpu.hpp
 * @author Alex Kyllo
 * @date 2020-02-13
 */
#ifndef SOFT_DTW_CPU_H
#define SOFT_DTW_CPU_H
#include <cblas.h>
#include <cmath>
#include <iostream>
#include <limits>

template <class T>
void gemm_blas(T *A, T *B, T *C, ulong m, ulong k, ulong n, T alpha);

/** Double Matrix-matrix multiply
    @param A The first input matrix
    @param X The second input matrix
    @param B The output matrix
    @param m The number of rows in A
    @param k The number of columns in A and rows in B
    @param n The number of columns in B
*/
template <>
void gemm_blas<double>(double *A, double *B, double *C, ulong m, ulong k,
                       ulong n, double alpha)
{
    cblas_dgemm(CblasRowMajor, // Row-major striding
                CblasNoTrans,  // Do not transpose A
                CblasTrans,    // Transpose B
                m,             // Rows in A
                n,             // Columns in B
                k,             // Columns in A
                alpha,         // Alpha (scalar used to scale A*B)
                A,             // Input Matrix A
                k,             // LDA stride of matrix A
                B,             // INput Matrix B
                k,             // LDA stride of matrix B
                0.0,           // Beta (scalar used to scale matrix C)
                C,             // Result Matrix C
                n);            // LDA stride of matrix C
}

/** Float Matrix-matrix multiply
    @param A The first input matrix
    @param X The second input matrix
    @param B The output matrix
    @param m The number of rows in A
    @param k The number of columns in A and rows in B
    @param n The number of columns in B
*/
template <>
void gemm_blas<float>(float *A, float *B, float *C, ulong m, ulong k, ulong n,
                      float alpha)
{
    cblas_sgemm(CblasRowMajor, // Row-major striding
                CblasNoTrans,  // Do not transpose A
                CblasTrans,    // Transpose B
                m,             // Rows in A
                n,             // Columns in B
                k,             // Columns in A
                alpha,         // Alpha (scalar used to scale A*B)
                A,             // Input Matrix A
                k,             // LDA stride of matrix A
                B,             // Input Matrix B
                k,             // LDA stride of matrix B
                0.0,           // Beta (scalar used to scale matrix C)
                C,             // Result Matrix C
                n);            // LDA stride of matrix C
}

/** Compute pairwise squared Euclidean distances between X and Y
 *  where each are multivariate time series with k features.
 *  @param X a matrix where rows are time steps and columns are variables
 *  @param Y a matrix where rows are time steps and columns are variables
 *  @param D The resulting distance matrix of size m x n
 *  @param m Number of rows in X
 *  @param n Number of rows in Y
 *  @param k Number of columns in X and Y
 */
template <class T>
void sq_euclidean_distance(T *X, T *Y, T *D, ulong m, ulong n, int k)
{
    T *XX = new T[m]{0};
    T *YY = new T[n]{0};
    T *XY = new T[m * n]{0};

    // compute squared euclidean norm of X
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            T x = X[i * k + j];
            XX[i] += x * x;
        }
    }

    // compute squared euclidean norm of Y
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < k; j++)
        {
            T y = Y[i * k + j];
            YY[i] += y * y;
        }
    }

    // compute (2*X)*YT
    gemm_blas<T>(X, Y, XY, m, k, n, 2.0);

    // compute x^2 + y^2 - 2xy
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            D[i * n + j] = XX[i] + YY[j] - (XY[i * n + j]);
        }
    }

    delete[] XX;
    delete[] YY;
    delete[] XY;
}

/** Take the softmin of 3 elements
 * @param a The first element
 * @param b The second element
 * @param c The third element
 * @param gamma The smoothing factor
 */
template <class T> T softmin(T a, T b, T c, T gamma)
{
    a /= -gamma;
    b /= -gamma;
    c /= -gamma;
    T max_of = std::max(std::max(a, b), c);
    T sum = 0;
    sum += exp(a - max_of);
    sum += exp(b - max_of);
    sum += exp(c - max_of);

    return -gamma * (log(sum) + max_of);
}

/** Soft DTW on two input time series
 * @param a The first series array
 * @param b The second series array
 * @param w An m+1 x n+1 array that will be filled with the alignment values.
 * @param m Length of array a
 * @param n Length of array b
 * @param gamma SoftDTW smoothing parameter
 */
template <class T> T softdtw(T *a, T *b, T *w, ulong m, ulong n, T gamma)
{
    // Create an m*n matrix for the warp path
    // and initialize it to infinite distances
    m++;
    n++;
    for (ulong i = 0; i < m; i++)
    {
        for (ulong j = 0; j < n; j++)
        {
            // this doesn't work for int type. Only float and double
            w[i * n + j] = std::numeric_limits<T>::infinity();
        }
    }
    w[0] = 0.0;

    // Iterate over each cell of the matrix to compute
    // lowest cost path through the preceding neighbor cells
    for (ulong i = 1; i < m; i++)
    {
        for (ulong j = 1; j < n; j++)
        {
            T cost = std::abs(a[i - 1] - b[j - 1]);
            double prev_min = softmin<T>(w[(i - 1) * n + j], w[i * n + j - 1],
                                         w[(i - 1) * n + j - 1], gamma);
            w[i * n + j] = cost + prev_min;
        }
    }
    // Return the total cost of the warp path
    T path_cost = w[m * n - 1];
    return path_cost;
}

template <class T> T softdtw2(T *D, T *R, ulong m, ulong n, T gamma)
{
    // Create an m*n matrix for the warp path
    // and initialize it to infinite distances
    m++;
    n++;
    for (ulong i = 0; i < m; i++)
    {
        R[i * n] = std::numeric_limits<T>::infinity();
    }
    for (ulong j = 0; j < n; j++)
    {
        R[j] = std::numeric_limits<T>::infinity();
    }
    R[0] = 0.0;

    // Iterate over each cell of the matrix to compute
    // lowest cost path through the preceding neighbor cells
    for (ulong i = 1; i < m; i++)
    {
        for (ulong j = 1; j < n; j++)
        {
            T cost = D[(i - 1) * n + j - 1];
            double prev_min = softmin<T>(R[(i - 1) * n + j], R[i * n + j - 1],
                                         R[(i - 1) * n + j - 1], gamma);
            R[i * n + j] = cost + prev_min;
        }
    }
    // Return the total cost of the warp path
    T path_cost = R[m * n - 1];
    return path_cost;
}

/** Soft DTW on pairwise Euclidean distance matrix for multivariate time series
 * @param D The pairwise squared Euclidean distance array of two time series
 * @param R An m+1 x n+1 array that will be filled with the alignment values.
 * @param m Length of first time series
 * @param n Length of second time series
 * @param gamma SoftDTW smoothing parameter
 */
template <class T> T softdtw(T *D, T *R, ulong m, ulong n, T gamma)
{
    // TODO: not working correctly, fix failing test
    // Create an m*n matrix for the warp path
    // and initialize it to infinite distances
    for (ulong i = 0; i <= m; i++)
    {
        R[i * (n + 1)] = std::numeric_limits<T>::infinity();
    }
    for (ulong j = 0; j <= n; j++)
    {
        R[j] = std::numeric_limits<T>::infinity();
    }
    R[0] = 0.0;

    // Iterate over each cell of the matrix to compute
    // lowest cost path through the preceding neighbor cells
    for (ulong i = 1; i <= m; i++)
    {
        for (ulong j = 1; j <= n; j++)
        {
            T cost = D[(i - 1) * (n + 1) + j - 1];
            double prev_min =
                softmin<T>(R[(i - 1) * (n + 1) + j], R[i * (n + 1) + j - 1],
                           R[(i - 1) * (n + 1) + j - 1], gamma);
            R[i * (n + 1) + j] = cost + prev_min;
        }
    }
    // Return the total cost of the warp path
    T path_cost = R[(m + 1) * (n + 1) - 1];
    return path_cost;
}

/** SoftDTW gradient by backpropagation
 * @param D The pairwise squared Euclidean distance array of two time series
 * @param R An m+1 x n+1 array that will be filled with the alignment values.
 * @param E An m+2 x n+2 array that will be filled with the gradient values.
 * @param m Length of first time series
 * @param n Length of second time series
 * @param gamma SoftDTW smoothing parameter
 */
template <class T>
void softdtw_grad(T *D, T *R, T *E, ulong m, ulong n, T gamma)
{
    for (ulong i = 1; i <= m; i++)
    {
        D[(i - 1) * n + n] = 0.0;
        R[i * n + n + 1] = -std::numeric_limits<T>::infinity();
    }
    for (ulong i = 1; i <= n; i++)
    {
        D[m * n + (i - 1)] = 0.0;
        R[(m + 1) * n + i] = -std::numeric_limits<T>::infinity();
    }

    E[(m + 1) * n + n + 1] = 1;
    R[(m + 1) * n + n + 1] = R[m * n + n];
    D[m * n + n] = 0;
    for (ulong j = n; j >= 0; j--)
    {
        for (ulong i = m; i >= 0; i--)
        {
            T a = exp((R[(i + 1) * n + j] - R[i * n + j] - D[i * n + (j - 1)]) /
                      gamma);
            T b = exp((R[i * n + (j + 1)] - R[i * n + j] - D[(i - 1) * n + j]) /
                      gamma);
            T c = exp((R[(i + 1) * n + j + 1] - R[i * n + j] - D[i * n + j]) /
                      gamma);
            E[i * n + j] = E[(i + 1) * n + j] * a + E[i * n + j + 1] * b +
                           E[(i + 1) * n + j + 1] * c;
        }
    }
}

#endif
