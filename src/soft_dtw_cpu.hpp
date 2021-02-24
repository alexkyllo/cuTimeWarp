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
void gemm_blas(T *A, T *B, T *C, size_t m, size_t k, size_t n, T alpha);

/** Double Matrix-matrix multiply
    @param A The first input matrix
    @param X The second input matrix
    @param B The output matrix
    @param m The number of rows in A
    @param k The number of columns in A and rows in B
    @param n The number of columns in B
*/
template <>
void gemm_blas<double>(double *A, double *B, double *C, size_t m, size_t k,
                       size_t n, double alpha)
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

/** Float Matrix-matrix vmultiply
    @param A The first input matrix
    @param X The second input matrix
    @param B The output matrix
    @param m The number of rows in A
    @param k The number of columns in A and rows in B
    @param n The number of columns in B
*/
template <>
void gemm_blas<float>(float *A, float *B, float *C, size_t m, size_t k,
                      size_t n, float alpha)
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
void sq_euclidean_distance(T *X, T *Y, T *D, size_t m, size_t n, int k)
{
    T *XX = new T[m]{0};
    T *YY = new T[n]{0};
    T *XY = new T[m * n]{0};

    // compute squared euclidean norm of X
    for (size_t i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            T x = X[i * k + j];
            XX[i] += x * x;
        }
    }

    // compute squared euclidean norm of Y
    for (size_t i = 0; i < n; i++)
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
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
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
    T sum = exp(a - max_of) + exp(b - max_of) + exp(c - max_of);

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
template <class T> T softdtw(T *a, T *b, T *w, size_t m, size_t n, T gamma)
{
    // Create an m*n matrix for the warp path
    // and initialize it to infinite distances
    m++;
    n++;
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            // this doesn't work for int type. Only float and double
            w[i * n + j] = std::numeric_limits<T>::infinity();
        }
    }
    w[0] = 0.0;

    // Iterate over each cell of the matrix to compute
    // lowest cost path through the preceding neighbor cells
    for (size_t i = 1; i < m; i++)
    {
        for (size_t j = 1; j < n; j++)
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

/** Soft DTW on pairwise Euclidean distance matrix for multivariate time
 * series
 * @param D The pairwise squared Euclidean distance array of two time series
 * @param R An m+1 x n+1 array that will be filled with the alignment
 * values.
 * @param m Length of first time series
 * @param n Length of second time series
 * @param gamma SoftDTW smoothing parameter
 */
template <class T> T softdtw(T *D, T *R, size_t m, size_t n, T gamma)
{
    // Create an m*n matrix for the warp path
    // and initialize it to infinite distances
    for (size_t i = 0; i < m + 2; i++)
    {
        for (size_t j = 0; j < n + 2; j++)
        {
            R[i * (n + 2) + j] = std::numeric_limits<T>::infinity();
        }
    }

    R[0] = 0.0;

    // Iterate over each cell of the matrix to compute
    // lowest cost path through the preceding neighbor cells
    for (size_t i = 1; i < m + 1; i++)
    {
        for (size_t j = 1; j < n + 1; j++)
        {
            T cost = D[(i - 1) * n + j - 1];
            T r1 = R[(i - 1) * (n + 2) + j];
            T r2 = R[i * (n + 2) + j - 1];
            T r3 = R[(i - 1) * (n + 2) + j - 1];
            double prev_min = softmin<T>(r1, r2, r3, gamma);
            R[i * (n + 2) + j] = cost + prev_min;
        }
    }
    // Return the total cost of the warp path
    T path_cost = R[m * (n + 2) + n];
    return path_cost;
}

/** SoftDTW gradient by backpropagation
 * @param D The pairwise squared Euclidean distance array of two time series
 * @param R An m+1 x n+1 array that will be filled with the alignment
 * values.
 * @param E An m+2 x n+2 array that will be filled with the gradient values.
 * @param m Length of first time series
 * @param n Length of second time series
 * @param gamma SoftDTW smoothing parameter
 */
template <class T>
void softdtw_grad(T *D_, T *R, T *E_, size_t m, size_t n, T gamma)
{
    // TODO: fix this, it's returning a wrong result
    // Add an extra row and column to D
    T *D = new T[(m + 1) * (n + 1)]{0};
    T *E = new T[(m + 2) * (n + 2)]{0};
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            D[i * (n + 1) + j] = D_[i * n + j];
        }
    }

    // D is m+1 x n+1
    // R and E are m+2 x n+2
    for (size_t i = 1; i < m + 1; i++)
    {
        D[(i - 1) * (n + 1) + n] = 0.0;
        R[i * (n + 2) + n + 1] = -std::numeric_limits<T>::infinity();
    }

    for (size_t i = 1; i < n + 1; i++)
    {
        D[m * (n + 1) + (i - 1)] = 0.0;
        R[(m + 1) * (n + 2) + i] = -std::numeric_limits<T>::infinity();
    }

    E[(m + 1) * (n + 2) + n + 1] = 1;
    R[(m + 1) * (n + 2) + n + 1] = R[m * (n + 2) + n];
    D[m * (n + 1) + n] = 0.0;
    for (size_t j = n; j > 0; j--)
    {
        for (size_t i = m; i > 0; i--)
        {
            T r0 = R[i * (n + 2) + j];
            T a =
                exp((R[(i + 1) * (n + 2) + j] - r0 - D[i * (n + 1) + (j - 1)]) /
                    gamma);
            T b =
                exp((R[i * (n + 2) + (j + 1)] - r0 - D[(i - 1) * (n + 1) + j]) /
                    gamma);
            T c = exp((R[(i + 1) * (n + 2) + j + 1] - r0 - D[i * (n + 1) + j]) /
                      gamma);
            E[i * (n + 2) + j] = E[(i + 1) * (n + 2) + j] * a +
                                 E[i * (n + 2) + j + 1] * b +
                                 E[(i + 1) * (n + 2) + j + 1] * c;
        }
    }
    // Copy E to the E_ without the first and last row and column
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            E_[i * n + j] = E[(i + 1) * (n + 2) + (j + 1)];
        }
    }
    delete[] D;
    delete[] E;
}

#endif
