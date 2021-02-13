/** Naive CPU implmentation of DTW as a baseline and reference.
 * @file softdtw.cpp
 * @author Alex Kyllo
 * @date 2020-02-12
 */
#include <cmath>
#include <iostream>
#include <limits>
#include <cblas.h>

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

/**
 * @param a The first series array
 * @param b The second series array
 * @param w An m+1 x n+1 array that will be filled with the alignment values.
 * @param m Length of array a
 * @param n Length of array b
 * @param gamma SoftDTW smoothing parameter
 */
template <class T> T softdtw(T *a, T *b, T *w, int m, int n, T gamma)
{
    // Create an m*n matrix for the warp path
    // and initialize it to infinite distances
    m++;
    n++;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            // this doesn't work for int type. Only float and double
            w[i * n + j] = std::numeric_limits<T>::infinity();
        }
    }
    w[0] = 0.0;

    // Iterate over each cell of the matrix to compute
    // lowest cost path through the preceding neighbor cells
    for (int i = 1; i < m; i++)
    {
        for (int j = 1; j < n; j++)
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

template <class T> void gemm_blas(T *A, T *B, T *C, long m, long k, long n, T alpha);

/** Double Matrix-matrix multiply
    @param A The first input matrix
    @param X The second input matrix
    @param B The output matrix
    @param m The number of rows in A
    @param k The number of columns in A and rows in B
    @param n The number of columns in B
*/
template <> void gemm_blas<double>(double *A, double *B, double *C, long m, long k, long n, double alpha)
{
    cblas_dgemm(CblasRowMajor, // Row-major striding
                CblasNoTrans,  // Do not transpose A
                CblasTrans,    // Transpose B
                m,             // Rows in A
                n,             // Columns in B
                k,             // Columns in A
                alpha,           // Alpha (scalar used to scale A*B)
                A,             // Input Matrix A
                k,             // LDA stride of matrix A
                B,             // INput Matrix B
                n,             // LDA stride of matrix B
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
template <> void gemm_blas<float>(float *A, float *B, float *C, long m, long k, long n, float alpha)
{
    cblas_sgemm(CblasRowMajor, // Row-major striding
                CblasNoTrans,  // Do not transpose A
                CblasTrans,    // Transpose B
                m,             // Rows in A
                n,             // Columns in B
                k,             // Columns in A
                alpha,           // Alpha (scalar used to scale A*B)
                A,             // Input Matrix A
                k,             // LDA stride of matrix A
                B,             // Input Matrix B
                n,             // LDA stride of matrix B
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
template <class T> T sq_euclidean_distance(T *X, T *Y, T *D, int m, int n, int k)
{
    // TODO
    // Need to implement matrix multiplication and transpose for this
    // dist(x, y) = dot(x, x) - 2 * dot(x, y) + dot(y, y)
    T *XX = new T[m];
    T *YY = new T[n];
    T *XY = new T[m * n];

    // square the elements of X and then row-wise sum them
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            T x = X[i * k + j];
            XX[i] += x * x;
        }
    }

    // square the elements of Y and then row-wise sum them
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < k; j++)
        {
            T y = Y[i * k + j];
            YY[i] += y * y;
        }
    }

    // compute 2*X*YT
    gemm_blas<T>(X, Y, XY, m, k, n, 2.0);

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++ )
        {
            D[i * n + j] = XX[i] + YY[j] - (2 * XY[i * n + j]);
        }
    }

    delete[] XX;
    delete[] YY;
    delete[] XY;
}



template <class T> void softdtw_grad(T *D, T *R, T *E, T gamma)
{
    // TODO
}

int main()
{
    // test values
    int m = 5;
    int n = 8;
    float *a = new float[m]{1.0, 2.0, 3.0, 3.0, 5.0};
    float *b = new float[n]{1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0};

    float *w = new float[(m + 1) * (n + 1)]{0.0};
    float cost = softdtw<float>(a, b, w, m, n, 0.1);
    // Should print: 2.80539
    std::cout << "SoftDTW distance: " << cost << "\n";
    std::cout << "Alignment matrix:\n";
    for (int i = 0; i < m + 1; i++)
    {
        for (int j = 0; j < n + 1; j++)
        {
            std::cout << w[i * (n + 1) + j] << " ";
        }
        std::cout << "\n";
    }
    delete[] a;
    delete[] b;
    delete[] w;
    return 0;
}
