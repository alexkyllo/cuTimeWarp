#include "../src/soft_dtw_cpu.hpp"
#include "catch.h"
#include <cmath>
#include <iostream>

bool is_close(float a, float b, float tol = 0.0001)
{
    return std::abs(a - b) < tol;
}

TEST_CASE("CBLAS_SGEMM")
{
    int m = 4;
    int k = 2;
    int n = 3;
    float *X = new float[m * k]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    float *Y = new float[k * n]{1.5, 2.6, 3.7, 4.8, 5.9, 6.1};
    float *XY = new float[m * n]{0};
    cblas_sgemm(CblasRowMajor, // Row-major striding
                CblasNoTrans,  // Do not transpose A
                CblasTrans,    // Transpose B
                m,             // Rows in A
                n,             // Columns in B
                k,             // Columns in A
                2.0,           // Alpha (scalar used to scale A*B)
                X,             // Input Matrix A
                k,             // LDA stride of matrix A
                Y,             // Input Matrix B
                k,             // LDA stride of matrix B
                0.0,           // Beta (scalar used to scale matrix C)
                XY,            // Result Matrix C
                n);            // LDA stride of matrix C
    float *XYE = new float[12]{13.4, 26.6, 36.2,  29.8, 60.6,  84.2,
                               46.2, 94.6, 132.2, 62.6, 128.6, 180.2};
    for (int i = 0; i < (m * n); i++)
    {
        REQUIRE(is_close(XY[i], XYE[i]));
    }
    delete[] X;
    delete[] Y;
    delete[] XY;
    delete[] XYE;
}

TEST_CASE("test squared euclidean distance 2d")
{
    int m = 4;
    int k = 2;
    int n = 3;
    float *X = new float[m * k]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    float *Y = new float[k * n]{1.5, 2.6, 3.7, 4.8, 5.9, 6.1};
    float *D = new float[m * n]{0};
    sq_euclidean_distance<float>(X, Y, D, m, n, k);
    /**
       sklearn.metrics.pairwise.euclidean_distances(X, Y, squared=True)

       array([[ 0.61, 15.13, 40.82],
              [ 4.21,  1.13, 12.82],
              [23.81,  3.13,  0.82],
              [59.41, 21.13,  4.82]])
     */

    REQUIRE(is_close(D[0], 0.61));
    REQUIRE(is_close(D[1], 15.13));
    REQUIRE(is_close(D[2], 40.82));
    REQUIRE(is_close(D[3], 4.21));
    REQUIRE(is_close(D[4], 1.13));
    REQUIRE(is_close(D[5], 12.82));
    REQUIRE(is_close(D[6], 23.81));
    REQUIRE(is_close(D[7], 3.13));
    REQUIRE(is_close(D[8], 0.82));
    REQUIRE(is_close(D[9], 59.41));
    REQUIRE(is_close(D[10], 21.13));
    REQUIRE(is_close(D[11], 4.82));
    delete[] X;
    delete[] Y;
    delete[] D;
}

TEST_CASE("test squared euclidean distance 1d")
{
    int m = 5;
    int k = 1;
    int n = 8;
    float *a = new float[m]{1.0, 2.0, 3.0, 3.0, 5.0};
    float *b = new float[n]{1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0};
    float *D = new float[m * n]{0};
    sq_euclidean_distance<float>(a, b, D, m, n, k);
    /**
       sklearn.metrics.pairwise.euclidean_distances(X, Y, squared=True)

       array([[ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  9.],
              [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  4.],
              [ 4.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
              [ 4.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
              [16.,  9.,  9.,  9.,  9.,  9.,  9.,  1.]])
    */

    REQUIRE(is_close(D[0], 0.0));
    REQUIRE(is_close(D[1], 1.0));
    REQUIRE(is_close(D[7], 9.0));
    REQUIRE(is_close(D[8], 1.0));
    REQUIRE(is_close(D[15], 4.0));
    REQUIRE(is_close(D[16], 4.0));
    REQUIRE(is_close(D[24], 4.0));
    REQUIRE(is_close(D[32], 16.0));
    REQUIRE(is_close(D[33], 9.0));
    REQUIRE(is_close(D[39], 1.0));

    delete[] a;
    delete[] b;
    delete[] D;
}

TEST_CASE("soft dtw")
{
    int m = 5;
    int n = 8;
    float gamma = 0.1;
    float *a = new float[m]{1.0, 2.0, 3.0, 3.0, 5.0};
    float *b = new float[n]{1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0};
    float *R = new float[(m + 1) * (n + 1)]{0.0};
    float cost = softdtw<float>(a, b, R, m, n, gamma);
    REQUIRE(is_close(2.80539, cost));
    delete[] a;
    delete[] b;
    delete[] R;
}

TEST_CASE("soft dtw for distance matrix (1d ts)")
{
    int m = 5;
    int k = 1;
    int n = 8;
    float gamma = 0.1;
    float *a = new float[m]{1.0, 2.0, 3.0, 3.0, 5.0};
    float *b = new float[n]{1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0};
    float *D = new float[m * n]{0};
    sq_euclidean_distance<float>(a, b, D, m, n, k);

    float *R = new float[(m + 1) * (n + 1)]{0.0};
    float cost = softdtw<float>(D, R, m, n, gamma);
    for (int i = 0; i <= m; i++)
    {
        for (int j = 0; j <= n; j++)
        {
            std::cout << R[i * (n + 1) + j] << " ";
        }
        std::cout << "\n";
    }
    /*
R expected:
array([[ 0.00000000e+000,  1.79769313e+308,  1.79769313e+308,
         1.79769313e+308,  1.79769313e+308,  1.79769313e+308,
         1.79769313e+308,  1.79769313e+308,  1.79769313e+308,
         0.00000000e+000],
       [ 1.79769313e+308,  0.00000000e+000,  1.00000000e+000,
         2.00000000e+000,  3.00000000e+000,  4.00000000e+000,
         5.00000000e+000,  6.00000000e+000,  1.50000000e+001,
         0.00000000e+000],
       [ 1.79769313e+308,  1.00000000e+000, -9.07957375e-006,
        -1.36192576e-005, -1.36194637e-005, -1.36194637e-005,
        -1.36194637e-005, -1.36194637e-005,  3.99998638e+000,
         0.00000000e+000],
       [ 1.79769313e+308,  5.00000000e+000,  9.99986381e-001,
         9.30671662e-001,  9.30667123e-001,  9.30667122e-001,
         9.30667122e-001,  9.30667122e-001,  9.99977301e-001,
         0.00000000e+000],
       [ 1.79769313e+308,  9.00000000e+000,  1.99998638e+000,
         1.89012364e+000,  1.86135127e+000,  1.86134786e+000,
         1.86134786e+000,  1.86134786e+000,  1.89011304e+000,
         0.00000000e+000],
       [ 1.79769313e+308,  2.50000000e+001,  1.09999864e+001,
         1.08613558e+001,  1.08053915e+001,  1.07920348e+001,
         1.07920331e+001,  1.07920331e+001,  2.80538499e+000,
         0.00000000e+000],
       [ 0.00000000e+000,  0.00000000e+000,  0.00000000e+000,
         0.00000000e+000,  0.00000000e+000,  0.00000000e+000,
         0.00000000e+000,  0.00000000e+000,  0.00000000e+000,
         0.00000000e+000]])
     */
    std::cout << "cost: " << cost << "\n";
    REQUIRE(is_close(2.80539, cost));
    delete[] a;
    delete[] b;
    delete[] D;
    delete[] R;
}

// TEST_CASE("soft dtw gradient")
// {
//     int m = 5;
//     int n = 8;
//     float gamma = 0.1;
//     float *a = new float[m]{1.0, 2.0, 3.0, 3.0, 5.0};
//     float *b = new float[n]{1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0};
//     float *D = new float[(m + 1) * (n + 1)]{0.0};
//     float *R = new float[(m + 1) * (n + 1)]{0.0};
//     float *E = new float[(m + 2) * (n + 2)]{0.0};
//     sq_euclidean_distance(a, b, D, m, n, 1);
//     softdtw<float>(a, b, D, m, n, 0.1);
//     softdtw_grad(D, R, E, m, n, gamma);
// }
