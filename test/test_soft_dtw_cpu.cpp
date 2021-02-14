#include "../src/soft_dtw_cpu.hpp"
#include "catch.h"
#include <cmath>
#include <iostream>

bool is_close(float a, float b, float tol=0.0001)
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
                2.0,         // Alpha (scalar used to scale A*B)
                X,             // Input Matrix A
                k,             // LDA stride of matrix A
                Y,             // Input Matrix B
                k,             // LDA stride of matrix B
                0.0,           // Beta (scalar used to scale matrix C)
                XY,             // Result Matrix C
                n);            // LDA stride of matrix C
    float *XYE = new float[12] {
         13.4,  26.6,  36.2,
         29.8,  60.6,  84.2,
         46.2,  94.6, 132.2,
         62.6, 128.6, 180.2
    };
    for (int i = 0; i < (m*n); i++)
    {
        REQUIRE(is_close(XY[i], XYE[i]));
    }
}

TEST_CASE("test squared euclidean distance")
{
    int m = 4;
    int k = 2;
    int n = 3;
    float *X = new float[m * k]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    float *Y = new float[k * n]{1.5, 2.6, 3.7, 4.8, 5.9, 6.1};
    float *D = new float[m * n]{0};
    sq_euclidean_distance(X, Y, D, m, n, k);
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
}

TEST_CASE("soft dtw")
{
    int m = 5;
    int n = 8;
    float *a = new float[m]{1.0, 2.0, 3.0, 3.0, 5.0};
    float *b = new float[n]{1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0};
    float *w = new float[(m + 1) * (n + 1)]{0.0};
    float cost = softdtw<float>(a, b, w, m, n, 0.1);
    REQUIRE(is_close(2.80539, cost));
}
