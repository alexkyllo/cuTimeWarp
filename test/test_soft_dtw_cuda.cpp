#include "../src/soft_dtw.cuh"
#include "catch.h"
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

#define cudaErrchk(ans)                                                        \
    {                                                                          \
        GPUAssert((ans), __FILE__, __LINE__);                                  \
    }
inline void GPUAssert(cudaError_t code, const char *file, int line,
                      bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
        {
            exit(code);
        }
    }
}

bool is_close(float a, float b, float tol = 0.0001)
{
    return std::abs(a - b) < tol;
}

TEST_CASE("test squared euclidean distance 2d")
{
    int m = 4;
    int k = 2;
    int n = 3;
    float *X = new float[m * k]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}; // 4x2
    float *Y = new float[k * n]{1.5, 2.6, 3.7, 4.8, 5.9, 6.1}; // 3 x 2
    float *D = new float[m * n]{0};                            // 4 x 3
    float *dX;
    float *dY;
    float *dD;
    size_t size_m = m * sizeof(float);
    size_t size_n = n * sizeof(float);
    size_t size_mn = n * size_m;
    size_t size_mk = k * size_m;
    size_t size_nk = k * size_n;
    cudaMalloc(&dD, size_mn);
    cudaMalloc(&dX, size_mk);
    cudaMalloc(&dY, size_nk);
    cudaMemset(dD, 0, size_mn);
    cudaMemcpy(dX, X, size_mk, cudaMemcpyHostToDevice);
    cudaMemcpy(dY, Y, size_nk, cudaMemcpyHostToDevice);
    sq_euclid_dist(dX, dY, dD, m, n, k);
    cudaErrchk(
        cudaMemcpy(D, dD, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    /**
       sklearn.metrics.pairwise.euclidean_distances(X, Y, squared=True)

       array([[ 0.61, 15.13, 40.82],
              [ 4.21,  1.13, 12.82],
              [23.81,  3.13,  0.82],
              [59.41, 21.13,  4.82]])
     */

    // for (int i = 0; i < m; i++)
    // {
    //     for (int j = 0; j < n; j++)
    //     {
    //         std::cout << D[i * n + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

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

TEST_CASE("soft dtw cuda for distance matrix (1d ts)")
{
    int m = 5;
    int k = 1;
    int n = 8;
    float gamma = 0.1;
    float *a = new float[m]{1.0, 2.0, 3.0, 3.0, 5.0};
    float *b = new float[n]{1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0};
    // device arrays
    float *D;
    float *da;
    float *db;
    cudaMalloc(&da, m * sizeof(float));
    cudaMalloc(&db, n * sizeof(float));
    cudaMalloc(&D, m * n * sizeof(float));
    cudaMemset(D, 0, m * n * sizeof(float));
    cudaMemcpy(da, a, m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, n * sizeof(float), cudaMemcpyHostToDevice);

    float *R;
    size_t m2n2 = (m + 2) * (n + 2);
    size_t sz_R = m2n2 * sizeof(float);
    cudaMalloc(&R, sz_R);

    sq_euclid_dist(da, db, D, m, n, k);

    float cost = softdtw_cuda_naive(D, R, m, n, gamma);
    /*
R expected:
[0. inf     inf     inf     inf     inf     inf     inf       inf          inf]
[inf  0.      1.      2.      3.      4.      5.      6.       15.         inf]
[inf  1.     -0.     -0.     -0.     -0.     -0.     -0.        4.         inf]
[inf  5.      1.      0.9307  0.9307  0.9307  0.9307  0.9307    1.         inf]
[inf  9.      2.      1.8901  1.8614  1.8613  1.8613  1.8613    1.8901     inf]
[inf 25.     11.     10.8614 10.8054 10.792  10.792  10.792     2.8054     inf]
[inf inf     inf     inf     inf     inf     inf     inf           inf     inf]
    */
    // std::cout << "cost: " << cost << std::endl;
    REQUIRE(is_close(2.80539, cost));
    delete[] a;
    delete[] b;
    cudaFree(D);
    cudaFree(da);
    cudaFree(db);
    cudaFree(R);
}

TEST_CASE("soft dtw gradient CUDA")
{
    int m = 5;
    int k = 1;
    int n = 8;
    float gamma = 0.1;
    float *a = new float[m]{1.0, 2.0, 3.0, 3.0, 5.0};
    float *b = new float[n]{1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0};
    float *E = new float[m * n];
    // device arrays
    float *da;
    cudaMalloc(&da, m * sizeof(float));
    cudaMemcpy(da, a, m * sizeof(float), cudaMemcpyHostToDevice);

    float *db;
    cudaMalloc(&db, n * sizeof(float));
    cudaMemcpy(db, b, n * sizeof(float), cudaMemcpyHostToDevice);

    float *D;
    cudaMalloc(&D, m * n * sizeof(float));
    cudaMemset(D, 0, m * n * sizeof(float));

    float *R;
    size_t m2n2 = (m + 2) * (n + 2);
    size_t sz_R = m2n2 * sizeof(float);
    cudaMalloc(&R, sz_R);
    cudaMemset(R, 0, sz_R);

    float *dE;
    cudaMalloc(&dE, m * n * sizeof(float));
    cudaMemset(dE, 0, m * n * sizeof(float));

    sq_euclid_dist(da, db, D, m, n, k);

    softdtw_cuda_naive(D, R, m, n, gamma);
    softdtw_grad_cuda_naive(D, R, dE, m, n, gamma);
    cudaMemcpy(E, dE, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < m; i++)
    // {
    //     for (int j = 0; j < n; j++)
    //     {
    //         std::cout << E[i * n + j] << " ";
    //     }
    //     std::cout << "\n";
    // }
    REQUIRE(is_close(E[0], 1.0));
    REQUIRE(is_close(E[1], 0.0001));
    REQUIRE(is_close(E[13], 0.8571));
    REQUIRE(is_close(E[14], 0.4285));
    REQUIRE(is_close(E[21], 0.2857));
    REQUIRE(is_close(E[22], 0.5714));
    REQUIRE(is_close(E[23], 0.1429));
    REQUIRE(is_close(E[31], 0.4286));
    REQUIRE(is_close(E[39], 1.0));
    /* Expected gradient:
array([[[1.    , 0.0001, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
        [0.    , 1.    , 1.    , 1.    , 1.    , 0.8571, 0.4285, 0.    ],
        [0.    , 0.    , 0.    , 0.    , 0.    , 0.2857, 0.5714, 0.1429],
        [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.5714, 0.4286],
        [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.    ]]])
     */
    delete[] a;
    delete[] b;
    delete[] E;
    cudaFree(da);
    cudaFree(db);
    cudaFree(D);
    cudaFree(R);
    cudaFree(dE);
}
