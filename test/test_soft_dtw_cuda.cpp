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

void print_matrix(const float *X, const uint m, const uint n)
{
    for (uint i = 0; i < m; i++)
    {
        for (uint j = 0; j < n; j++)
        {
            std::cout << X[i * n + j] << " ";
        }
        std::cout << "\n";
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

    cudaFree(dX);
    cudaFree(dY);
    cudaFree(dD);
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
    const int m = 5;
    const int k = 1;
    const int n = 8;
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

    float E_ex[40]{1.0, 0.0001, 0.0, 0.0, 0.0, 0.0,    0.0,    0.0,
                   0.0, 1.0,    1.0, 1.0, 1.0, 0.8571, 0.4285, 0.0,
                   0.0, 0.0,    0.0, 0.0, 0.0, 0.2857, 0.5714, 0.1429,
                   0.0, 0.0,    0.0, 0.0, 0.0, 0.0,    0.5714, 0.4286,
                   0.0, 0.0,    0.0, 0.0, 0.0, 0.0,    0.0,    1.0};
    for (int i = 0; i < m * n; i++)
    {
        REQUIRE(is_close(E[i], E_ex[i]));
    }
    delete[] a;
    delete[] b;
    delete[] E;
    cudaFree(da);
    cudaFree(db);
    cudaFree(D);
    cudaFree(R);
    cudaFree(dE);
}

TEST_CASE("test squared euclidean distance 2d with multi nX = 1 nY = 1")
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
    sq_euclid_dist_multi(dX, dY, dD, 1, 1, m, n, k);
    cudaErrchk(
        cudaMemcpy(D, dD, m * n * sizeof(float), cudaMemcpyDeviceToHost));
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

    cudaFree(dX);
    cudaFree(dY);
    cudaFree(dD);
}

TEST_CASE("Test squared Euclidan distance multi")
{
    /*
    import numpy as np
    from sklearn.metrics.pairwise import euclidean_distances
    >>> a
    np.array([[1., 2., 3., 3., 5.],
             [5., 3., 3., 2., 1.]])
    >>> b
    np.array([[1., 2., 2., 2., 2., 2., 2., 4.],
              [4., 2., 2., 2., 2., 2., 2., 1.]])
    D = np.array([euclidean_distances(x.reshape(-1,1), y.reshape(-1,1),
    squared=True) for x in a for y in b])
    np.array([[
        [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  9.],
        [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  4.],
        [ 4.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
        [ 4.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
        [16.,  9.,  9.,  9.,  9.,  9.,  9.,  1.]],

       [[ 9.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
        [ 4.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
        [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  4.],
        [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  4.],
        [ 1.,  9.,  9.,  9.,  9.,  9.,  9., 16.]],

       [[16.,  9.,  9.,  9.,  9.,  9.,  9.,  1.],
        [ 4.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
        [ 4.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
        [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  4.],
        [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  9.]],

       [[ 1.,  9.,  9.,  9.,  9.,  9.,  9., 16.],
        [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  4.],
        [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  4.],
        [ 4.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
        [ 9.,  1.,  1.,  1.,  1.,  1.,  1.,  0.]]])
     */
    const int m = 5;
    const int n = 8;
    const int nX = 2;
    const int nY = 2;
    const int k = 1;
    float *X =
        new float[nX * m * k]{1.0, 2.0, 3.0, 3.0, 5.0, 5.0, 3.0, 3.0, 2.0, 1.0};
    float *Y = new float[nY * n * k]{1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0,
                                     4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0};
    float *D = new float[m * nX * n * nY]{0};
    float *dX;
    float *dY;
    float *dD;
    size_t size_mx = nX * m * sizeof(float);
    size_t size_ny = nY * n * sizeof(float);
    size_t size_mnxy = nX * m * size_ny;
    cudaMalloc(&dD, size_mnxy);
    cudaMalloc(&dX, size_mx * k);
    cudaMalloc(&dY, size_ny * k);
    cudaMemset(dD, 0, size_mnxy);
    cudaMemcpy(dX, X, size_mx * k, cudaMemcpyHostToDevice);
    cudaMemcpy(dY, Y, size_ny * k, cudaMemcpyHostToDevice);
    sq_euclid_dist_multi(dX, dY, dD, nX, nY, m, n, k);
    cudaErrchk(cudaMemcpy(D, dD, size_mnxy, cudaMemcpyDeviceToHost));

    float D_exp[m * n * nX * nY]{0, 1, 1, 1, 1, 1, 1, 9,  // dist(X[0], Y[0])
                                 1, 0, 0, 0, 0, 0, 0, 4,  //
                                 4, 1, 1, 1, 1, 1, 1, 1,  //
                                 4, 1, 1, 1, 1, 1, 1, 1,  //
                                 16, 9, 9, 9, 9, 9, 9, 1, //
                                                          //
                                 9, 1, 1, 1, 1, 1, 1, 0,  // dist(X[0], Y[1])
                                 4, 0, 0, 0, 0, 0, 0, 1,  //
                                 1, 1, 1, 1, 1, 1, 1, 4,  //
                                 1, 1, 1, 1, 1, 1, 1, 4,  //
                                 1, 9, 9, 9, 9, 9, 9, 16, //
                                                          //
                                 16, 9, 9, 9, 9, 9, 9, 1, // dist(X[1], Y[0])
                                 4, 1, 1, 1, 1, 1, 1, 1,  //
                                 4, 1, 1, 1, 1, 1, 1, 1,  //
                                 1, 0, 0, 0, 0, 0, 0, 4,  //
                                 0, 1, 1, 1, 1, 1, 1, 9,  //
                                                          //
                                 1, 9, 9, 9, 9, 9, 9, 16, // dist(X[1], Y[1])
                                 1, 1, 1, 1, 1, 1, 1, 4,  //
                                 1, 1, 1, 1, 1, 1, 1, 4,  //
                                 4, 0, 0, 0, 0, 0, 0, 1,  //
                                 9, 1, 1, 1, 1, 1, 1, 0};

    for (int i = 0; i < m * n * nX * nY; i++)
    {
        REQUIRE(is_close(D_exp[i], D[i]));
    }
    delete[] X;
    delete[] Y;
    delete[] D;
    cudaFree(dX);
    cudaFree(dY);
    cudaFree(dD);
}

TEST_CASE("Test squared Euclidan distance multi nX is 2 ny is 1")
{
    const int m = 5;
    const int n = 8;
    const int nX = 2;
    const int nY = 1;
    const int k = 1;
    float *X =
        new float[nX * m * k]{1.0, 2.0, 3.0, 3.0, 5.0, 1.0, 2.0, 3.0, 3.0, 5.0};
    float *Y = new float[nY * n * k]{1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0};
    float *D = new float[m * nX * n * nY]{0};
    float *dX;
    float *dY;
    float *dD;
    size_t size_mx = nX * m * sizeof(float);
    size_t size_ny = nY * n * sizeof(float);
    size_t size_mnxy = nX * m * size_ny;
    cudaMalloc(&dD, size_mnxy);
    cudaMalloc(&dX, size_mx * k);
    cudaMalloc(&dY, size_ny * k);
    cudaMemset(dD, 0, size_mnxy);
    cudaMemcpy(dX, X, size_mx * k, cudaMemcpyHostToDevice);
    cudaMemcpy(dY, Y, size_ny * k, cudaMemcpyHostToDevice);
    sq_euclid_dist_multi(dX, dY, dD, nX, nY, m, n, k);
    cudaErrchk(cudaMemcpy(D, dD, size_mnxy, cudaMemcpyDeviceToHost));

    float D_exp[m * n * nX * nY]{0, 1, 1, 1, 1, 1, 1, 9,  // dist(X[0], Y[0])
                                 1, 0, 0, 0, 0, 0, 0, 4,  //
                                 4, 1, 1, 1, 1, 1, 1, 1,  //
                                 4, 1, 1, 1, 1, 1, 1, 1,  //
                                 16, 9, 9, 9, 9, 9, 9, 1, //
                                                          //
                                 0, 1, 1, 1, 1, 1, 1, 9,  // dist(X[1], Y[0])
                                 1, 0, 0, 0, 0, 0, 0, 4,  //
                                 4, 1, 1, 1, 1, 1, 1, 1,  //
                                 4, 1, 1, 1, 1, 1, 1, 1,  //
                                 16, 9, 9, 9, 9, 9, 9, 1};
    // print_matrix(D, m * nX * nY, n);
    for (int i = 0; i < m * n * nX * nY; i++)
    {
        REQUIRE(is_close(D_exp[i], D[i]));
    }
    delete[] X;
    delete[] Y;
    delete[] D;
    cudaFree(dX);
    cudaFree(dY);
    cudaFree(dD);
}

TEST_CASE("soft dtw cuda multi nX is 1 nY is 1")
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

    float costs = 0;
    softdtw_cuda_naive_multi(D, R, &costs, 1, m, n, gamma);
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
    REQUIRE(is_close(2.80539, costs));
    delete[] a;
    delete[] b;
    cudaFree(D);
    cudaFree(da);
    cudaFree(db);
    cudaFree(R);
}

TEST_CASE("soft dtw cuda multi nX is 2 nY is 2")
{
    const int m = 5;
    const int k = 1;
    const int n = 8;
    const int nX = 2;
    const int nY = 2;
    const float gamma = 0.1;
    float *a =
        new float[nX * m * k]{1.0, 2.0, 3.0, 3.0, 5.0, 5.0, 3.0, 3.0, 2.0, 1.0};
    float *b = new float[nY * n * k]{1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0,
                                     4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0};
    // device arrays
    float *D;
    float *da;
    float *db;
    cudaMalloc(&da, m * nX * sizeof(float));
    cudaMalloc(&db, n * nY * sizeof(float));
    cudaMalloc(&D, m * n * nX * nY * sizeof(float));
    cudaMemset(D, 0, m * n * nX * nY * sizeof(float));
    cudaMemcpy(da, a, m * nX * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, n * nY * sizeof(float), cudaMemcpyHostToDevice);

    float *R;
    size_t m2n2 = nX * nY * (m + 2) * (n + 2);
    size_t sz_R = m2n2 * sizeof(float);
    cudaMalloc(&R, sz_R);

    sq_euclid_dist_multi(da, db, D, nX, nY, m, n, k);

    // float hD[m * n * nX * nY];
    // cudaMemcpy(&hD, D, m * n * nX * nY * sizeof(float),
    // cudaMemcpyDeviceToHost); print_matrix(hD, m * nX * nY, n);

    float costs[nX * nY];
    softdtw_cuda_naive_multi(D, R, (float *)&costs, nX * nY, m, n, gamma);
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
    // print_matrix(R, nX * nY * (m + 2), n + 2);
    float hR[m2n2]{0};
    cudaMemcpy(hR, R, m2n2 * sizeof(float), cudaMemcpyDeviceToHost);
    REQUIRE(is_close(2.80539, costs[0]));
    REQUIRE(is_close(26.86135, costs[1]));
    REQUIRE(is_close(26.86135, costs[2]));
    REQUIRE(is_close(2.80539, costs[3]));
    delete[] a;
    delete[] b;
    cudaFree(D);
    cudaFree(da);
    cudaFree(db);
    cudaFree(R);
}

TEST_CASE("soft dtw cuda multi nX is 2 nY is 1")
{
    const int m = 5;
    const int k = 1;
    const int n = 8;
    const int nX = 2;
    const int nY = 1;
    const float gamma = 0.1;
    float *a =
        new float[nX * m * k]{1.0, 2.0, 3.0, 3.0, 5.0, 1.0, 2.0, 3.0, 3.0, 5.0};
    float *b = new float[nY * n * k]{1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0};
    // device arrays
    float *D;
    float *da;
    float *db;
    cudaMalloc(&da, m * nX * sizeof(float));
    cudaMalloc(&db, n * nY * sizeof(float));
    cudaMalloc(&D, m * n * nX * nY * sizeof(float));
    cudaMemset(D, 0, m * n * nX * nY * sizeof(float));
    cudaMemcpy(da, a, m * nX * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, n * nY * sizeof(float), cudaMemcpyHostToDevice);

    float *R;
    size_t m2n2 = nX * nY * (m + 2) * (n + 2);
    size_t sz_R = m2n2 * sizeof(float);
    cudaMalloc(&R, sz_R);

    sq_euclid_dist_multi(da, db, D, nX, nY, m, n, k);

    float hD[m * n * nX * nY];
    cudaMemcpy(&hD, D, m * n * nX * nY * sizeof(float), cudaMemcpyDeviceToHost);
    print_matrix(hD, m * nX * nY, n);

    float costs[nX * nY];
    softdtw_cuda_naive_multi(D, R, (float *)&costs, nX * nY, m, n, gamma);
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
    // print_matrix(costs, nX, nY);
    float hR[m2n2]{0};
    cudaMemcpy(hR, R, m2n2 * sizeof(float), cudaMemcpyDeviceToHost);
    // print_matrix(hR, nX * nY * (m + 2), n + 2);
    REQUIRE(is_close(2.80539, costs[0]));
    REQUIRE(is_close(2.80539, costs[1]));
    delete[] a;
    delete[] b;
    cudaFree(D);
    cudaFree(da);
    cudaFree(db);
    cudaFree(R);
}

TEST_CASE("soft dtw cuda stencil D1")
{
    const int m = 5;
    const int k = 1;
    const int n = 8;
    const int nX = 1;
    const int nY = 1;
    const float gamma = 0.1;
    float *a = new float[nX * m * k]{1.0, 2.0, 3.0, 3.0, 5.0};
    float *b = new float[nY * n * k]{1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0};
    // device arrays
    float *D;
    float *da;
    float *db;
    cudaMalloc(&da, m * nX * sizeof(float));
    cudaMalloc(&db, n * nY * sizeof(float));
    cudaMalloc(&D, m * n * nX * nY * sizeof(float));
    cudaMemset(D, 0, m * n * nX * nY * sizeof(float));
    cudaMemcpy(da, a, m * nX * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, n * nY * sizeof(float), cudaMemcpyHostToDevice);

    float *R;
    size_t m2n2 = nX * nY * (m + 2) * (n + 2);
    size_t sz_R = m2n2 * sizeof(float);
    cudaMalloc(&R, sz_R);

    sq_euclid_dist_multi(da, db, D, nX, nY, m, n, k);

    // float hD[m * n * nX * nY];
    // cudaMemcpy(&hD, D, m * n * nX * nY * sizeof(float),
    // cudaMemcpyDeviceToHost); print_matrix(hD, m * nX * nY, n);

    float costs[nX * nY]{0};
    // softdtw_cuda_stencil(D, R, (float *)&costs, 1, m, n, gamma);
    softdtw_cuda_stencil(D, R, (float *)&costs, nX * nY, m, n, gamma);
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
    // print_matrix(costs, nX, nY);
    float hR[m2n2]{0};
    cudaMemcpy(hR, R, m2n2 * sizeof(float), cudaMemcpyDeviceToHost);
    // print_matrix(hR, nX * nY * (m + 2), n + 2);
    // print_matrix(hR, (m + 2), n + 2);
    REQUIRE(is_close(2.80539, costs[0]));
    delete[] a;
    delete[] b;
    cudaFree(D);
    cudaFree(da);
    cudaFree(db);
    cudaFree(R);
}

TEST_CASE("soft dtw cuda stencil D2")
{
    const int m = 5;
    const int k = 1;
    const int n = 8;
    const int nX = 2;
    const int nY = 1;
    const float gamma = 0.1;
    float *a =
        new float[nX * m * k]{1.0, 2.0, 3.0, 3.0, 5.0, 1.0, 2.0, 3.0, 3.0, 5.0};
    float *b = new float[nY * n * k]{1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0};
    // device arrays
    float *D;
    float *da;
    float *db;
    cudaMalloc(&da, m * nX * sizeof(float));
    cudaMalloc(&db, n * nY * sizeof(float));
    cudaMalloc(&D, m * n * nX * nY * sizeof(float));
    cudaMemset(D, 0, m * n * nX * nY * sizeof(float));
    cudaMemcpy(da, a, m * nX * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, n * nY * sizeof(float), cudaMemcpyHostToDevice);

    float *R;
    size_t m2n2 = nX * nY * (m + 2) * (n + 2);
    size_t sz_R = m2n2 * sizeof(float);
    cudaMalloc(&R, sz_R);

    sq_euclid_dist_multi(da, db, D, nX, nY, m, n, k);

    // float hD[m * n * nX * nY];
    // cudaMemcpy(&hD, D, m * n * nX * nY * sizeof(float),
    // cudaMemcpyDeviceToHost); print_matrix(hD, m * nX * nY, n);

    float costs[nX * nY]{0};
    // softdtw_cuda_stencil(D, R, (float *)&costs, 1, m, n, gamma);
    softdtw_cuda_stencil(D, R, (float *)&costs, nX * nY, m, n, gamma);
    /*
R expected:
[0. inf     inf     inf     inf     inf     inf     inf       inf inf] [inf
0.      1.      2.      3.      4.      5.      6.       15.         inf]
[inf  1.     -0.     -0.     -0.     -0.     -0.     -0.        4. inf]
[inf  5.      1.      0.9307  0.9307  0.9307  0.9307  0.9307    1. inf]
[inf  9.      2.      1.8901  1.8614  1.8613  1.8613  1.8613    1.8901 inf]
[inf 25.     11.     10.8614 10.8054 10.792  10.792  10.792     2.8054 inf]
[inf inf     inf     inf     inf     inf     inf     inf           inf inf]
    */
    // print_matrix(costs, nX, nY);
    float hR[m2n2]{0};
    cudaMemcpy(hR, R, m2n2 * sizeof(float), cudaMemcpyDeviceToHost);
    // print_matrix(hR, nX * nY * (m + 2), n + 2);
    REQUIRE(is_close(2.80539, costs[0]));
    REQUIRE(is_close(2.80539, costs[1]));
    delete[] a;
    delete[] b;
    cudaFree(D);
    cudaFree(da);
    cudaFree(db);
    cudaFree(R);
}

// TEST_CASE("soft dtw cuda stencil D4") // TODO: FIX FAILING TEST
// {
//     const int m = 5;
//     const int k = 1;
//     const int n = 8;
//     const int nX = 2;
//     const int nY = 2;
//     const float gamma = 0.1;
//     float *a =
//         new float[nX * m *
//         k]{1.0, 2.0, 3.0, 3.0, 5.0, 5.0, 3.0, 3.0, 2.0, 1.0};
//     float *b = new float[nY * n * k]{1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0,
//                                      4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0};
//     // device arrays
//     float *D;
//     float *da;
//     float *db;
//     cudaMalloc(&da, m * nX * sizeof(float));
//     cudaMalloc(&db, n * nY * sizeof(float));
//     cudaMalloc(&D, m * n * nX * nY * sizeof(float));
//     cudaMemset(D, 0, m * n * nX * nY * sizeof(float));
//     cudaMemcpy(da, a, m * nX * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(db, b, n * nY * sizeof(float), cudaMemcpyHostToDevice);

//     float *R;
//     size_t m2n2 = nX * nY * (m + 2) * (n + 2);
//     size_t sz_R = m2n2 * sizeof(float);
//     cudaMalloc(&R, sz_R);

//     sq_euclid_dist_multi(da, db, D, nX, nY, m, n, k);

//     // float hD[m * n * nX * nY];
//     // cudaMemcpy(&hD, D, m * n * nX * nY * sizeof(float),
//     // cudaMemcpyDeviceToHost); print_matrix(hD, m * nX * nY, n);

//     float costs[nX * nY]{0};
//     // softdtw_cuda_stencil(D, R, (float *)&costs, 1, m, n, gamma);
//     softdtw_cuda_stencil(D, R, (float *)&costs, nX * nY, m, n, gamma);
//     /*
// R expected:
// [0. inf     inf     inf     inf     inf     inf     inf       inf inf] [inf
// 0.      1.      2.      3.      4.      5.      6.       15.         inf]
// [inf  1.     -0.     -0.     -0.     -0.     -0.     -0.        4. inf]
// [inf  5.      1.      0.9307  0.9307  0.9307  0.9307  0.9307    1. inf]
// [inf  9.      2.      1.8901  1.8614  1.8613  1.8613  1.8613    1.8901 inf]
// [inf 25.     11.     10.8614 10.8054 10.792  10.792  10.792     2.8054 inf]
// [inf inf     inf     inf     inf     inf     inf     inf           inf inf]
//     */
//     print_matrix(costs, nX, nY);
//     float hR[m2n2]{0};
//     cudaMemcpy(hR, R, m2n2 * sizeof(float), cudaMemcpyDeviceToHost);
//     print_matrix(hR, nX * nY * (m + 2), n + 2);
//     // print_matrix(hR, (m + 2), n + 2);
//     REQUIRE(is_close(2.80539, costs[0]));
//     REQUIRE(is_close(26.86135, costs[1]));
//     REQUIRE(is_close(26.86135, costs[2]));
//     REQUIRE(is_close(2.80539, costs[3]));
//     delete[] a;
//     delete[] b;
//     cudaFree(D);
//     cudaFree(da);
//     cudaFree(db);
//     cudaFree(R);
// }
