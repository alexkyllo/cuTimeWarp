#include "../src/soft_dtw.hcu"
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

TEST_CASE("Soft DTW CUDA compare with CPU version")
{
    // Generate two time series randomly
    // Compute Soft DTW distance with CPU
    // Compute with CUDA and the distance should be nearly equal
}
