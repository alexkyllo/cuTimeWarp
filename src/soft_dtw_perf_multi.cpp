/** Performance testing program for multi-block CUDA Soft-DTW kernels
 * @file soft_dtw_perf_multi.cpp
 * @author Alex Kyllo
 * @date 2021-03-17
 */
#include "soft_dtw.cuh"
#include <chrono> // timing
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

using namespace std::chrono;
typedef unsigned int uint;

#define cudaErrchk(ans)                                                        \
    {                                                                          \
        GPUAssert((ans), __FILE__, __LINE__);                                  \
    }
inline void GPUAssert(cudaError_t code, const char *file, int line,
                      bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUError: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
        {
            exit(code);
        }
    }
}

/** Host function to record the performance of each kernel
 *  on a given dataset
 *  @param X A vector of time series dataset with lenght m
 *  @param time_series_lenght The length of each series in X
 *  @param count The number of time series in Dataset
 */
__host__ void comparison(std::vector<float> X, int time_series_len, int count)
{
    // Soft-DTW smoothing param
    float gamma = 0.1;
    // univariate time series
    const uint k = 1;
    // compare the same batch to itself
    const uint m = time_series_len;
    const uint n = time_series_len;
    const uint nX = count;
    const uint nY = count;
    float *dX;
    float *dY;
    float *dD;

    cudaErrchk(cudaMalloc(&dD, m * nX * n * nY * sizeof(float)));
    cudaErrchk(cudaMalloc(&dX, m * k * nX));
    cudaErrchk(cudaMalloc(&dY, n * k * nY));
    cudaMemset(dD, 0, m * nX * n * nY * sizeof(float));
    cudaMemcpy(dX, &X[0], m * k * nX * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dY, &X[0], n * k * nX * sizeof(float), cudaMemcpyHostToDevice);

    float *dR;
    size_t m2n2 = count * count * (m + 2) * (n + 2);
    size_t sz_R = m2n2 * sizeof(float);
    cudaErrchk(cudaMalloc(&dR, sz_R));
    cudaMemset(dR, 0, sz_R);

    // std::cout << "kernel length count microseconds\n";
    // the pairwise squared Euclidean distances kernel execution
    auto start = high_resolution_clock::now();
    sq_euclid_dist_multi(dX, dY, dD, nX, nY, m, n, k);
    cudaErrchk(cudaDeviceSynchronize());
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();
    std::cout << "sq_euclid_dist_multi " << m << " " << nX << " " << duration
              << std::endl;

    // now that dD is constructed we don't need dX and dY anymore
    cudaFree(dX);
    cudaFree(dY);

    // the softdtw cuda naive kernel execution .....timing....
    float *costs = new float[nX * nY]{0};
    start = high_resolution_clock::now();
    softdtw_cuda_naive_multi(dD, dR, costs, nX * nY, m, n, gamma);
    cudaErrchk(cudaDeviceSynchronize());
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start).count();
    std::cout << "softdtw_cuda_naive_multi " << m << " " << nX << " "
              << duration << std::endl;
    // zero out costs so we can reuse it
    memset(costs, 0, nX * nY * sizeof(float));

    // the softdtw cuda naive kernel execution bandwidth = 80% of m
    uint bw = floor(0.8 * m);
    start = high_resolution_clock::now();
    softdtw_cuda_naive_multi(dD, dR, costs, nX * nY, m, n, gamma, bw);
    cudaErrchk(cudaDeviceSynchronize());
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start).count();
    std::cout << "softdtw_cuda_naive_multi_bw_80 " << m << " " << nX << " "
              << duration << std::endl;
    memset(costs, 0, nX * nY * sizeof(float));

    // the softdtw cuda naive kernel execution bandwidth = 60%
    bw = floor(0.6 * m);
    start = high_resolution_clock::now();
    softdtw_cuda_naive_multi(dD, dR, costs, nX * nY, m, n, gamma, bw);
    cudaErrchk(cudaDeviceSynchronize());
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start).count();
    std::cout << "softdtw_cuda_naive_multi_bw_60 " << m << " " << nX << " "
              << duration << std::endl;
    memset(costs, 0, nX * nY * sizeof(float));

    // the softdtw cuda naive kernel execution bandwidth = 40%
    bw = floor(0.4 * m);
    start = high_resolution_clock::now();
    softdtw_cuda_naive_multi(dD, dR, costs, nX * nY, m, n, gamma, bw);
    cudaErrchk(cudaDeviceSynchronize());
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start).count();
    std::cout << "softdtw_cuda_naive_multi_bw_40 " << m << " " << nX << " "
              << duration << std::endl;
    memset(costs, 0, nX * nY * sizeof(float));

    // the softdtw cuda naive kernel execution bandwidth = 20%
    bw = floor(0.2 * m);
    start = high_resolution_clock::now();
    softdtw_cuda_naive_multi(dD, dR, costs, nX * nY, m, n, gamma, bw);
    cudaErrchk(cudaDeviceSynchronize());
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start).count();
    std::cout << "softdtw_cuda_naive_multi_bw_20 " << m << " " << nX << " "
              << duration << std::endl;
    memset(costs, 0, nX * nY * sizeof(float));

    // the softdtw cuda stencil kernel execution .....timing....
    start = high_resolution_clock::now();
    softdtw_cuda_stencil(dD, dR, costs, nX * nY, m, n, gamma);
    cudaErrchk(cudaDeviceSynchronize());
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start).count();
    std::cout << "softdtw_cuda_stencil_multi " << m << " " << nX << " "
              << duration << std::endl;
    memset(costs, 0, nX * nY * sizeof(float));

    // the softdtw cuda stencil kernel execution .....timing....
    bw = floor(0.8 * m);
    start = high_resolution_clock::now();
    softdtw_cuda_stencil(dD, dR, costs, nX * nY, m, n, gamma, bw);
    cudaErrchk(cudaDeviceSynchronize());
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start).count();
    std::cout << "softdtw_cuda_stencil_multi_80 " << m << " " << nX << " "
              << duration << std::endl;
    memset(costs, 0, nX * nY * sizeof(float));

    // the softdtw cuda stencil kernel execution .....timing....
    bw = floor(0.6 * m);
    start = high_resolution_clock::now();
    softdtw_cuda_stencil(dD, dR, costs, nX * nY, m, n, gamma, bw);
    cudaErrchk(cudaDeviceSynchronize());
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start).count();
    std::cout << "softdtw_cuda_stencil_multi_60 " << m << " " << nX << " "
              << duration << std::endl;
    memset(costs, 0, nX * nY * sizeof(float));

    // the softdtw cuda stencil kernel execution .....timing....
    bw = floor(0.4 * m);
    start = high_resolution_clock::now();
    softdtw_cuda_stencil(dD, dR, costs, nX * nY, m, n, gamma, bw);
    cudaErrchk(cudaDeviceSynchronize());
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start).count();
    std::cout << "softdtw_cuda_stencil_multi_40 " << m << " " << nX << " "
              << duration << std::endl;
    memset(costs, 0, nX * nY * sizeof(float));

    // the softdtw cuda stencil kernel execution .....timing....
    bw = floor(0.2 * m);
    start = high_resolution_clock::now();
    softdtw_cuda_stencil(dD, dR, costs, nX * nY, m, n, gamma, bw);
    cudaErrchk(cudaDeviceSynchronize());
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start).count();
    std::cout << "softdtw_cuda_stencil_multi_40 " << m << " " << nX << " "
              << duration << std::endl;
    memset(costs, 0, nX * nY * sizeof(float));

    // the softdtw cuda stencil kernel execution .....timing....
    start = high_resolution_clock::now();
    softdtw_cuda_stencil(dD, dR, costs, nX * nY, m, n, gamma, 40);
    cudaErrchk(cudaDeviceSynchronize());
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start).count();
    std::cout << "softdtw_cuda_stencil_multi_20 " << m << " " << nX << " "
              << duration << std::endl;
    memset(costs, 0, nX * nY * sizeof(float));

    // converting D from row-major to diagonal-major format
    // Free the row-major R we aren't using anymore
    cudaFree(dR);
    float *dDD;
    uint nDD = std::min(m, n) * nX * nY * (m + n - 1);
    uint szDD = nDD * sizeof(float);
    cudaErrchk(cudaMalloc(&dDD, szDD));
    cudaMemset(dDD, 0, szDD);
    start = high_resolution_clock::now();
    convert_diagonal_major_multi(dD, dDD, nX * nY, m, n);
    cudaErrchk(cudaDeviceSynchronize());
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start).count();
    std::cout << "convert_diagonal_multi " << m << " " << nX << " " << duration
              << std::endl;
    // Free the row-major D we aren't using anymore
    cudaFree(dD);

    // transform R into diagonal-major layout
    float *dRD;
    uint nRD = (std::min(m, n) + 2) * nX * nY * (m + n + 3);
    uint szRD = nRD * sizeof(float);
    cudaErrchk(cudaMalloc(&dRD, szRD));
    cudaMemset(dRD, 0, szRD);

    // the softdtw cuda stencil kernel execution .....timing....
    start = high_resolution_clock::now();
    softdtw_cuda_diagonal_multi(dDD, dRD, costs, nX * nY, m, n, gamma);
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start).count();
    std::cout << "softdtw_cuda_diagonal_multi " << m << " " << nX << " "
              << duration << std::endl;
    memset(costs, 0, nX * nY * sizeof(float));

    delete[] costs;
    cudaFree(dDD);
    cudaFree(dRD);
}

/** Fill a vector with n random floats drawn from unit normal distribution.
 */
void fill_random(std::vector<float> vec, uint n)
{
    std::default_random_engine gen;
    std::normal_distribution<float> dist(0.0, 1.0);
    for (uint i = 0; i < n; i++)
    {
        vec.push_back(dist(gen));
    }
}

// To run as an example:
// make build
// ./bin/soft_dtw_perf data/ECG200/ECG200_TRAIN.txt
// output/ECG200/PERFORMANCE.CSV
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0]
                  << " [INPUT_FILENAME] | random [length] [count]\n";
        return 1;
    }

    std::vector<float> data_vec;
    std::string filename = argv[1];
    uint m = 0; // length of time series
    uint n = 0; // number of time series

    if (filename == "random")
    {
        if (argc < 4)
        {
            std::cerr << "Usage: " << argv[0] << " random [length] [count]\n";
            return 1;
        }
        m = atol(argv[2]);
        n = atol(argv[3]);
        if (m < 2 || n < 1)
        {
            std::cerr << "Input time series must have length at least 2 and "
                         "count at least 1.\n";
            return 1;
        }
        fill_random(data_vec, m * n);
        comparison(data_vec, m, n);
        return 0;
    }

    std::ifstream input_file(filename);

    if (!input_file.is_open())
    {
        std::cerr << "Unable to open file " << argv[1] << "\n";
        return 1;
    }

    std::string str_buf;
    std::stringstream ss;
    float float_buf;

    while (!input_file.eof())
    {
        getline(input_file, str_buf);
        ss.str(str_buf);
        // first element per line is a class label not a data point.
        bool is_data = false;
        while (!ss.eof())
        {
            ss >> float_buf;
            if (is_data)
            {
                data_vec.push_back(float_buf);
            }
            is_data = true;
        }
        ss.clear();
        n++;
    }
    n--;
    m = data_vec.size() / n;
    // n will overcount by 1 line when we reach the end.
    std::cerr << "Data file " << argv[1] << " contains " << n
              << " time series of length " << m << "\n";

    // Get a pointer to the array data which is dimension (m x n)

    // Let's start checking the performance
    comparison(data_vec, m, n);

    return 0;
}
