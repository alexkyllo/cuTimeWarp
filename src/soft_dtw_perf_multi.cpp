#include "soft_dtw.cuh"
#include <chrono> // timing
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std::chrono;
typedef unsigned int uint;
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

    cudaMalloc(&dD, m * nX * n * nY * sizeof(float));
    cudaMalloc(&dX, m * k * nX);
    cudaMalloc(&dY, n * k * nY);
    cudaMemset(dD, 0, m * nX * n * nY * sizeof(float));
    cudaMemcpy(dX, &X[0], m * k * nX * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dY, &X[0], n * k * nX * sizeof(float), cudaMemcpyHostToDevice);

    float *dR;
    size_t m2n2 = count * count * (m + 2) * (n + 2);
    size_t sz_R = m2n2 * sizeof(float);
    cudaMalloc(&dR, sz_R);
    cudaMemset(dR, 0, sz_R);

    // the pairwise squared Euclidean distances kernel execution
    auto sq_euclid_dist_start = high_resolution_clock::now();

    sq_euclid_dist_multi(dX, dY, dD, nX, nY, m, n, k);

    cudaDeviceSynchronize();
    auto sq_euclid_dist_end = high_resolution_clock::now();
    auto sq_euclid_dist_duration =
        duration_cast<microseconds>(sq_euclid_dist_end - sq_euclid_dist_start)
            .count();

    std::cout << "sq_euclid_dist_multi " << sq_euclid_dist_duration
              << std::endl;

    // the softdtw cuda naive kernel execution .....timing....
    auto softdtw_cuda_naive_start = high_resolution_clock::now();

    float *costs = new float[nX * nY]{0};
    softdtw_cuda_naive_multi(dD, dR, costs, nX * nY, m, n, gamma);

    cudaDeviceSynchronize();
    auto softdtw_cuda_naive_end = high_resolution_clock::now();
    auto softdtw_cuda_naive_duration =
        duration_cast<microseconds>(softdtw_cuda_naive_end -
                                    softdtw_cuda_naive_start)
            .count();

    std::cout << "softdtw_cuda_naive_multi " << softdtw_cuda_naive_duration
              << std::endl;

    // zero out costs so we can reuse it
    memset(costs, 0, nX * nY * sizeof(float));
    // the softdtw cuda stencil kernel execution .....timing....
    auto softdtw_cuda_stencil_start = high_resolution_clock::now();
    softdtw_cuda_stencil(dD, dR, costs, nX * nY, m, n, gamma);

    cudaDeviceSynchronize();
    auto softdtw_cuda_stencil_end = high_resolution_clock::now();
    auto softdtw_cuda_stencil_duration =
        duration_cast<microseconds>(softdtw_cuda_stencil_end -
                                    softdtw_cuda_stencil_start)
            .count();

    std::cout << "softdtw_cuda_stencil_multi " << softdtw_cuda_stencil_duration
              << std::endl;
    delete[] costs;
    cudaFree(dX);
    cudaFree(dY);
    cudaFree(dD);
    cudaFree(dR);
}

// To run as an example:
// make build
// ./bin/soft_dtw_perf data/ECG200/ECG200_TRAIN.txt
// output/ECG200/PERFORMANCE.CSV
int main(int argc, char **argv)
{
    // TODO
    // Read in one or more input time series files (delimited format) into
    // arrays and allocate / copy to device
    // Launch function to calculate all soft-dtw pairwise distances and
    // time execution
    // Repeat for multiple input dataset sizes and univariate vs. multivariate
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " INPUT_FILENAME\n";
        return 1;
    }

    std::ifstream input_file(argv[1]);

    if (!input_file.is_open())
    {
        std::cerr << "Unable to open file " << argv[1] << "\n";
        return 1;
    }

    std::vector<float> data_vec;
    std::string str_buf;
    std::stringstream ss;
    float float_buf;
    uint m = 0; // length of time series
    uint n = 0; // number of time series
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
    std::cout << "Data file " << argv[1] << " contains " << n
              << " time series of length " << m << "\n";

    // Get a pointer to the array data which is dimension (m x n)

    // Let's start checking the performance
    comparison(data_vec, m, n);

    return 0;
}
