/** Main file for running performance experiments on test data
 *  @file soft_dtw_perf_main.cpp
 */
#include "soft_dtw.cuh"
#include <chrono> // timing
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std::chrono;

/** Host function to record the performance of each kernel
 *  on a given dataset
 *  @param X A vector of time series dataset with lenght m
 *  @param time_series_lenght The length of each series in X
 *  @param count The number of time series in Dataset
 *  @param filename The output CSV file, storing the timing
 */
__host__ void comparison(std::vector<float> X, int time_series_lenght,
                         int count, char *filename)
{
    std::ofstream output_file;
    output_file.open(filename, std::ios_base::out);

    if (!output_file.is_open())
    {
        std::cerr << "Unable to open file " << filename << "\n";
        return;
    }

    const int k = 1;
    for (int i{0}; i < count; i++)
    {
        float *a = &X[i];
        // memcpy(&X[X.size() - dataArraySize], &dataArray[0], dataArraySize *
        // sizeof(int));

        const int m = time_series_lenght;

        for (int j{i + 1}; j < count; j++)
        {

            float gamma = 0.1;
            float *b = &X[j];
            const int n = time_series_lenght;
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

            // the pairwise squared Euclidean distances kernel execution
            // .....timing....
            std::cout << "STARTING squared Euclidean distances" << std::endl;
            auto sq_euclid_dist_start =
                std::chrono::high_resolution_clock::now();

            sq_euclid_dist(da, db, D, m, n, k);

            // since the kernels are executed asynchronously, need to sync so
            // that we can get accurate timing
            cudaDeviceSynchronize();
            auto sq_euclid_dist_end = std::chrono::high_resolution_clock::now();
            std::cout << "FINISHED squared Euclidean distances" << std::endl;
            auto sq_euclid_dist_duration =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    sq_euclid_dist_end - sq_euclid_dist_start)
                    .count();

            // the softdtw cuda naive kernel execution .....timing....
            std::cout << "STARTING softdtw cuda naive" << std::endl;
            auto softdtw_cuda_naive_start =
                std::chrono::high_resolution_clock::now();

            softdtw_cuda_naive(D, R, m, n, gamma);

            // since the kernels are executed asynchronously, need to sync so
            // that we can get accurate timing
            cudaDeviceSynchronize();
            auto softdtw_cuda_naive_end =
                std::chrono::high_resolution_clock::now();
            std::cout << "FINISHED softdtw cuda naive" << std::endl;
            auto softdtw_cuda_naive_duration =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    softdtw_cuda_naive_end - softdtw_cuda_naive_start)
                    .count();

            // the softdtw grad cuda naive kernel execution.....timing....
            std::cout << "STARTING softdtw grad cuda naive" << std::endl;
            auto softdtw_grad_cuda_naive_start =
                std::chrono::high_resolution_clock::now();

            softdtw_grad_cuda_naive(D, R, dE, m, n, gamma);

            // since the kernels are executed asynchronously, need to sync so
            // that we can get accurate timing
            cudaDeviceSynchronize();
            auto softdtw_grad_cuda_naive_end =
                std::chrono::high_resolution_clock::now();
            std::cout << "FINISHED softdtw grad cuda naive" << std::endl;
            auto softdtw_grad_cuda_naive_duration =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    softdtw_grad_cuda_naive_end - softdtw_grad_cuda_naive_start)
                    .count();

            cudaMemcpy(E, dE, m * n * sizeof(float), cudaMemcpyDeviceToHost);

            std::cout << " i, j  squared Euclidean distances Execution Time , "
                         "softdtw cuda naive Execution Time, softdtw grad cuda "
                         "naive\n";
            std::cout << i << ", " << j << " , " << sq_euclid_dist_duration
                      << " , " << softdtw_cuda_naive_duration << " , "
                      << softdtw_grad_cuda_naive_duration << '\n';

            output_file << i << ", " << j << " , " << sq_euclid_dist_duration
                        << "," << softdtw_cuda_naive_duration << ","
                        << softdtw_grad_cuda_naive_duration << '\n';

            // delete[] a;
            // delete[] b;
            delete[] E;
            cudaFree(da);
            cudaFree(db);
            cudaFree(D);
            cudaFree(R);
            cudaFree(dE);
        }
    }

    output_file.close();
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
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " INPUT_FILENAME "
                  << " OUTPUT_FILENAME \n";
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
    comparison(data_vec, m, n, argv[2]);

    return 0;
}
