/** Performance testing program for tiled CUDA Soft-DTW
 * @file soft_dtw_perf_tiled.cpp
 * @author Afrooz Rahmati
 * @date 2021-03-17
 */
#include "soft_dtw.cuh"
#include <chrono> // timing
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <iterator>
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
__host__ void comparison(std::vector<float> X, std::vector<float> Y, int m,
                         int n, int count)
{
    // const int k = 1;
    for (int i = 0; i < count; i++)
    {
        float *a = &X[i];

        for (int j = 0; j < count; j++)
        {
            // std::cerr << "round i =" << i << " , j = " << j << std::endl;

            float gamma = 0.1;
            float *b = &Y[j];
            // std::cerr << " b done "<< std::endl;
            // distance matrix
            float *d;
            d = (float *)malloc(m * n * sizeof(float));
            // device arrays
            float *da;
            cudaMalloc(&da, m * sizeof(float));
            cudaMemcpy(da, a, m * sizeof(float), cudaMemcpyHostToDevice);

            // std::cerr << " da done "<< std::endl;

            float *db;
            cudaMalloc(&db, n * sizeof(float));
            // std::cerr << " db done 1"<< std::endl;
            cudaMemcpy(db, b, n * sizeof(float), cudaMemcpyHostToDevice);

            // std::cerr << " db done "<< std::endl;

            float *dd;
            cudaMalloc(&dd, m * n * sizeof(float));

            // std::cerr << " dd done "<< std::endl;

            // TODO: //need to check again
            // initializing distance matrix with 0
            // for (int i = 0; i < m * n; i++)
            //   d[i] = 0.0f;

            // Start Soft DTW for tiled multi kernel
            // TODO: check with different tile_size and see the perfromance
            // TODO: just need to change the tile kernel ofr shared memory size
            // base on tile width and height defined here
            uint tile_width = 16;
            // uint tile_height = 16;
            uint total_tiles_columns = (m + tile_width - 1) / tile_width;
            uint total_tiles_rows = (n + tile_width - 1) / tile_width;
            uint total_tiles_waves = total_tiles_columns + total_tiles_rows - 1;
            uint min_tiles = std::min(total_tiles_columns, total_tiles_rows);

            // the softdtw cuda multi tiled kernel execution .....timing....
            auto start = high_resolution_clock::now();
            soft_dtw_tiled(da, db, dd, tile_width, total_tiles_waves,
                           total_tiles_columns, total_tiles_rows, min_tiles,
                           gamma);
            cudaDeviceSynchronize();
            auto end = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(end - start).count();
            std::cout << "soft_dtw_tiled m =" << m << " n =" << n
                      << " duration =" << duration << std::endl;
            cudaMemcpy(d, dd, m * n * sizeof(float), cudaMemcpyDeviceToHost);

            delete[] d;

            cudaFree(da);
            cudaFree(db);
            cudaFree(dd);
        }
    }
}

__host__ void compare_two_sequence(std::vector<float> X, std::vector<float> Y,
                                   int m, int n)
{
    // std::cerr << "round i =" << i << " , j = " << j << std::endl;
    float *a = &X[0];
    float *b = &Y[0];
    float gamma = 0.1;

    // distance matrix
    float *d;
    d = (float *)malloc(m * n * sizeof(float));
    // device arrays
    float *da;
    cudaMalloc(&da, m * sizeof(float));
    cudaMemcpy(da, a, m * sizeof(float), cudaMemcpyHostToDevice);

    // std::cerr << " da done "<< std::endl;

    float *db;
    cudaMalloc(&db, n * sizeof(float));
    // std::cerr << " db done 1"<< std::endl;
    cudaMemcpy(db, b, n * sizeof(float), cudaMemcpyHostToDevice);

    // std::cerr << " db done "<< std::endl;

    float *dd;
    cudaMalloc(&dd, m * n * sizeof(float));

    // Start Soft DTW for tiled multi kernel
    // TODO: check with different tile_size and see the perfromance
    // TODO: just need to change the tile kernel ofr shared memory size
    // base on tile width and height defined here
    uint tile_width = 16;
    // uint tile_height = 16;
    uint total_tiles_columns = (m + tile_width - 1) / tile_width;
    uint total_tiles_rows = (n + tile_width - 1) / tile_width;
    uint total_tiles_waves = total_tiles_columns + total_tiles_rows - 1;
    uint min_tiles = std::min(total_tiles_columns, total_tiles_rows);

    // the softdtw cuda multi tiled kernel execution .....timing....
    auto start = high_resolution_clock::now();
    soft_dtw_tiled(da, db, dd, tile_width, total_tiles_waves,
                   total_tiles_columns, total_tiles_rows, min_tiles, gamma);
    cudaDeviceSynchronize();
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();
    std::cout << "soft_dtw_tiled " << m * n << " " << total_tiles_waves << " "
              << duration << std::endl;
    cudaMemcpy(d, dd, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    delete[] d;

    cudaFree(da);
    cudaFree(db);
    cudaFree(dd);
}

/** Fill a vector with n random floats drawn f unit normal distribution.
 */
std::vector<float> fill_random(std::vector<float> vec, uint n)
{
    std::default_random_engine gen;
    std::normal_distribution<float> dist(0.0, 1.0);
    for (uint i = 0; i < n; i++)
    {
        vec.push_back(dist(gen));
    }

    return vec;
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

    std::vector<float> data_vec_a;
    std::vector<float> data_vec_b;
    std::string filename = argv[1];
    int m = 0;   // length of time series
    int max = 0; // number of time series
    int k = 0;   // number of time series

    if (filename == "random")
    {
        if (argc < 5)
        {
            std::cerr << "Usage: " << argv[0]
                      << " random [start_length] [maximum_length] [interval]\n";
            return 1;
        }
        m = atol(argv[2]);
        max = atol(argv[3]);
        k = atol(argv[4]);

        // int i= m;
        // int j= m;
        // while (i <max)
        // {
        //     while (j <max)
        //     {
        //         std::cout<< " i = "<< i << " j= "<< j<<std::endl;
        //         data_vec_a = fill_random(data_vec_a, i );
        //         data_vec_b =fill_random(data_vec_b, j );
        //         compare_two_sequence(data_vec_a , data_vec_b, i, j);
        //         j= j+ k;

        //     }
        //     i= i+k;
        // }

        for (int i = m; i <= max; i += k)
            for (int j = i; j <= max; j += k)
            {
                // int i =1000;
                // int j =1000;

                data_vec_a = fill_random(data_vec_a, i);
                data_vec_b = fill_random(data_vec_b, j);
                compare_two_sequence(data_vec_a, data_vec_b, i, j);
                data_vec_a.clear();
                data_vec_b.clear();
            }
        return 0;
    }

    std::ifstream input_file(filename);

    if (!input_file.is_open())
    {
        std::cerr << "Unable to open file " << argv[1] << "\n";
        return 1;
    }

    return 0;
}
