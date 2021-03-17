#include "soft_dtw.cuh"
#include <chrono> // timing
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>
#include <iterator>

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
__host__ void comparison(std::vector<float> X , std::vector<float> Y, int m , int n , int count)
{
    //const int k = 1;
    for (int i = 0; i < count; i++)
    {
        float *a = &X[i];

        //for (int h=0; h <m * count ; h++)
        // std::cout << a[h]<< std::endl;

       // &X[i];
       // std::cerr << " a done "<< std::endl;
        
        for (int j =0 ; j < count; j++)
        {
            //std::cerr << "round i =" << i << " , j = " << j << std::endl;

            float gamma = 0.1;
            float *b = &Y[j];         
            //std::cerr << " b done "<< std::endl;
            // distance matrix
            float *d;
            d = (float *)malloc(m * n * sizeof(float));
            // device arrays
            float *da;
            cudaMalloc(&da, m * sizeof(float));
            cudaMemcpy(da, a, m * sizeof(float), cudaMemcpyHostToDevice);

            //std::cerr << " da done "<< std::endl;

            float *db;
            cudaMalloc(&db, n * sizeof(float));
            //std::cerr << " db done 1"<< std::endl;
            cudaMemcpy(db, b, n * sizeof(float), cudaMemcpyHostToDevice);
            
            //std::cerr << " db done "<< std::endl;

            float *dd;
            cudaMalloc(&dd, m * n * sizeof(float));
            
            //std::cerr << " dd done "<< std::endl;

            // TODO: //need to check again
            // initializing distance matrix with 0
            //for (int i = 0; i < m * n; i++)
             //   d[i] = 0.0f;


            //Start Soft DTW for tiled multi kernel
            //TODO: check with different tile_size and see the perfromance
            //TODO: just need to change the tile kernel ofr shared memory size
            // base on tile width and height defined here
            uint tile_width = 16;
            //uint tile_height = 16;
            uint total_tiles_columns = (m + tile_width - 1) / tile_width;
            uint total_tiles_rows = (n + tile_width - 1) / tile_width;
            uint total_tiles_waves = total_tiles_columns + total_tiles_rows - 1;
            uint min_tiles = std::min(total_tiles_columns, total_tiles_rows);

           

            // the softdtw cuda multi tiled kernel execution .....timing....
            auto start = high_resolution_clock::now();
            soft_dtw_tiled(da, db,dd, tile_width, total_tiles_waves,
                                total_tiles_columns, total_tiles_rows, min_tiles,
                                gamma);
            cudaDeviceSynchronize();
            auto end = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(end - start).count();
            std::cout << "soft_dtw_tiled m =" << m << " n =" << n  << " duration =" << duration << std::endl;
            cudaMemcpy(d, dd, m * n * sizeof(float), cudaMemcpyDeviceToHost);

            delete[] d;

            cudaFree(da);
            cudaFree(db);
            cudaFree(dd);


        }
    }
}

/** Fill a vector with n random floats drawn from unit normal distribution.
 */
std::vector<float> fill_random( std::vector<float> vec, uint n)
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
    uint m = 0; // length of time series
    uint n = 0; // number of time series
    uint k = 0; // number of time series
    
    if (filename == "random")
    {
        if (argc < 4)
        {
            std::cerr << "Usage: " << argv[0] << " random [first-length] [second-length] [count]\n";
            return 1;
        }
        m = atol(argv[2]);
        n = atol(argv[3]);
        k = atol(argv[4]);

        
        data_vec_a = fill_random(data_vec_a, m * k);
        


        data_vec_b =fill_random(data_vec_b, n * k);       
        comparison(data_vec_a , data_vec_b, m, n , k);
        return 0;
    }

    std::ifstream input_file(filename);

    if (!input_file.is_open())
    {
        std::cerr << "Unable to open file " << argv[1] << "\n";
        return 1;
    }

    // std::string str_buf;
    // std::stringstream ss;
    // float float_buf;

    // while (!input_file.eof())
    // {
    //     getline(input_file, str_buf);
    //     ss.str(str_buf);
    //     // first element per line is a class label not a data point.
    //     bool is_data = false;
    //     while (!ss.eof())
    //     {
    //         ss >> float_buf;
    //         if (is_data)
    //         {
    //             data_vec.push_back(float_buf);
    //         }
    //         is_data = true;
    //     }
    //     ss.clear();
    //     n++;
    // }
    // n--;
    // m = data_vec.size() / n;
    // // n will overcount by 1 line when we reach the end.
    // std::cout << "Data file " << argv[1] << " contains " << n
    //           << " time series of length " << m << "\n";

    // // Get a pointer to the array data which is dimension (m x n)

    // // Let's start checking the performance
    // comparison(data_vec, m, n , k);

    return 0;
}
