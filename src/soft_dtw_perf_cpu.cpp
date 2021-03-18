#include "soft_dtw_cpu.hpp"
#include <chrono> // timing
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

using namespace std::chrono;
typedef unsigned int uint;

/** Host function to record the performance of each kernel
 *  on a given dataset
 *  @param X A vector of time series dataset with lenght m
 *  @param time_series_length The length of each series in X
 *  @param count The number of time series in Dataset
 */
void time(std::vector<float> &vec, int time_series_len, int count)
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
    const size_t m2n2 = count * count * (m + 2) * (n + 2);
    float *X = &vec[0];
    float *Y = X;
    float *D = new float[m * nX * n * nY]{0};
    float *R = new float[m2n2]{0};

    // std::cout << "kernel length count microseconds\n";
    // the pairwise squared Euclidean distances kernel execution
    auto start = high_resolution_clock::now();
    for (uint i = 0; i < nX; i++)
    {
        for (uint j = 0; j < nY; j++)
        {
            sq_euclidean_distance<float>(&X[i * m * k], &Y[j * n * k],
                                         &D[i * m * n + j], m, n, k);
        }
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();
    std::cout << "sq_euclidean_distance " << m << " " << nX << " " << duration
              << std::endl;

    // the softdtw cuda naive CPU kernel execution .....timing....
    start = high_resolution_clock::now();
    for (uint i = 0; i < nX; i++)
    {
        for (uint j = 0; j < nY; j++)
        {
            uint offset = (i * nY + j) * m * n;
            softdtw<float>(&D[offset], &R[offset], m, n, gamma);
        }
    }
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start).count();
    std::cout << "softdtw " << m << " " << nX << " " << duration << std::endl;
    // zero out costs so we can reuse it
    delete[] D;
    delete[] R;
}

/** Fill a vector with n random floats drawn from unit normal distribution.
 */
void fill_random(std::vector<float> &vec, uint n)
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
        if (m < 2 || n < 2)
        {
            std::cerr << "Input time series must have length at least 2.\n";
            return 1;
        }
        fill_random(data_vec, m * n);
        time(data_vec, m, n);
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
    std::cout << "Data file " << argv[1] << " contains " << n
              << " time series of length " << m << "\n";

    // Let's start checking the performance
    time(data_vec, m, n);

    return 0;
}
