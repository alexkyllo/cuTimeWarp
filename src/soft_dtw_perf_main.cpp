/** Main file for running performance experiments on test data
 *  @file soft_dtw_perf_main.cpp
 */
#include "soft_dtw.cuh"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

// To run as an example:
// make build
// ./bin/soft_dtw_perf data/ECG200/ECG200_TRAIN.txt
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
        std::cerr << "Usage: " << argv[0] << " FILENAME\n";
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
    float *X = &data_vec[0];

    // TODO: calculate and time SoftDTW between every time series and every
    // other time series in the dataset.

    return 0;
}
