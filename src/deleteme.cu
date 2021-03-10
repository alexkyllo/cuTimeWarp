#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <chrono> // timing
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std::chrono;












 // __global__ void softdtw_tiled_kernel(float *D, float *R, float *cost, uint m,
 //                                      uint n, float gamma)
 // {
 //     // TODO
 //     // Divide R into tiles
 //     // Each tile depends on the tiles to its top, left, and top-left
 //     // Assign one thread to spin on the signal variable for this tile
 //     // Process the tile diagonally from upper left to lower right
 //     // using a loop counter to keep track of fully processed diagonals
 //     // and while loop and syncthreads to spin on it
 //     // Write to the signal variables to signal the next tiles
 // }
 
 

 
 
 /** Kernel function for computing tiled Soft DTW using wave front approach
  * matrix for multivariate time series with CUDA. Input a , b and D should be a
  * __device__ array.
  * @param a The first time series data 
  * @param b The second time series data 
  * @param m Length of first time series
  * @param n Length of second time series
  * @param D The pairwise distance array of two time series
  * @param waveId the ID for the wave
  * @param total_tiles_rows
  * @param total_tiles_columns
  * @param tile_width
  * @param tileRow
  * @param tileColumn
  * @param gamma SoftDTW smoothing parameter
  */
 
 __device__ void softdtw_tiled_wavefront(float *a ,float *b ,float *D, int waveId, 
         uint total_tiles_rows, 
         uint total_tiles_columns , uint tile_width , uint tileRow , uint  tileColumn){
     
         
     //the main tile computation start here
     const int tid = threadIdx.x;
     const int tile_size = tile_width * tile_width;
     const int tileBaseIndex = tile_size * total_tiles_columns * tileRow
             + tile_size * tileColumn;
 
     const int nWaves = tile_width * 2 - 1;
 
     //TODO: check with 1024 
     __shared__ float seq1_local[512];
     __shared__ float seq2_local[512];
 
     __syncthreads();
 
     seq1_local[tid] = a[ tileColumn * tile_width + tid ];
     seq2_local[tid] = b[ tileRow * tile_width + tid ];
 
     __syncthreads();
 
     for (int waveId = 0; waveId < nWaves; waveId++) {
         const int row = waveId - tid;
         const int column = tid;
 
         const int index = tileBaseIndex + tile_width * column + row;
 
         if ( row >= 0 && row < tile_width && column >= 0 && column < tile_width )
         {
 
             const int cost = fabs(seq2_local[row] - seq1_local[column]);
 
             int upleft=0;
             int left = 0;
             int up = 0;
 
             // LEFT & UP
             if (tileColumn > 0 || column > 0) //left
             {
                 const int leftIndex = index - tile_width;
                 left = D[leftIndex] - 1;
             }
 
             if (tileRow > 0 || row > 0) {
                 int upIndex = index - 1;
                 if (row == 0) {
                     upIndex = tileBaseIndex - tile_size * total_tiles_columns;
                     upIndex += tile_width * (column + 1) - 1;
                 }
 
                 up = D[upIndex] - 1;
 
             }
 
             ////  UPLEFT //
             if ((tileColumn > 0 || column > 0) && (tileRow > 0 || row > 0)) {
                 int upLeftIndex = index - 1 - tile_width ;
                 if (row == 0) {
                     upLeftIndex = tileBaseIndex - tile_size * total_tiles_columns;
                     upLeftIndex += tile_width * (column) - 1;
                 }
 
                 upleft = D[upLeftIndex];
             }
 
             //TODO: should be change to softmin
             D[index] = ((int)cost/100)+min(upleft,min(left,up));
 
         }
 
         __syncthreads();        
     
     }
     
 }
 
 
 
 /** Kernel function for computing tiled Soft DTW using wave front approach
  * matrix for multivariate time series with CUDA. Input a , b and D should be a
  * __device__ array.
  * @param a The first time series data 
  * @param b The second time series data 
  * @param m Length of first time series
  * @param n Length of second time series
  * @param D The pairwise distance array of two time series
  * @param waveId the ID for the wave
  * @param total_tiles_rows
  * @param total_tiles_columns
  * @param tile_width
  * @param tileRow
  * @param tileColumn
  * @param gamma SoftDTW smoothing parameter
  */
 
  __global__ void softdtw_global_tiled(float *da ,float *db ,float *D, int waveId, 
    uint total_tiles_rows, 
    uint total_tiles_columns , uint tile_width){

    uint tile_Row = waveId - blockIdx.x;
    uint tile_Column = blockIdx.x;

    //calculating the rows and columns for the tiles
    if (waveId > total_tiles_rows - 1 ){
        tile_Row = total_tiles_rows -1 - blockIdx.x;
        tile_Column = waveId - tile_Row;
    }

    softdtw_tiled_wavefront(da , db , D , waveId , total_tiles_rows, total_tiles_columns
         , tile_width ,tile_Row , tile_Column);
}


/** Host function for computing soft DTW tiled and shared memory version
 *  we consider the tile with as square, otherwise need to define tile width and height
 * @param a The first time series data 
 * @param b The second time series data 
 * @param D The distance matrix
 * @param m Length of first time series
 * @param n Length of second time series
 * @param tile_width the width of our tiled for takin the advantage of shared memory 
 */
 __host__ void soft_dtw_tiled(float *a , float *b, float *D , uint m, uint n , uint tile_width )
 {
     uint total_tiles_columns = ( m + tile_width - 1 ) / tile_width ;
     uint total_tiles_rows = ( n + tile_width - 1 ) / tile_width ;
     uint total_tiles_waves = total_tiles_columns + total_tiles_rows - 1 ;
     
     uint min_tiles = min ( total_tiles_columns , total_tiles_rows );
     uint max_tiles = max ( total_tiles_columns , total_tiles_rows );
     
     uint tile_size = tile_width * tile_width ;
     
     size_t mn_size =  m * n * sizeof(float);
     size_t m_size =  m * sizeof(float);
     size_t n_size =  n * sizeof(float);
     
 
     float *da ,*db ;
     float *D_;
     cudaMalloc( &da , m_size);
     cudaMalloc( &db, n_size);
     cudaMalloc( &D_, mn_size);
 
     cudaMemcpy(da, a, m_size , cudaMemcpyHostToDevice);
     cudaMemcpy(db, b, n_size , cudaMemcpyHostToDevice);
     
     //TODO: not yet sure about this one, need to check
     cudaMemcpy(D_, D, mn_size , cudaMemcpyHostToDevice);
 
 
     //start wave fron t process
     // Populate dependency managed by loop
     for (int waveId=0; waveId < total_tiles_waves; waveId++)
     {
         int wave_Len = waveId + 1;
         if (wave_Len > min_tiles)
             wave_Len = min(min_tiles,total_tiles_waves-waveId);
 
         //call kernel
         //for none squeare block size, we need to pass the min for threadsPerBlock value
         dim3 blockPerGrid (wave_Len);      
         dim3 threadPerBlock (tile_width,tile_width);
 
         softdtw_global_tiled<<<blockPerGrid,threadPerBlock>>>(da , db , D_ , waveId , total_tiles_rows, total_tiles_columns , tile_width );
 
         cudaDeviceSynchronize();
     }
     //copy back data to host
     cudaMemcpy( D, D_, mn_size, cudaMemcpyDeviceToDevice);
 
     //TODO: result verification here
 
 }

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

// // the pairwise squared Euclidean distances kernel execution
// // .....timing....
// std::cout << "STARTING squared Euclidean distances" << std::endl;
// auto sq_euclid_dist_start =
// std::chrono::high_resolution_clock::now();

// sq_euclid_dist(da, db, D, m, n, k);

// // since the kernels are executed asynchronously, need to sync so
// // that we can get accurate timing
// cudaDeviceSynchronize();
// auto sq_euclid_dist_end = std::chrono::high_resolution_clock::now();
// std::cout << "FINISHED squared Euclidean distances" << std::endl;
// auto sq_euclid_dist_duration =
// std::chrono::duration_cast<std::chrono::microseconds>(
// sq_euclid_dist_end - sq_euclid_dist_start)
// .count();

// // the softdtw cuda naive kernel execution .....timing....
// std::cout << "STARTING softdtw cuda naive" << std::endl;
// auto softdtw_cuda_naive_start =
// std::chrono::high_resolution_clock::now();

// softdtw_cuda_naive(D, R, m, n, gamma);

// // since the kernels are executed asynchronously, need to sync so
// // that we can get accurate timing
// cudaDeviceSynchronize();
// auto softdtw_cuda_naive_end =
// std::chrono::high_resolution_clock::now();
// std::cout << "FINISHED softdtw cuda naive" << std::endl;
// auto softdtw_cuda_naive_duration =
// std::chrono::duration_cast<std::chrono::microseconds>(
// softdtw_cuda_naive_end - softdtw_cuda_naive_start)
// .count();

// // the softdtw grad cuda naive kernel execution.....timing....
// std::cout << "STARTING softdtw grad cuda naive" << std::endl;
// auto softdtw_grad_cuda_naive_start =
// std::chrono::high_resolution_clock::now();

// softdtw_grad_cuda_naive(D, R, dE, m, n, gamma);

// since the kernels are executed asynchronously, need to sync so
// that we can get accurate timing
// cudaDeviceSynchronize();
// auto softdtw_grad_cuda_naive_end =
// std::chrono::high_resolution_clock::now();
// std::cout << "FINISHED softdtw grad cuda naive" << std::endl;
// auto softdtw_grad_cuda_naive_duration =
// std::chrono::duration_cast<std::chrono::microseconds>(
// softdtw_grad_cuda_naive_end - softdtw_grad_cuda_naive_start)
// .count();

// cudaMemcpy(E, dE, m * n * sizeof(float), cudaMemcpyDeviceToHost);


//parameters need to revise

//distance matrix
float *D_tiled ;
D_tiled = (float*) malloc(m * n * sizeof(float));

//TODO: //need to check again
//initializing distance matrix with 0    
for (int i = 0; i < m * n; i++)
D_tiled[i]=0.0f;

//TODO: I need to remove the memcopy from the soft_dtw to here
//for timing 
soft_dtw_tiled(a , b, D_tiled, m, n , 16 ) ;


std::cout << " i, j  squared Euclidean distances Execution Time , "
    "softdtw cuda naive Execution Time, softdtw grad cuda "
    "naive\n";
// std::cout << i << ", " << j << " , " << sq_euclid_dist_duration
//  << " , " << softdtw_cuda_naive_duration << " , "
//  << softdtw_grad_cuda_naive_duration << '\n';

// output_file << i << ", " << j << " , " << sq_euclid_dist_duration
//    << "," << softdtw_cuda_naive_duration << ","
//    << softdtw_grad_cuda_naive_duration << '\n';

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


 