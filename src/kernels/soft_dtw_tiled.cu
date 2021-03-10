#include "soft_dtw_tiled.cuh"

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


