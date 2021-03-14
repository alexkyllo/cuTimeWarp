#include "soft_dtw_tiled.cuh"
#include "helper_functions.cuh"

/*credit and appreciation to Mehmet E. Belviranli
https://mehmet.belviranli.com/papers/ics15.pdf
*/


/** Kernel function for computing tiled Soft DTW using wave front approach
 * matrix for multivariate time series with CUDA. Input a , b and D should be a
 * __device__ array.
 * @param a The first time series data
 * @param b The second time series data
 * @param m Length of first time series
 * @param n Length of second time series
 * @param D The pairwise distance array of two time series
 * @param waveId the ID for the wave
 * @param total_tiles_rows The total number of tiles within the row
 * @param total_tiles_columns The total number of tiles within column
 * @param tile_width The tile width
 * @param gamma SoftDTW smoothing parameter
 */

__global__ void softdtw_global_tiled(float *da, float *db, float *D, int waveId,
                                     uint total_tiles_rows,
                                     uint total_tiles_columns, uint tile_width,float gamma)
{

    uint tile_Row = waveId - blockIdx.x;
    uint tile_Column = blockIdx.x;

    // calculating the rows and columns for the tiles
    if (waveId > total_tiles_rows - 1)
    {
        tile_Row = total_tiles_rows - 1 - blockIdx.x;
        tile_Column = waveId - tile_Row;
    }

    softdtw_tiled_wavefront(da, db, D, total_tiles_rows,
                            total_tiles_columns, tile_width, tile_Row,
                            tile_Column,gamma);
}

/** Kernel function for computing tiled Soft DTW using wave front approach
 * matrix for multivariate time series with CUDA. Input a , b and D should be a
 * __device__ array.
 * @param a The first time series data
 * @param b The second time series data
 * @param m Length of first time series
 * @param n Length of second time series
 * @param D The pairwise distance array of two time series
 * @param total_tiles_rows The total number of tiles within the row
 * @param total_tiles_columns The total number of tiles within the column
 * @param tile_width  The tile width
 * @param tileRow
 * @param tileColumn
 * @param gamma SoftDTW smoothing parameter
 */

__device__ void softdtw_tiled_wavefront(float *a, float *b, float *D
                                        , uint total_tiles_rows,
                                        uint total_tiles_columns,
                                        uint tile_width, uint tileRow,
                                        uint tileColumn, float gamma)
{

    // the main tile computation start here
    const int tid = threadIdx.x;
    const int tile_size = tile_width * tile_width;
    const int tileBaseIndex =
        tile_size * total_tiles_columns * tileRow + tile_size * tileColumn;

    const int nWaves = tile_width * 2 - 1;

    // TODO: check with 1024
    __shared__ float seq1[512];
    __shared__ float seq2[512];

    __syncthreads();

    seq1[tid] = a[tileColumn * tile_width + tid];
    seq2[tid] = b[tileRow * tile_width + tid];

    __syncthreads();

    for (int waveId = 0; waveId < nWaves; waveId++)
    {
        const int row = waveId - tid;
        const int column = tid;

        const int index = tileBaseIndex + tile_width * column + row;

        if (row >= 0 && row < tile_width && column >= 0 && column < tile_width)
        {

            const float cost = fabs(seq2[row] - seq1[column]);

            float upleft = 0;
            float left = 0;
            float up = 0;

            // LEFT index
            if (tileColumn > 0 || column > 0) 
            {
                const int leftIndex = index - tile_width;
                left = D[leftIndex] - 1;
            }

            //UP index
            if (tileRow > 0 || row > 0)
            {
                int upIndex = index - 1;
                if (row == 0)
                {
                    upIndex = tileBaseIndex - tile_size * total_tiles_columns;
                    upIndex += tile_width * (column + 1) - 1;
                }

                up = D[upIndex] - 1;
            }

            ////  UPLEFT //
            if ((tileColumn > 0 || column > 0) && (tileRow > 0 || row > 0))
            {
                int upLeftIndex = index - 1 - tile_width;
                if (row == 0)
                {
                    upLeftIndex =
                        tileBaseIndex - tile_size * total_tiles_columns;
                    upLeftIndex += tile_width * (column)-1;
                }

                upleft = D[upLeftIndex];
            }

            // TODO: should be change to softmin
            //D[index] = ((int)cost / 100) + min(upleft, min(left, up));

            D[index] = (float)cost + softmin(upleft, left, up, gamma);

        }

        __syncthreads();
    }
}
