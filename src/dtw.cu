/**
 * CSS-535 Final Project : Soft DTW CUDA Implementation
 * Authors: Alex Kyllo & Afrooz Rahmati
 * @file dtw.cu
 * Description: CUDA implementation of naive DTW
 */

// CUDA stuff 
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h> // as a benchmark 

#include <random> // for random initialization
#include <chrono> // timing
#include <iostream> // for output 
#include <limits>

/** GPU function to initilize the Warp matrix with infinity value
 * we can implement this part on host as well
 *  @param W A Warp path for the distance matrix of dimension (m x n)
 *  @param m The length of vectors in X
 *  @param n The length of vectors in Y
 */

__global__ void initialize_warp(const float *W, const uint m, const uint n) {

    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= m)
    {
        for (uint j = 0; j <= n; j++)
        {
            W[i * n + j] = std::numeric_limits<T>::infinity();
        }
    }
}



__device__ void dtwm_task(size_t i, size_t j,
               size_t* I, double* C, double* D, size_t* L,
               size_t* Rsi, size_t* Rsj, size_t* Rli, size_t* Rlj, size_t* Pi, size_t* Pj,
               double t, size_t o, const size_t n){
    double minpre, dtwm;

    size_t idx   = I[i*n +j];
    size_t min_idx = idx;
    size_t mini = i; size_t minj = j;

    if ( i==0 || j==0 ){    // special case where index is 0 there is no previous
        minpre = 0.0;
//      mini = i; minj = j;
    } else{
        size_t idx_d = I[(i-1)*n +(j-1)];
        size_t idx_t = I[(i-1)*n +(j  )];
        size_t idx_l = I[(i  )*n +(j-1)];

        minpre = min3( D[idx_d], D[idx_t], D[idx_l] );

        // mini, minj are the index of the min previous cells
        if (minpre == D[idx_d]){
            mini = i-1;
            minj = j-1;
            min_idx = idx_d;
        }else if(minpre == D[idx_t]){
            mini = i-1;
            minj = j;
            min_idx = idx_t;
        }else if(minpre == D[idx_l]){
            mini = i;
            minj = j-1;
            min_idx = idx_l;
        }

        // todo: cannot call clib inf from here, use cuda math_constrains.h
        if (minpre==CUDART_INF){ minpre = 0.0; }
    }

    // calculated average cost for the path adding the current cell
    dtwm = (minpre + C[idx]) / (L[min_idx] + 1.0);

    // only consider this cell if average cost dtwm smaller than t
    if (dtwm < t && (L[min_idx] == 0)) {
//            if ( dtwm<t && L[getIndex(i-1,j-1)]==0
//                        && L[getIndex(i-1,j  )]==0
//                        && L[getIndex(i  ,j-1)]==0) {

        // if previous cell not in a path, start new path

        D[idx] = C[idx];  // update current cell dtw distance
        L[idx] = 1;                 // update current cell dtw length

        Rsi[idx] = i; // this path start at i
        Rsj[idx] = j; // this path start at j
        Rli[idx] = i; // this path ends at i
        Rlj[idx] = j; // this path ends at j

        // else add to the previous cell's path
        // if the current cell is not diverge more than the offset o
    }else if (dtwm < t) {
        size_t si = Rsi[min_idx];
        size_t sj = Rsj[min_idx];

        // Note: have to use comparison since size_t is unsigned !!
        // guarantee si is smaller than i for this implementation but watch out
        size_t offset = (i-si)>(j-sj) ? (i-si)-(j-sj) : (j-sj)-(i-si);
        if ( offset < o){
            D  [idx] = minpre + C[idx];  // update current cell dtw distance
            L  [idx] = ( i==0 || j==0 ) ? 1 : L[min_idx] + 1;// update current cell dtw length

            Rsi[idx] = si; // this path start at same as previous cell
            Rsj[idx] = sj; // this path start at same as previous cell

            Pi [idx] = mini; // mark path
            Pj [idx] = minj; // mark path

            // update last position further away
            size_t s_idx = I[si*n +sj];
            size_t li = Rli[ s_idx ]; // Path end i, only stored in start cell
            size_t lj = Rlj[ s_idx ]; // Path end j, only stored in start
            if ( i > li && j > lj ){
                Rli[s_idx] = i;
                Rlj[s_idx] = j;
            }
        }
    }
}




__global__  void cuda_dtw(size_t* I, double* C, double* D, size_t* L,
              size_t* Rsi, size_t* Rsj, size_t* Rli, size_t* Rlj, size_t* Pi, size_t* Pj,
              double t, size_t o, const size_t m, const size_t n){
    const size_t tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (tid < n){
        __syncthreads(); 
        size_t i,j;
        for (size_t si = 0; si < m; ++si) {
            // only the thread within the anti-diagonal region is called
            if ( tid <= min2(si, n-1)){    // careful with case m > n
                i = si - tid; // start i position (si) walking up tid times
                j = tid;
                dtwm_task(i, j,
                          I, C, D, L,
                          Rsi, Rsj, Rli, Rlj, Pi, Pj,
                          t, o, n);
            }
            __syncthreads();
        }

        for (size_t sj = 1; sj < n; ++sj) {
            // only the thread within the anti-diagonal region is called
            if ( tid >= sj ){                  // careful with case n > m
                i = m-1 - min2(tid-sj, m-1); // last i - step from cell position (tid) to sj
                j = tid;
                dtwm_task(i, j,
                          I, C, D, L,
                          Rsi, Rsj, Rli, Rlj, Pi, Pj,
                          t, o, n);
            }
            __syncthreads();
        }
    } // run total for m+n-1 times in parallel

}

__device__ double getCost(size_t i, size_t j, double* dX, double* dY){
    return dX[i] > dY[j] ? dX[i] - dY[j] : dY[j] - dX[i];
}


__global__ void initCuda(size_t *I, double* C, double* D, double* dX, double* dY,
              const size_t m, const size_t n) {
    const size_t global_index = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (global_index < m*n){
        const size_t i = global_index / n;
        const size_t j = global_index % n;

        // calculate cost matrix using the anti diagonal index just got
        size_t idx = i*n+ j;
        I[i*n+ j] = idx;
        C[idx] = getCost(i, j, dX, dY);
        D[idx] = CUDART_INF;
    }
}



/** Host function to compute the Squared Euclidean distance between two sets of
 *  column vectors (e.g. two multivariate time series)
 *  X and Y by using the euclidian norms, i.e. X*X + Y*Y - 2X*Y
 *  Inputs X, Y, D should be __device__ vectors
 *  @param X A set of vectors of length (row count) m
 *  @param Y A set of vectors of length (row count) n
 *  @param D A result array for the distance matrix of dimension (m x n)
 *  @param m The length of vectors in X
 *  @param n The length of vectors in Y
 *  @param k The number of vectors in X and Y (columns)
 */
 __host__ void sq_dtw(const size_t t , const size_t o)
{
float *XX;
float *YY;
float *C;
float *D;
float *I;
float *L;
float *Rsi;
float *Rsj;
float *Rli;
float *Pi;
float *Pj;

size_t size_m = m * sizeof(float);
size_t size_n = n * sizeof(float);
size_t size_mn = m * n * sizeof(float);



cudaMalloc(&XX, size_m);
cudaMalloc(&YY, size_n);
cudaMalloc(&C, size_mn);  
cudaMalloc(&D, size_mn); 
cudaMalloc(&I, size_mn); 
cudaMalloc(&L, size_mn); 
cudaMalloc(&Rsi, size_mn); 
cudaMalloc(&Rsj, size_mn); 
cudaMalloc(&Rli, size_mn); 
cudaMalloc(&Rlj, size_mn); 
cudaMalloc(&Pi, size_mn); 
cudaMalloc(&Pj, size_mn); 

cudaMemset(XX, 0, size_m);
cudaMemset(YY, 0, size_n);
cudaMemset(I, 0, size_mn);      
cudaMemset(L, 0, size_mn);         
cudaMemset(Rsi, 0, size_mn);
cudaMemset(Rsj, 0, size_mn);
cudaMemset(Rli, 0, size_mn);
cudaMemset(Rlj, 0, size_mn);
cudaMemset(Pi, 0, size_mn);
cudaMemset(Pj, 0, size_mn);


uint block_size = min(m, 1024);


const uint num_blocks = (m*n + block_size-1)/block_size; 
initCuda<<<num_blocks, block_size>>>(I, C, D, XX, YY, m, n);


uint grid_size = (m + block_size - 1) / block_size;
cuda_dtw<<<grid_size, block_size>>>(I, C, D, L, Rsi, Rsj, Rli, Rlj, Pi, Pj, t, o, m, n);


cudaFree(XX);
cudaFree(YY);

cudaFree(I);
cudaFree(C);
cudaFree(D);
cudaFree(L);
cudaFree(Rsi);
cudaFree(Rsj);
cudaFree(Rli);
cudaFree(Rlj);

cudaFree(Pi);
cudaFree(Pj);

}
