# cuTimeWarp

CUDA C++ implementations of Dynamic Time Warping and SoftDTW loss function for
time series machine learning.

Based on algorithms described in:

- ["Soft-DTW: A Differentiable Loss Function for Time Series"](https://arxiv.org/pdf/1703.01541.pdf)
- ["Developing a pattern discovery method in time series data and its GPU acceleration"](https://ieeexplore.ieee.org/document/8400444)

## Building

This project uses a Makefile to coordinate separate compilation of CUDA kernels
and C++ code and is tested on Ubuntu Linux. Typing `make` will list the
available commands:

``` shell
$ make

Available rules:

build               Build binaries
clean               Delete binaries
fmt                 Format the code with clang-format
plot                Run python script to generate plots
report              Compile the PDF report
run                 Run experiments
run_multi           Run multi-distance experiments
test                Build and run unit tests
```

To compile the kernels and the test programs, use the `make build` command.

## Running

The three programs to use for running comparative performance experiments are:


- `bin/soft_dtw_perf_cpu` for timing CPU performance
- `bin/soft_dtw_perf_multi` for timing GPU performance
- `bin/soft_dtw_perf_tiled` for timing the tiled kernel on GPU (for long time
  series > 1024)

The programs accept as arguments either a filename containing space-delimited
data (see [data/ECG200/ECG200_ALL.txt](data/ECG200/ECG200_ALL.txt)) or the word
`random` and a time series length and count. The program will compute the
Soft-DTW dissimilarity between all pairs of time series in the batch and then
print output in four columns:

- Kernel function name
- The input time series length (number of columns per row)
- The input time series count (number of rows)
- The execution time in microseconds

Example:

``` shell
$ ./bin/soft_dtw_perf_multi
Usage: ./bin/soft_dtw_perf_multi [INPUT_FILENAME] | random [length] [count]

$  ./bin/soft_dtw_perf_multi ./data/ECG200/ECG200_ALL.txt
Data file ./data/ECG200/ECG200_ALL.txt contains 200 time series of length 96
sq_euclid_dist_multi 96 200 515037
softdtw_cuda_naive_multi 96 200 264987
softdtw_cuda_naive_multi_bw_80 96 200 235089
softdtw_cuda_naive_multi_bw_60 96 200 168621
softdtw_cuda_naive_multi_bw_40 96 200 83501
softdtw_cuda_naive_multi_bw_20 96 200 51338
softdtw_cuda_stencil_multi 96 200 100990
softdtw_cuda_stencil_multi_80 96 200 100408
softdtw_cuda_stencil_multi_60 96 200 100844
softdtw_cuda_stencil_multi_40 96 200 101215
softdtw_cuda_stencil_multi_40 96 200 100436
softdtw_cuda_stencil_multi_20 96 200 100647
convert_diagonal_multi 96 200 332664
softdtw_cuda_diagonal_multi 96 200 149158

$ ./bin/soft_dtw_perf_multi random 100 100
sq_euclid_dist_multi 100 100 335883
softdtw_cuda_naive_multi 100 100 61576
softdtw_cuda_naive_multi_bw_80 100 100 52272
softdtw_cuda_naive_multi_bw_60 100 100 32211
softdtw_cuda_naive_multi_bw_40 100 100 18919
softdtw_cuda_naive_multi_bw_20 100 100 18725
softdtw_cuda_stencil_multi 100 100 26558
softdtw_cuda_stencil_multi_80 100 100 25803
softdtw_cuda_stencil_multi_60 100 100 31000
softdtw_cuda_stencil_multi_40 100 100 26120
softdtw_cuda_stencil_multi_40 100 100 25804
softdtw_cuda_stencil_multi_20 100 100 30992
convert_diagonal_multi 100 100 87427
softdtw_cuda_diagonal_multi 100 100 43893
```

## TODO List

- [x] Implement naive DTW on CPU
- [x] Implement soft DTW on CPU
- [x] Choose benchmarking datasets
- [x] Implement pairwise squared Euclidean distance on CPU
- [x] Implement soft DTW gradient on CPU
- [ ] Implement soft DTW barycenter estimation on CPU
- [x] Implement naive soft DTW in CUDA
- [x] Implement pairwise squared Euclidean distance in CUDA
- [x] Implement soft DTW gradient in CUDA
- [ ] Implement soft DTW barycenter estimation in CUDA
- [x] Tiling
- [x] Shared memory stencil
- [x] Sakoe-Chiba bands
- [x] Contiguous diagonal-major array storage layout
- [x] Run benchmark experiments
- [x] Analysis of experiment results
