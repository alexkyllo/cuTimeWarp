# cuTimeWarp

CUDA Implementations of Dynamic Time Warping and SoftDTW loss function
for time series machine learning.

Based on algorithms described in:

- ["Soft-DTW: A Differentiable Loss Function for Time Series"](https://arxiv.org/pdf/1703.01541.pdf)
- ["Developing a pattern discovery method in time series data and its GPU acceleration"](https://ieeexplore.ieee.org/document/8400444)


TODO:

- [x] Implement naive DTW on CPU
- [x] Implement soft DTW on CPU
- [x] Implement pairwise squared Euclidean distance on CPU
- [ ] Implement soft DTW gradient on CPU
- [ ] Implement soft DTW barycenter estimation on CPU
- [ ] Implement naive DTW in CUDA
- [ ] Implement soft DTW in CUDA
- [x] Implement pairwise squared Euclidean distance in CUDA
- [ ] Implement soft DTW gradient in CUDA
- [ ] Implement soft DTW barycenter estimation in CUDA
- [ ] Choose benchmarking datasets
- [ ] Implement optimizations for soft DTW in CUDA
- [ ] Run benchmark experiments
