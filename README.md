# cuTimeWarp

CUDA Implementations of Dynamic Time Warping and SoftDTW loss function
for time series machine learning.

Based on algorithms described in:

- ["Soft-DTW: A Differentiable Loss Function for Time Series"](https://arxiv.org/pdf/1703.01541.pdf)
- ["Developing a pattern discovery method in time series data and its GPU acceleration"](https://ieeexplore.ieee.org/document/8400444)


## TODO

- [x] Implement naive DTW on CPU
- [x] Implement soft DTW on CPU
- [ ] Choose benchmarking datasets
- [ ] Implement pairwise squared Euclidean distance on CPU
- [ ] Implement soft DTW gradient on CPU
- [ ] Implement soft DTW barycenter estimation on CPU
- [ ] Implement naive DTW in CUDA
- [ ] Implement soft DTW in CUDA
- [ ] Implement pairwise squared Euclidean distance in CUDA
- [ ] Implement soft DTW gradient in CUDA
- [ ] Implement soft DTW barycenter estimation in CUDA
- [ ] Implement optimizations for soft DTW in CUDA
- [ ] Run benchmark experiments
- [ ] Analysis of experiment results

## Dataset candidates for benchmarking:

- [Urban Sound 8K](https://www.kaggle.com/chrisfilo/urbansound8k)
  10-class classification task on 8732 samples of about 4 seconds of audio each
- [FreeSound](https://annotator.freesound.org/fsd/downloads/) or some
  subset of human or animal sounds
- [Speech Recognition Challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data)
- [Smartphone and Smart Watch
  Activity](https://archive.ics.uci.edu/ml/datasets/WISDM+Smartphone+and+Smartwatch+Activity+and+Biometrics+Dataset+)
  (or another one of the UCR labeled human activity recognition
  datasets)

## Tasks for performance testing

- k-means clustering with SoftDTW barycenters
- Nearest Barycenter Classification
- Supervised classification gradient descent fitting with SoftDTW loss

## Performance metrics

- GFLOP/s
- Execution time
- Cache hit rates
- Achieved Occupancy

## Task metrics

- Clustering: k-means distortion
- Classification: F1 Score
