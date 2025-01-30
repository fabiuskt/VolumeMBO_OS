## VolumeMBO_OS
This is the GitHub repository to the Paper 
Krämer, Fabius, and Tim Laux. "An efficient volume-preserving MBO scheme for data clustering and classification." arXiv preprint arXiv:2412.17694 (2024).

## Files
The [LU_order_statistic.py](https://github.com/fabiuskt/VolumeMBO_OS/blob/main/LU_order_statistic.py) contains the implementation of the algorithm to find the $(L,U)$-order statistic.
The files [mnist_final.py](https://github.com/fabiuskt/VolumeMBO_OS/blob/main/mnist_final.py), [optdigits_final.py](https://github.com/fabiuskt/VolumeMBO_OS/blob/main/optdigits_final.py) and [three_moons_final.py](https://github.com/fabiuskt/VolumeMBO_OS/blob/main/three_moons_final.py) implement the MBO scheme with different diffusion kernels on the MNIST, OPT-digits and Three-Moons dataset, respectively. 

## Libraries
Necessary libraries:
1. [scikit-learn](https://pypi.org/project/scikit-learn/)
2. [scipy](https://scipy.org/)
3. [graphlearning](https://github.com/jwcalder/GraphLearning?tab=readme-ov-file)
