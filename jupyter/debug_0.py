import numpy as np

import volumembo
from volumembo.utils import assign_clusters
from volumembo.utils import onehot_to_labels
from _volumembo import fit_median_cpp
from volumembo.legacy import fit_median as fit_median_legacy


def build_transform_heat_kernel_2D(N, M, t):
    ith, jth = np.meshgrid(range(N), range(M), indexing="ij")
    eigenvalues = (
        2.0 * np.cos(2.0 * np.pi * ith / N) + 2.0 * np.cos(2.0 * np.pi * jth / M) - 4.0
    )
    kernel = np.exp(t * eigenvalues)
    return kernel


def diffuse_on_2Dgrid(image, kernel):
    transformed_image = np.fft.fft2(image)
    product = transformed_image * kernel
    # test = np.fft.ifft2(product)
    return np.fft.ifft2(product).real


def diffused_to_onehot(u):
    return volumembo.MBO._diffused_to_onehot(u)


N = 20
M = 20
t = 1
P = 10
kernel = build_transform_heat_kernel_2D(N, M, t)
print("kernel:\t\t", kernel.shape)

image = np.zeros((N, M))
print("image:\t\t", image.shape)
image_one_hot = np.zeros((N, M, P))
print("image_one_hot:\t", image_one_hot.shape)


line = np.array(range(N * M))
print("line: ", line.shape)
np.random.shuffle(line)
line = line.reshape((N, M))
print("line: ", line.shape)
for i in range(N):
    for j in range(M):
        index = int(line[i, j] / (N * M / P))
        image_one_hot[i, j, index] = 1.0


volume = np.array(np.sum(image_one_hot.reshape(N * M, P), axis=0).astype(int).tolist())
# print('volumes:', volume, volume.shape, flush=True)
# print('', flush=True)

diffused = np.zeros_like(image_one_hot, dtype=np.float64)

for i in range(50):
    for p in range(P):
        diffused[:, :, p] = diffuse_on_2Dgrid(image_one_hot[:, :, p], kernel)

    median = np.array(fit_median_cpp(diffused.reshape((N * M, P)), volume, volume))
    # median = np.array([1/3.,1/3.,1/3.])

    image_one_hot = diffused_to_onehot(diffused.reshape((N * M, P)) - median).reshape(
        (N, M, P)
    )
    # image_one_hot = assign_clusters(diffused.reshape((N*M, P)), median).reshape((N,M,P))

print(median)
