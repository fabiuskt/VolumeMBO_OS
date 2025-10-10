import numpy as np
import scipy as sp


def direction_to_grow(cluster_idx: int, number_of_labels: int) -> np.ndarray:
    """Return the direction to grow cluster `cluster_idx` in the P-simplex."""
    direction = np.full(number_of_labels, 1 / (number_of_labels - 1))
    direction[cluster_idx] = -1
    return direction


def onehot_to_labels(onehot: np.ndarray) -> np.ndarray:
    """
    Convert one-hot encoding to integer labels.

    Args:
        onehot (np.ndarray): One-hot encoded matrix of shape (N, number_of_labels).

    Returns:
        np.ndarray: Array of shape (N,) containing integer labels.
    """
    return np.argmax(onehot, axis=1)


def assign_clusters(u: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Assign clusters based on the input matrix u and the median m.

    Args:
        u (np.ndarray): Input matrix of shape (N, number_of_labels).
        m (np.ndarray): Median values of shape (number_of_labels,).

    Returns:
        np.ndarray: Assigned labels of shape (N,).
    """
    return onehot_to_labels(u - m)


def bellman_ford_voronoi_initialization(
    N: int, fidelity: np.ndarray, y: np.ndarray, W: sp.sparse.spmatrix
) -> tuple[np.ndarray, np.ndarray]:
    # voronoi tesselation via bellman-ford algorithm
    active = np.zeros(N)
    fixedLabels = fidelity
    labels = np.full(N, -1, dtype=int)
    labels[fixedLabels] = y[fixedLabels]
    active[fixedLabels] = True
    voronoiDistances = np.zeros(N)
    for i in range(N):

        if not active[i]:
            voronoiDistances[i] = np.inf

    done = False
    while not done:
        done = 1
        for i in range(N):
            if active[i]:
                done = 0
                for j in range(W.indptr[i], W.indptr[i + 1]):
                    index = W.indices[j]
                    if W.data[j] != 0:
                        dist = W.data[j]
                        current = voronoiDistances[i]
                        if current + dist < voronoiDistances[index]:
                            voronoiDistances[index] = current + dist
                            active[index] = True
                            labels[index] = labels[i]
                active[i] = False

    return voronoiDistances, labels
