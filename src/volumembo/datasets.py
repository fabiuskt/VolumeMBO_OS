import graphlearning as gl
import numpy as np
from sklearn.datasets import make_moons


def load_dataset(
    name: str, N: int = 100, noise: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a dataset by name.
    Args:
        name (str): Name of the dataset.
        N (int): Number of samples to generate for synthetic datasets, only for moons datasets.
        noise (float): Noise level for synthetic datasets.
    Returns:
        data (np.ndarray): , (N, M) shape, where N is the number of points and M is the number of features/labels
        labels (np.ndarray): The dataset labels.
    """
    if name == "2_moons":
        data, labels = make_moons(N=N, noise=noise)
    elif name == "3_moons":
        data, labels = make_3_moons(N=N, noise=noise)
    elif name == "optdigits":
        d = np.loadtxt("../data/optdigits.csv", delimiter=",", dtype=int)
        data = d[:, :-1]
        labels = d[:, -1]

    else:
        raise ValueError(f"Unknown dataset: {name}")
    return data, labels


def make_3_moons(
    N: int = 100, noise: float | None = None, random_state: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a 3-moons dataset with three distinct half-moon shapes.
    Args:
        N (int): Total number of samples to generate.
        noise (float): Standard deviation of Gaussian noise added to the data.
        random_state (int, optional): Random seed for reproducibility.
    Returns:
        X (np.ndarray): Generated data points, shape (N, 2).
        y (np.ndarray): Labels for the data points, shape (N,).
    """
    N_out = N // 3
    N_in = N_out
    N_out2 = N - N_out - N_in

    outer_circ_x = np.cos(np.linspace(0, np.pi, N_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, N_out))

    inner_circ_x = 1.5 - 1.5 * np.cos(np.linspace(0, np.pi, N_in))
    inner_circ_y = 0.4 - 1.5 * np.sin(np.linspace(0, np.pi, N_in))

    outer_circ_x2 = np.cos(np.linspace(0, np.pi, N_out2)) + 3.0
    outer_circ_y2 = np.sin(np.linspace(0, np.pi, N_out2))

    X = np.vstack(
        [
            np.append(np.append(outer_circ_x, inner_circ_x), outer_circ_x2),
            np.append(np.append(outer_circ_y, inner_circ_y), outer_circ_y2),
        ]
    ).T

    y = np.hstack(
        [
            np.zeros(N_out, dtype=int),
            np.ones(N_in, dtype=int),
            2 * np.ones(N_out2, dtype=int),
        ]
    )

    if noise is not None:
        rng = np.random.default_rng(random_state)
        X += rng.normal(scale=noise, size=X.shape)

    return X, y
