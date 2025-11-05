import numpy as np

import volumembo
import _volumembo


def test_fit_median():
    # Load three moons dataset
    data, labels = volumembo.datasets.load_dataset(
        "3_moons", N=100, noise=0.075, random_state=0
    )

    # Create MBO object
    MBO = volumembo.MBO(
        data=data,
        labels=labels,
        number_of_neighbors=6,
        diffusion_time=1.0,
        number_of_known_labels=5,
        initial_clustering_method="random",
        threshold_method="fit_median_cpp",
        diffusion_method="A_3",
    )

    number_of_labels = MBO.number_of_labels

    # Get diffused labels u
    MBO.make_fidelity_set()
    labels_before_diffusion, _, _ = MBO.get_initial_cluster()
    diffused_labels = MBO.diffuse(labels_before_diffusion)

    # Restrict to exact volume
    target = MBO.volume
    P = len(target)

    # Get labels and median from C++ fitter
    median_cpp = _volumembo.fit_median_cpp(diffused_labels, target)
    clustering_cpp = volumembo.utils.assign_clusters(diffused_labels, median_cpp)
    count_cpp = np.bincount(clustering_cpp)
    assert np.sum(count_cpp) == np.sum(target)
    assert np.all(np.abs(count_cpp - target) < P)  # Maximum volume error = P-1

    # Get labels and median from legacy fitter
    _, clustering_legacy, _ = volumembo.legacy.fit_median(
        number_of_labels,
        target,
        target,
        diffused_labels,
        np.full(number_of_labels, 1 / number_of_labels),
    )
    count_legacy = np.bincount(clustering_legacy)
    assert np.sum(count_legacy) == np.sum(target)
    assert np.all(np.abs(count_legacy - target) < P)  # Maximum volume error = P-1
