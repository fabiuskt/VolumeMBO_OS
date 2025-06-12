import numpy as np

import volumembo
import _volumembo


def test_fit_median():
    # Load three moons dataset
    data, labels = volumembo.datasets.load_dataset("3_moons", N=100, noise=0.075)

    # Create MBO object
    MBO = volumembo.MBO(
        data=data,
        labels=labels,
        number_of_neighbors=6,
        diffusion_time=1.0,
        number_of_known_labels=5,
        initial_clustering_method="random",
        threshold_method="fit_median",
        diffusion_method="A_3",
    )

    number_of_labels = MBO.number_of_labels

    # Get diffused labels u
    MBO.make_fidelity_set()
    labels_before_diffusion, _, _ = MBO.get_initial_cluster()
    diffused_labels = MBO.diffuse(labels_before_diffusion)

    # Restrict to exact volume
    upper_limit = MBO.volume
    lower_limit = MBO.volume

    # Get labels and median from class-based fitter
    fitter = volumembo.median_fitter.VolumeMedianFitter(
        diffused_labels, lower_limit, upper_limit
    )
    clustering, _ = fitter.run(return_history=True)

    # Get labels and median from C++ fitter
    median_cpp = _volumembo.fit_median_cpp(diffused_labels, lower_limit, upper_limit)
    clustering_cpp = volumembo.utils.assign_clusters(diffused_labels, median_cpp)

    # Get labels and median from legacy fitter
    _, clustering_legacy, _ = volumembo.legacy.fit_median(
        number_of_labels,
        lower_limit,
        upper_limit,
        diffused_labels,
        np.full(number_of_labels, 1 / number_of_labels),
    )

    assert np.all(clustering == clustering_legacy)
    assert np.all(clustering == clustering_cpp)
