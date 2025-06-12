import datetime
import pickle
import numpy as np

import volumembo


def setup_MBO(N, method):
    data, labels = volumembo.datasets.load_dataset("3_moons", N=N, noise=0.075)

    MBO = volumembo.MBO(
        data=data,
        labels=labels,
        number_of_neighbors=6,
        diffusion_time=1.0,
        number_of_known_labels=5,
        initial_clustering_method="random",
        threshold_method=method,
        diffusion_method="A_3",
    )

    return MBO


def main():
    methods = ["argmax", "fit_median_cpp", "fit_median", "fit_median_legacy"]
    NSamples = [
        100,
        200,
        350,
        550,
        1000,
        2000,
        3500,
        5500,
        10000,
        20000,
        55000,
        100000,
        150000,
        200000,
    ]
    NSamples.reverse()
    results = {method: [] for method in methods}
    iterations = 10

    # Run benchmarks
    for n, N in enumerate(NSamples):
        MBO = setup_MBO(N, methods[0])
        for m, method in enumerate(methods):
            MBO.set_threshold_function(method)
            _, elapsed = MBO.run(iterations=iterations)
            results[method].append((N, elapsed / iterations))

    # Save results to a compressed .npz file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./benchmark_three_moons_{timestamp}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
