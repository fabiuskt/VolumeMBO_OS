import datetime
import pickle
import numpy as np

import volumembo


def main():
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
    iterations = 10

    benchmark_plan = [
        ("argmax", "A_3"),
        ("argmax", "W2"),
        ("fit_median_cpp", "A_3"),
        ("fit_median_legacy", "A_3"),
    ]

    results = {}  # key: (threshold_method, diffusion_method), value: list of tuples

    # Run benchmarks
    for n, N in enumerate(NSamples):
        print("Working on N =", N, f"({n+1}/{len(NSamples)})")
        data, labels = volumembo.datasets.load_dataset("3_moons", N=N, noise=0.075)
        MBO = volumembo.MBO(
            data=data,
            labels=labels,
            number_of_neighbors=6,
            diffusion_time=1.0,
            number_of_known_labels=5,
            initial_clustering_method="random",
            threshold_method="argmax",
            diffusion_method="A_3",
        )

        for threshold_method, diffusion_method in benchmark_plan:
            MBO.set_diffusion_method(diffusion_method)
            MBO.set_threshold_function(threshold_method)
            MBO.run(iterations=iterations, enable_timing=True)

            # Extract averaged diffusion and threshold times
            summary = MBO.timer.summary()
            run_avg = summary.get("run", {}).get("mean", None)
            diffusion_avg = summary.get("diffusion", {}).get("mean", None)
            threshold_avg = summary.get("threshold", {}).get("mean", None)
            build_avg = summary.get("build_matrices", {}).get("mean", None)

            key = (threshold_method, diffusion_method)
            results.setdefault(key, []).append(
                (N, run_avg, diffusion_avg, threshold_avg, build_avg)
            )

    # Save results to a compressed .npz file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./benchmark_three_moons_{timestamp}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
