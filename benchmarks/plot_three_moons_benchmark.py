import argparse
import datetime
import pickle

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


def plot_results(input_data):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./benchmark_three_moons_{timestamp}.png"

    colors = np.array(["darkorange", "forestgreen", "dodgerblue", "crimson", "cyan"])
    markersize = 3.0
    lw = 0.75

    #############################################################################################
    fig = plt.figure(figsize=(7.22433, 6.0))
    gs = gridspec.GridSpec(nrows=2, ncols=2, wspace=0.25, hspace=0.25)
    #############################################################################################
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.tick_params(
        direction="in", which="both", bottom=True, top=True, left=True, right=True
    )
    ax0.minorticks_on()

    for i, ((threshold_method, diffusion_method), entries) in enumerate(
        input_data.items()
    ):
        label = f"{threshold_method} + {diffusion_method}"
        N, total, _, _, _ = zip(*entries, strict=True)
        ax0.plot(
            N,
            total,
            marker="o",
            markersize=markersize,
            lw=lw,
            label=label,
            color=colors[i],
            clip_on=True,
            zorder=10,
        )

    ax0.set_xscale("log")
    ax0.set_yscale("log")

    ax0.set_xlim(left=9.0e1, right=2.5e5)
    ax0.set_ylim(bottom=1.0e-4, top=1.0e1)

    ax0.grid(True)
    ax0.legend(loc="upper left")

    ax0.set_xlabel("Number of samples N")
    ax0.set_ylabel("Elapsed time t /s")
    #############################################################################################
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.tick_params(
        direction="in", which="both", bottom=True, top=True, left=True, right=True
    )
    ax1.minorticks_on()

    for i, ((threshold_method, diffusion_method), entries) in enumerate(
        input_data.items()
    ):
        label = f"{threshold_method} + {diffusion_method}"
        N, total, _, _, _ = zip(*entries, strict=True)
        N_scaled = N * np.log(N)
        ax1.plot(
            N_scaled,
            total,
            marker="o",
            markersize=markersize,
            lw=lw,
            label=label,
            color=colors[i],
            clip_on=True,
            zorder=10,
        )

    ax1.set_xlim(left=100, right=2.5e6)
    ax1.set_ylim(bottom=0.0, top=0.8)
    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1.yaxis.get_major_formatter().set_scientific(True)
    ax1.yaxis.get_major_formatter().set_powerlimits((0, 0))

    # ax1.grid(True)
    # ax1.legend()

    ax1.set_xlabel("Number of samples N log(N)")
    ax1.set_ylabel("Elapsed time t /s")
    #############################################################################################
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.tick_params(
        direction="in", which="both", bottom=True, top=True, left=True, right=True
    )
    ax2.minorticks_on()

    for i, ((threshold_method, diffusion_method), entries) in enumerate(
        input_data.items()
    ):
        label = f"{threshold_method} + {diffusion_method}"
        N, total, diffusion, threshold, build_matrices = zip(*entries, strict=True)
        ax2.plot(
            N,
            diffusion,
            marker="o",
            markersize=markersize,
            lw=lw,
            label=label,
            color=colors[i],
            clip_on=True,
            zorder=10,
        )
        ax2.plot(
            N,
            threshold,
            marker="x",
            markersize=markersize,
            lw=lw,
            label=label,
            color=colors[i],
            clip_on=True,
            zorder=10,
        )
        ax2.plot(
            N,
            build_matrices,
            marker="P",
            markersize=markersize,
            lw=lw,
            label=label,
            color="k",
            clip_on=True,
            zorder=10,
        )

    ax2.set_xscale("log")
    ax2.set_yscale("log")

    ax2.set_xlim(left=9.0e1, right=2.5e5)
    ax2.set_ylim(bottom=1.0e-6, top=1.0e3)

    # ax2.grid(True)
    # ax2.legend()

    ax2.set_xlabel("Number of samples N")
    ax2.set_ylabel("Elapsed time t /s")
    #############################################################################################
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    fig.savefig(filename, transparent=False, dpi=600)
    print(f"Saved benchmark plot to: {filename}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("filename", type=str, help="Path to the results .pkl file")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.filename, "rb") as f:
        data = pickle.load(f)

    plot_results(data)


if __name__ == "__main__":
    main()
