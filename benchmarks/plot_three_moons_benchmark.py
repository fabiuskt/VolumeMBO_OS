import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
import pickle
import numpy as np


def fit_line(x, y):
    """
    Fits a line to log-log data (y = a * x^b).
    Returns slope (b), intercept (log(a)), and r-value.
    """
    log_x = np.log10(x)
    log_y = np.log10(y)
    slope, intercept, r_value, _, _ = linregress(log_x, log_y)
    return slope, intercept, r_value


def plot_results(input_data):
    filename = f"./benchmark_three_moons.png"

    markersize = 3.0
    lw = 0.75

    #############################################################################################
    fig = plt.figure(figsize=(7.22433, 4.0))
    gs = gridspec.GridSpec(nrows=1, ncols=2, wspace=0.25)
    #############################################################################################
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.tick_params(
        direction="in", which="both", bottom=True, top=True, left=True, right=True
    )
    ax0.minorticks_on()

    for method, data in input_data.items():
        x, y = zip(*data)
        ax0.plot(
            x,
            y,
            marker="o",
            markersize=markersize,
            lw=lw,
            label=method,
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

    for method, data in input_data.items():
        x, y = zip(*data)
        x_scaled = x * np.log(x)
        ax1.plot(
            x_scaled,
            y,
            marker="o",
            markersize=markersize,
            lw=lw,
            label=method,
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
