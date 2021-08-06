import argparse

import matplotlib

matplotlib.use("TkAgg")
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_mean_std(ax, x, y_mean, y_std, color="darkorange", alpha=0.2, label=None):
    ax.fill_between(
        x, y_mean - y_std, y_mean + y_std, color=color, alpha=alpha, label=label
    )


def plot(logdir, vis_headers_y_mean, vis_header_y_std, optional_legends, delta=None):
    """Plot mean and std of given statistics."""
    progress_path = os.path.join(logdir, "progress.csv")
    reader = csv.reader(open(progress_path, "rt"), delimiter=",")
    raw_data = list(reader)

    headers = raw_data[0]
    data = np.array(raw_data[1:]).astype("float")

    vis_headers_x = [
        "Epoch",
    ]
    vis_header_idx_x = [headers.index(x) for x in vis_headers_x]
    data_x = data[:, vis_header_idx_x[0]]

    vis_header_idx_y_mean = [headers.index(x) for x in vis_headers_y_mean]
    vis_header_idx_y_std = [headers.index(x) for x in vis_header_y_std]

    data_y_mean = []
    data_y_std = []
    for idx in vis_header_idx_y_mean:
        data_y_mean.append(data[:, idx])
    for idx in vis_header_idx_y_std:
        data_y_std.append(data[:, idx])

    for i in range(len(data_y_mean)):
        fig, ax = plt.subplots()
        plot_mean_std(
            ax,
            data_x,
            data_y_mean[i],
            data_y_std[i],
            color="gray",
            label=vis_header_y_std[i],
        )
        ax.plot(data_x, data_y_mean[i], label=vis_headers_y_mean[i])

        # Plot a line for upper risk bound if it exists.
        if "Risk" in vis_headers_y_mean[i] and delta is not None:
            ax.hlines(
                y=delta,
                xmin=0,
                xmax=np.max(data_x),
                linewidth=2,
                color="r",
                linestyle="--",
            )

        ax.legend()
        ax.set_xlabel(vis_headers_x[0])
        ax.grid(True)

        fig_path = os.path.join(logdir, optional_legends[i] + ".png")
        fig.savefig(fig_path, dpi=320, bbox_inches="tight")


def plot_loss(logdir, stats):
    """Plot progression of loss."""
    progress_path = os.path.join(logdir, "progress.csv")
    reader = csv.reader(open(progress_path, "rt"), delimiter=",")
    raw_data = list(reader)

    headers = raw_data[0]
    data = np.array(raw_data[1:]).astype("float")

    vis_headers_x = ["Epoch"]
    vis_header_idx_x = [headers.index(x) for x in vis_headers_x]
    data_x = data[:, vis_header_idx_x[0]]

    vis_header_idx_y = [headers.index(y) for y in stats]
    data_y_mean = []
    for idx in vis_header_idx_y:
        data_y_mean.append(data[:, idx])

    for i in range(len(data_y_mean)):
        fig, ax = plt.subplots()
        ax.plot(data_x, data_y_mean[i], color="gray", label=stats[i])

        ax.legend()
        ax.set_xlabel(vis_headers_x[0])
        ax.grid(True)

        fig_path = os.path.join(logdir, stats[i].split("/")[-1] + ".png")
        fig.savefig(fig_path, dpi=320)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input-dir", help="Log directory.")
    parser.add_argument(
        "-l",
        "--loss",
        action="store_true",
        help="Whether to plot loss stats in trainer.",
    )
    parser.add_argument(
        "--delta", type=float, help="Whether to visualize risk upper bound."
    )
    args = parser.parse_args()

    logdir = args.input_dir

    # See possible Headers in utils/stats_examples.txt
    if args.loss:
        plot_stats = ["trainer/RF1 Loss", "trainer/RF2 Loss"]
        plot_loss(logdir, plot_stats)
    else:
        plot_stats = ["Distance", "Risks", "path length"]
        for stat in plot_stats:
            vis_headers_y_mean = [
                "evaluation/{} Mean".format(stat),
                "exploration/{} Mean".format(stat),
            ]
            vis_header_y_std = [
                "evaluation/{} Std".format(stat),
                "exploration/{} Std".format(stat),
            ]
            optional_legends = [
                "Evaluation {}".format(stat),
                "Exploration {}".format(stat),
            ]
            plot(
                logdir,
                vis_headers_y_mean,
                vis_header_y_std,
                optional_legends,
                args.delta,
            )

    print("DONE")
