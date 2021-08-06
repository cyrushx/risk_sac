# Script to generate comparison plots between two models.
# Example command:
# SAC: python model/plot_result_comparisons.py

import argparse

import matplotlib

matplotlib.use("TkAgg")
import csv
import os
import pdb

import matplotlib.pyplot as plt
import numpy as np


def plot_mean_std(ax, x, y_mean, y_std, color="darkorange", alpha=0.1, label=None):
    # pdb.set_trace()
    ax.fill_between(
        x, y_mean - y_std, y_mean + y_std, color=color, alpha=alpha, label=label
    )


def plot(logdirs, plot_stats, log_names, delta=0.2):
    stats_color = ["r", "b"]
    for s, stat in enumerate(plot_stats):
        if stat == "Risks":
            vis_headers_y_mean = ["evaluation/{} Mean".format(stat)]
            vis_header_y_std = ["evaluation/{} Std".format(stat)]
        else:
            vis_headers_y_mean = ["exploration/{} Mean".format(stat)]
            vis_header_y_std = ["exploration/{} Std".format(stat)]
        fig, ax = plt.subplots()
        for l, logdir in enumerate(logdirs):
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
                plot_mean_std(
                    ax, data_x, data_y_mean[i], data_y_std[i], color=stats_color[l]
                )
                ax.plot(
                    data_x, data_y_mean[i], color=stats_color[l], label=log_names[l]
                )

                if "Risk" in vis_headers_y_mean[i] and delta is not None:
                    ax.hlines(
                        y=delta,
                        xmin=0,
                        xmax=np.max(data_x),
                        linewidth=5,
                        color="orange",
                        linestyle="--",
                    )
                    # print(delta)
                    # ax.axhline(y=delta, color='r', linestyle='-.')

                ax.legend(prop={"size": 20})
                ax.set_xlabel(vis_headers_x[0])
                ax.grid(True)

            fig_path = os.path.join(logdir, stat + ".png")
        fig.savefig(fig_path, dpi=320)


def plot_multiple(logdirs, plot_stats, log_names, delta=0.2):
    stats_color = ["r", "b"]
    for s, stat in enumerate(plot_stats):
        vis_headers_y_mean = ["evaluation/{} Mean".format(stat)]
        vis_header_y_std = ["evaluation/{} Std".format(stat)]
        fig, ax = plt.subplots()
        for l, logdir in enumerate(logdirs):
            data_y_mean_all = []
            data_y_std_all = []
            subdirs = [x[0] for x in os.walk(logdir)][1:]
            for dir in subdirs:
                progress_path = os.path.join(dir, "progress.csv")
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
                data_y_mean_all.append(data_y_mean)
                data_y_std_all.append(data_y_std)
            data_y_mean_all = np.array(data_y_mean_all)
            data_y_std_all = np.array(data_y_std_all)
            data_y_mean = np.mean(data_y_mean_all, 0)
            data_y_std = np.std(data_y_mean_all, 0)

            for i in range(len(data_y_mean)):
                plot_mean_std(
                    ax, data_x, data_y_mean[i], data_y_std[i], color=stats_color[l]
                )
                ax.plot(
                    data_x, data_y_mean[i], color=stats_color[l], label=log_names[l]
                )

                if "Risk" in vis_headers_y_mean[i] and delta is not None:
                    ax.hlines(
                        y=delta,
                        xmin=0,
                        xmax=np.max(data_x),
                        linewidth=5,
                        color="orange",
                        linestyle="--",
                    )
                    # print(delta)
                    # ax.axhline(y=delta, color='r', linestyle='-.')

                ax.legend(prop={"size": 20})
                ax.set_xlabel(vis_headers_x[0])
                ax.grid(True)

            fig_path = os.path.join(logdir, stat + ".png")
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
        "--multiple",
        action="store_true",
        help="Whether to plot results from multiple seeds.",
    )
    parser.add_argument("--delta", type=float, help="Risk upper bound.")
    args = parser.parse_args()

    if args.multiple:
        logdirs = [
            "/home/cyrushuang/workspace/risk_deeprl/external/rlkit/data/pretrained_models/Small-linear-dynamics-multiple/",
            "/home/cyrushuang/workspace/risk_deeprl/external/rlkit/data/pretrained_models/Risk-Simple-Maze-delta-0.2-risk-coeff-10.0-multiple-runs/",
        ]
        log_names = ["Standard SAC", "Risk-Bounded SAC"]
        # See possible Headers in stats_examples.txt
        plot_stats = ["Risks", "path length", "Distance"]
        plot_multiple(logdirs, plot_stats, log_names)
    else:
        logdirs = [
            "/home/cyrushuang/workspace/risk_deeprl/external/rlkit/data/pretrained_models/standard_sac_one_obstacle/",
            "/home/cyrushuang/workspace/risk_deeprl/external/rlkit/data/pretrained_models/risk_bounded_sac_one_obstacle/",
        ]
        log_names = ["Standard SAC", "Risk-Bounded SAC"]
        # See possible Headers in stats_examples.txt
        plot_stats = ["Risks", "path length", "Distance"]
        plot(logdirs, plot_stats, log_names)

    print("DONE")
