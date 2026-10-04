"""Offline experiments: runs every method several times and saves the comparison graphs."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import (ACCESS_CATEGORIES, CDF_ALGORITHM, DROP_STEP, GRAPH_CYCLES, GRAPH_RUNS, RESULTS_DIR,
                     SMOOTHING, TARGET_DROP, TXOP_MS, URGENCY_HOLD, URGENCY_WEIGHT)
from .methods import METHOD_KEYS, METRICS, method_label, methods_of, simulate, smoothed
from .style import AC_COLORS, ALG_COLORS
from .traffic import delay_cdf
from .urgency import MODES

TAIL = 1000   # the last 1000 cycles (5.5 s) are treated as steady state


class Experiment:
    def __init__(self):
        self.time_s = np.arange(SMOOTHING, GRAPH_CYCLES + 1) * TXOP_MS / 1000
        self.runs = {}
        for i, key in enumerate(METHOD_KEYS, 1):
            print(f"  [{i:2}/{len(METHOD_KEYS)}] {method_label(key)}")
            self.runs[key] = [simulate(key, seed, GRAPH_CYCLES) for seed in range(GRAPH_RUNS)]
        self.mean = {k: {m: np.mean([r[m] for r in runs], axis=0) for m in METRICS} for k, runs in self.runs.items()}

    def steady(self, key, metric):
        mean = self.mean[key]
        if metric == "drop":
            return 100 * mean["dropped"][-TAIL:].sum() / mean["arrived"][-TAIL:].sum()
        return mean[metric][-TAIL:].mean()

    def ac_drop(self, key, ac):
        runs = self.runs[key]
        return 100 * sum(r["ac_dropped"][ac] for r in runs) / max(sum(r["ac_arrived"][ac] for r in runs), 1)

    def delay_cdf(self, key, categories):
        hist = sum(r["delay_hist"][ac] for r in self.runs[key] for ac in categories)
        arrived = sum(r["ac_arrived"][ac] for r in self.runs[key] for ac in categories)
        return delay_cdf(hist, arrived)


def line_style(key):
    algorithm = key.split("|")[0]
    return {"color": ALG_COLORS[algorithm], "lw": 2.2, "ls": "--" if algorithm == "random" else "-"}


def time_panel(ax, exp, keys, metric, ylabel, title):
    for key in keys:
        ax.plot(exp.time_s, smoothed(exp.mean[key], metric, SMOOTHING),
                label=f"{method_label(key, short=True)}: {exp.steady(key, metric):.2f}", **line_style(key))
    ax.set(xlabel="Time [s]", ylabel=ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, fontsize=10)


def save(fig, name, title):
    note = f"average of {GRAPH_RUNS} runs, {SMOOTHING}-cycle moving average, legend value = last {TAIL} cycles"
    fig.suptitle(f"{title} ({note})", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def graph_no_urgency(exp):
    keys = methods_of("none")
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    time_panel(axes[0], exp, keys, "rate", "Data rate [Mb/s]", "Data rate")
    time_panel(axes[1], exp, keys, "drop", "Dropped packets [%]", "Packets dropped")
    for key in keys:
        rates = np.sort(np.concatenate([r["rate"][GRAPH_CYCLES // 2:] for r in exp.runs[key]]))
        axes[2].plot(rates, np.arange(1, len(rates) + 1) / len(rates), label=method_label(key), **line_style(key))
    axes[2].set(xlabel="Data rate per TXOP [Mb/s]", ylabel="CDF")
    axes[2].set_title("CDF of the data rate (second half of each run)", loc="left", fontweight="bold")
    axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, fontsize=10)
    save(fig, "graph_no_urgency.png", "Without urgency")


def graph_urgency(exp, mode, name):
    keys = methods_of(mode)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    if mode == "drop":
        time_panel(axes[0], exp, keys, "rate", "Data rate [Mb/s]", "Data rate")
        time_panel(axes[1], exp, keys, "drop", "Dropped packets [%]", "Packets dropped")
        time_panel(axes[2], exp, keys, "mean_u", "Mean AP urgency level", "AP urgency level")
        title = (f"{MODES[mode]} (per AP), w = {URGENCY_WEIGHT:g}, step {DROP_STEP:g}, target {TARGET_DROP:g}, "
                 f"hold {URGENCY_HOLD}")
    else:
        time_panel(axes[0], exp, keys, "utility", "Utility r + w·u [Mb/s]", "Utility")
        time_panel(axes[1], exp, keys, "u", "Urgency of the served partner station", "Urgency served")
        time_panel(axes[2], exp, keys, "drop", "Dropped packets [%]", "Packets dropped")
        title = f"{MODES[mode]}, w = {URGENCY_WEIGHT:g}"
    save(fig, name, title)


def graph_cdf(exp):
    fig, axes = plt.subplots(2, 4, figsize=(24, 10.5))
    for col, mode in enumerate(MODES):
        top, bottom = axes[0, col], axes[1, col]
        for key in methods_of(mode):
            top.step(*exp.delay_cdf(key, ACCESS_CATEGORIES), where="post",
                     label=method_label(key, short=True), **line_style(key))
        key = f"{CDF_ALGORITHM}|{mode}"
        for ac, info in ACCESS_CATEGORIES.items():
            bottom.step(*exp.delay_cdf(key, [ac]), where="post", color=AC_COLORS[ac], lw=2.2,
                        label=f"{ac} ({info['name']})")
        for ax, title in ((top, f"{MODES[mode]}: all packets"), (bottom, f"{method_label(key)}:\nper access category")):
            ax.set(xlabel="Packet delay [ms]", ylabel="Packets delivered within delay [%]", ylim=(0, 100))
            ax.set_title(title, loc="left", fontweight="bold")
            ax.legend(loc="lower right", fontsize=10)
    fig.suptitle("CDF of packet delay, second half of each run. Dropped packets never arrive, so each curve "
                 "ends at the delivered share.", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "graph_cdf.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_table(exp):
    print(f"\n{'method':<34}{'rate':>8}{'utility':>9}{'u served':>10}{'dropped %':>11}{'VO drop %':>11}")
    for key in METHOD_KEYS:
        print(f"{method_label(key):<34}{exp.steady(key, 'rate'):8.2f}{exp.steady(key, 'utility'):9.2f}"
              f"{exp.steady(key, 'u'):10.3f}{exp.steady(key, 'drop'):11.2f}{exp.ac_drop(key, 'AC_VO'):11.2f}")


def make_graphs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False})
    print(f"Running {len(METHOD_KEYS)} methods x {GRAPH_RUNS} runs x {GRAPH_CYCLES} cycles ...")
    exp = Experiment()
    graph_no_urgency(exp)
    graph_urgency(exp, "app", "graph_app_urgency.png")
    graph_urgency(exp, "drop", "graph_drop_urgency.png")
    graph_urgency(exp, "combined", "graph_combined_urgency.png")
    graph_cdf(exp)
    print_table(exp)
    print(f"\nGraphs saved in {RESULTS_DIR}/")
