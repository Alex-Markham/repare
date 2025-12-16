import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FONT_SIZE = 16

sns.set_context("paper", font_scale=1)
plt.rcParams.update(
    {
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "legend.title_fontsize": FONT_SIZE,
    }
)


def _style_generator():
    #colors = sns.color_palette("colorblind")
    colors = sns.color_palette("ch:s=-.2,r=.6", n_colors=10)

    markers = ["o", "s", "^", "D", "P", "X", "v", ">", "<", "*"]
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 5))]
    for idx in itertools.count():
        yield (
            colors[idx % len(colors)],
            markers[idx % len(markers)],
            linestyles[idx % len(linestyles)],
        )


_STYLE_SEQUENCE = _style_generator()
_STYLE_CACHE = {}


def _get_style(key):
    if key not in _STYLE_CACHE:
        _STYLE_CACHE[key] = next(_STYLE_SEQUENCE)
    return _STYLE_CACHE[key]


def _empty_plot(path: str, title: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, "No data", ha="center", va="center")
    ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


df = pd.read_csv(snakemake.input.summary)
method_label = snakemake.params.method_label

if df.empty:
    _empty_plot(
        snakemake.output.sample_plot, f"{method_label} failure sweep (sample size)"
    )
    _empty_plot(
        snakemake.output.runtime_nodes_plot,
        f"{method_label} runtime vs. node count",
    )
    raise SystemExit(0)

stats_cols = {"ari_mean", "runtime_mean"}
if stats_cols.issubset(df.columns):
    agg = (
        df.sort_values(["num_nodes", "samp_size"])
        .drop_duplicates(["num_nodes", "samp_size"], keep="last")
        [["num_nodes", "samp_size", "ari_mean", "runtime_mean"]]
        .sort_values("samp_size")
    )
else:
    agg = (
        df.groupby(["num_nodes", "samp_size"])
        .agg(
            ari_mean=("ari", "median"),
            runtime_mean=("runtime_sec", "median"),
        )
        .reset_index()
        .sort_values("samp_size")
    )

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

for idx, (num_nodes, subset) in enumerate(agg.groupby("num_nodes")):
    subset = subset.sort_values("samp_size")
    ari_mean = subset["ari_mean"].to_numpy()
    runtime_mean = subset["runtime_mean"].to_numpy()
    color, marker, linestyle = _get_style(("nodes", num_nodes))
    axes[0].plot(
        subset["samp_size"],
        ari_mean,
        marker=marker,
        linestyle=linestyle,
        color=color,
        label=f"d={num_nodes}",
    )
    axes[1].plot(
        subset["samp_size"],
        runtime_mean,
        marker=marker,
        linestyle=linestyle,
        color=color,
        label=f"d={num_nodes}",
    )

axes[0].set_xscale("log")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[0].set_ylim(0, 1)
axes[0].set_xlabel("sample size")
axes[0].set_ylabel("ARI ↑")
axes[1].set_xlabel("sample size")
axes[1].set_ylabel("runtime (s, log) ↓")

axes[0].grid(True, which="both", linestyle="--", alpha=0.4)
axes[1].grid(True, which="both", linestyle="--", alpha=0.4)
node_levels = agg["num_nodes"].unique()
axes[1].legend(
    title="node count",
    loc="upper left",
    frameon=True,
    fontsize=FONT_SIZE,
    title_fontsize=FONT_SIZE,
    #bbox_to_anchor=(0.5, -0.25),
    #ncol=len(node_levels),
    ncol = 1
)

fig.tight_layout()
fig.savefig(snakemake.output.sample_plot)
plt.close(fig)

runtime_vs_nodes = agg.copy()

fig2, ax = plt.subplots(figsize=(7, 5))
for samp_idx, (samp_size, subset) in enumerate(runtime_vs_nodes.groupby("samp_size")):
    subset = subset.sort_values("num_nodes")
    runtime_mean = subset["runtime_mean"].to_numpy()
    color, marker, linestyle = _get_style(("samp", samp_size))
    ax.plot(
        subset["num_nodes"],
        runtime_mean,
        marker=marker,
        linestyle=linestyle,
        color=color,
        label=f"m={samp_size}",
    )

ax.set_xlabel("node count")
ax.set_ylabel("runtime (s, log) ↓")
ax.set_yscale("log")
ax.grid(True, linestyle="--", alpha=0.4)
sample_levels = runtime_vs_nodes["samp_size"].unique()
ax.legend(
    title="sample size",
    loc="upper center",
    #bbox_to_anchor=(0.5, -0.15),
    #ncol=len(sample_levels),
    ncol=1,
    fontsize=FONT_SIZE,
    title_fontsize=FONT_SIZE,
)

fig2.tight_layout()
fig2.savefig(snakemake.output.runtime_nodes_plot)
plt.close(fig2)
