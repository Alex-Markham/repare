import itertools

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# pick a palette with good categorical separation
sns.set_palette("colorblind")  # or "colorblind", "Dark2", etc.

# global cycler for markers & linestyles if you want matplotlib to help
markers = itertools.cycle(["o", "s", "D", "^", "v"])
linestyles = itertools.cycle(["-", "--", ":", "-."])

sns.set_context("paper", font_scale=2.3)

results_df = pd.read_csv(snakemake.input[0])

# ARI vs Sample Size
plt.figure()
ariplot = sns.lineplot(
    data=results_df,
    y="ari",
    x="samp_size",
    marker="o",
    hue="num_nodes",
    style="num_nodes",
    markers=True,
    estimator="median",
    errorbar="ci",
    linewidth=2.0,
    markersize=6,
)
plt.xscale("log")
plt.ylim(0, 1)
plt.ylabel("ARI ↑")
plt.xlabel("sample size (n)")
plt.legend(title="nodes (d)", loc="upper left")
plt.tight_layout()
plt.savefig(snakemake.output["ari_samp"], bbox_inches="tight", pad_inches=0.02)

# Run Time vs Sample Size
plt.figure()
ariplot = sns.lineplot(
    data=results_df,
    y="runtime_sec",
    x="samp_size",
    marker="o",
    hue="num_nodes",
    style="num_nodes",
    markers=True,
    estimator="median",
    errorbar="ci",
    linewidth=2.0,
    markersize=6,
)

plt.ylabel("run time (s)")
plt.yscale("log")
plt.xlabel("sample size (n)")
plt.xscale("log")
plt.legend(title="nodes (d)", loc="upper left")
plt.tight_layout()
plt.savefig(snakemake.output["time_samp"], bbox_inches="tight", pad_inches=0.02)


#  Run Time vs Num Nodes
plt.figure()
ariplot = sns.lineplot(
    data=results_df,
    y="runtime_sec",
    x="num_nodes",
    marker="o",
    hue="samp_size",
    style="samp_size",
    markers=True,
    estimator="median",
    errorbar="ci",
    linewidth=2.0,
    markersize=6,
)

plt.ylabel("run time (s)")
plt.yscale("log")
plt.xlabel("number of nodes (d)")
plt.xscale("log")
plt.legend(title="sample size (n)", loc="upper left")
plt.tight_layout()
plt.savefig(snakemake.output["time_nodes"], bbox_inches="tight", pad_inches=0.02)
