import matplotlib.pyplot as plt
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

results_df = pd.read_csv(snakemake.input[0])

plt.figure()
sns.lineplot(
    data=results_df,
    x="samp_size",
    y="fscore",
    marker="o",
    hue="density",
    estimator="median",
    errorbar="ci",
)
plt.xscale("log")
plt.ylim(0, 1)
plt.ylabel("F-score ↑")
plt.xlabel("Sample Size (log)")
plt.legend(title="density", fontsize=FONT_SIZE, title_fontsize=FONT_SIZE)
plt.tight_layout()
plt.savefig(snakemake.output[0])

plt.figure()
sns.lineplot(
    data=results_df,
    x="samp_size",
    y="ari",
    marker="o",
    hue="density",
    estimator="median",
    errorbar="ci",
)
plt.xscale("log")
plt.ylim(0, 1)
plt.ylabel("adjusted Rand index ↑")
plt.xlabel("Sample Size (log)")
plt.legend(title="density", fontsize=FONT_SIZE, title_fontsize=FONT_SIZE)
plt.tight_layout()
plt.savefig(snakemake.output[1])
