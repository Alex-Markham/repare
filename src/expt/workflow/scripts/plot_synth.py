import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

results_df = pd.read_csv(snakemake.input[0])
sns.set_context("paper", font_scale=1.5)

# F-score plot
plt.figure()
fplot = sns.lineplot(
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
plt.tight_layout()
plt.savefig(snakemake.output[0])


# ARI plot
plt.figure()
ariplot = sns.lineplot(
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
plt.tight_layout()
plt.savefig(snakemake.output[1])
