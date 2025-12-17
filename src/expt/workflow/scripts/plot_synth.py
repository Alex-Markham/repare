import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_context("paper", font_scale=2.3)

results_df = pd.read_csv(snakemake.input[0])

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
plt.xlabel("sample size (n)")
plt.legend(title="density")
plt.tight_layout()
plt.savefig(snakemake.output[0], bbox_inches="tight", pad_inches=0.02)


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
plt.ylabel("ARI ↑")
plt.xlabel("sample size (n)")
plt.legend(title="density")
plt.tight_layout()
plt.savefig(snakemake.output[1], bbox_inches="tight", pad_inches=0.02)
