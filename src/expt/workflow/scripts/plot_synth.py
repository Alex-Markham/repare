import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

results_df = pd.read_csv(snakemake.input[0])
sns.set_context("paper", font_scale=1.5)
fplot = sns.catplot(
    data=results_df,
    x="density",
    y="fscore",
    kind="box",
    hue="alg"
    if "alg" in results_df.columns
    else "samp_size"
    if "samp_size" in results_df.columns
    else None,
    row="num_nodes" if "num_nodes" in results_df.columns else None,
    palette="RdPu",
)
fplot.set_titles("")
fplot.set_ylabels("F-score ↑")
fplot.tight_layout()
fplot.figure.savefig(snakemake.output[0])

ariplot = sns.catplot(
    data=results_df,
    x="density",
    y="ari",
    kind="box",
    hue="alg"
    if "alg" in results_df.columns
    else "samp_size"
    if "samp_size" in results_df.columns
    else None,
    row="num_nodes" if "num_nodes" in results_df.columns else None,
    palette="RdPu",
)
ariplot.set_titles("")
ariplot.set_ylabels("adjusted Rand index ↑")
ariplot.tight_layout()
ariplot.figure.savefig(snakemake.output[1])
