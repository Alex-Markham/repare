import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_ari_fscore_combined():
    input_csv = snakemake.input[0]
    output_file = snakemake.output[0]

    df = pd.read_csv(input_csv)
    long_df = df.melt(
        id_vars=["samp_size", "seed"],  # include seed as grouping var
        value_vars=["fscore", "ari"],
        var_name="metric",
        value_name="score",
    )

    sns.set_context("paper", font_scale=1.5)
    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=long_df,
        x="samp_size",
        y="score",
        hue="metric",
        style="metric",
        size="samp_size",
        sizes=(1, 4),
        palette="muted",
        markers=True,
        dashes=False,
        legend="full",
        errorbar="sd",
    )
    plt.xlabel("Sample Size")
    plt.ylabel("Score")
    plt.title("F-score and Adjusted Rand Index vs Sample Size")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


if __name__ == "__main__":
    plot_ari_fscore_combined()
