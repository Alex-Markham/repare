import pandas as pd

# load inputs
input_csv = snakemake.input[0]
output_tex = snakemake.output[0]

df = pd.read_csv(input_csv)


def dataframe_to_custom_latex(df):
    header = r"""
\begin{tabular}{ccccc}
    \toprule
    Reference & Parts & Edges & F-score & ARI   \\
    \midrule
"""
    footer = r"""
    \bottomrule
\end{tabular}
"""

    body_rows = ""
    for _, row in df.iterrows():
        reference = str(row["reference"])
        parts = int(row["num_parts"])
        edges = int(row["num_edges"])
        fscore = f"{row['fscore']:.2f}"
        ari = f"{row['ari']:.2f}"

        body_rows += f"\t{reference} & {parts} & {edges} & {fscore} & {ari} \\\\\n"

    return header + body_rows + footer


latex_table = dataframe_to_custom_latex(df)

# save output
with open(output_tex, "w") as f:
    f.write(latex_table)
