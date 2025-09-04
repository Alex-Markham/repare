import pandas as pd

dfs = [pd.read_csv(f) for f in snakemake.input]
df_final = pd.concat(dfs, ignore_index=True)
df_final.to_csv(snakemake.output[0], index=False)
