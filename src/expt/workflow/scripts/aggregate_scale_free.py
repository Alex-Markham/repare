import pandas as pd

frames = [pd.read_csv(path) for path in snakemake.input]
result = pd.concat(frames, ignore_index=True)
result.to_csv(snakemake.output[0], index=False)
