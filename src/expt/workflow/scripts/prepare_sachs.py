import pandas as pd

# url = "https://www.bnlearn.com/book-crc/code/sachs.interventional.txt.gz"
# df = pd.read_csv(url, sep="\t", compression="gzip")

df = pd.read_csv(snakemake.input[0], sep=" ")
df.to_pickle(snakemake.output[0])
