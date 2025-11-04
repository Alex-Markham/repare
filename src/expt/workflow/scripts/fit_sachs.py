import pickle

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from repare.repare import PartitionDagModelIvn

# load inputs and params
df = pd.read_pickle(snakemake.input[0])
obs_idx = int(snakemake.wildcards.obs_idx)
alpha = float(snakemake.params.alpha)
beta = float(snakemake.params.beta)
assume = snakemake.params.assume

# Prepare data_dict using obs_idx as reference
ivn_idcs = df["INT"].unique()
data_dict = {}
for idx in ivn_idcs:
    subset = df[df["INT"] == idx].drop("INT", axis=1).to_numpy()
    if idx == obs_idx:
        data_dict["obs"] = (subset, set())
    else:
        data_dict[str(idx)] = (subset, {int(idx)})

# fit model
model = PartitionDagModelIvn()
model.fit(data_dict, alpha, beta, assume)

# plot learned DAG
fig = plt.figure()
layout = nx.kamada_kawai_layout
nx.draw(model.dag, pos=layout(model.dag), ax=fig.add_subplot())
nx.draw_networkx_labels(model.dag, pos=layout(model.dag))
fig.savefig(snakemake.output.dag)

# save model
with open(snakemake.output.model, "wb") as f:
    pickle.dump(model, f)
