import pickle

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from repare.repare import PartitionDagModelIvn

# load inputs and params
df = pd.read_pickle(snakemake.input[0])
obs_idx = int(snakemake.wildcards.obs_idx)
alpha = float(snakemake.params.alpha)
mu = float(snakemake.params.mu)

# Prepare data_dict using obs_idx as reference
ivn_idcs = df["INT"].unique()
data_dict = {
    str(idx): df[df["INT"] == idx].drop("INT", axis=1).copy() for idx in ivn_idcs
}
data_dict["obs"] = data_dict.pop(str(obs_idx))

# fit model
model = PartitionDagModelIvn()
model.fit(data_dict, alpha, mu, disc=True)

# plot learned DAG
fig = plt.figure()
layout = nx.kamada_kawai_layout
nx.draw(model.dag, pos=layout(model.dag), ax=fig.add_subplot())
nx.draw_networkx_labels(model.dag, pos=layout(model.dag))
fig.savefig(snakemake.output.dag)

# save model
with open(snakemake.output.model, "wb") as f:
    pickle.dump(model, f)
