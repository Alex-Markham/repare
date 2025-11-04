import pickle

import numpy as np
from repare.repare import PartitionDagModelIvn

# load params and input
alpha = float(snakemake.params.alpha)
beta = float(snakemake.params.beta)
assume = snakemake.params.assume
data = np.load(snakemake.input.data, allow_pickle=True)

# fit
targets = data["targets"]
data_dict = {"obs": (data["obs"], set())}
for idx, target in enumerate(targets):
    tgt = set(np.atleast_1d(target).astype(int))
    data_dict[str(idx)] = (data[str(idx)], tgt)
model = PartitionDagModelIvn()
model.fit(data_dict, alpha, beta, assume)

# save output
with open(snakemake.output[0], "wb") as f:
    pickle.dump(model, f)
