import pickle

import numpy as np
from repare.repare import PartitionDagModelIvn

# load params and input
alpha = float(snakemake.params.alpha)
mu = float(snakemake.params.alpha)
normalize = float(snakemake.params.normalize)
data = np.load(snakemake.input.data, allow_pickle=True)

# fit
data_dict = {k: data[k] for k in data.files if k not in ("weights", "targets")}
model = PartitionDagModelIvn()
model.fit(data_dict, alpha, mu, normalize)

# save output
with open(snakemake.output[0], "wb") as f:
    pickle.dump(model, f)
