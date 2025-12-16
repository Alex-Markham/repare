import pickle
import time

import numpy as np
from repare.repare import PartitionDagModelIvn

alpha = float(snakemake.params.alpha)
beta = float(snakemake.params.beta)
assume_param = snakemake.params.assume
assume = None if assume_param in (None, "None", "none") else assume_param
refine_param = getattr(snakemake.params, "refine_test", "ks")
refine_test = ("ks" if refine_param is None else str(refine_param)).lower()
intervention_param = getattr(snakemake.params, "intervention_type", "hard")


def _map_type(value: str) -> str:
    lowered = str(value).lower()
    if lowered in {"shift", "soft"}:
        return "soft"
    if lowered in {"hard", "do"}:
        return "hard"
    return lowered


intervention_type = _map_type(intervention_param)
data = np.load(snakemake.input.data, allow_pickle=True)

targets = data["targets"]
data_dict = {"obs": (data["obs"], set(), "obs")}
for idx, target in enumerate(targets):
    tgt = set(np.atleast_1d(target).astype(int))
    data_dict[str(idx)] = (data[str(idx)], tgt, intervention_type)
model = PartitionDagModelIvn()
start = time.perf_counter()
model.fit(data_dict, alpha, beta, assume, refine_test=refine_test)
model.fit_runtime_sec = time.perf_counter() - start

with open(snakemake.output[0], "wb") as f:
    pickle.dump(model, f)
