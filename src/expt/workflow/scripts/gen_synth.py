import numpy as np
from numpy.random import default_rng
from sempler import LGANM
from sempler.generators import dag_avg_deg, intervention_targets

seed = int(snakemake.wildcards.seed)
density = float(snakemake.wildcards.density)
samp_size = int(snakemake.wildcards.samp_size)
num_nodes = int(snakemake.params.num_nodes)
num_intervs = int(snakemake.params.num_intervs)

deg = density * (num_nodes - 1)
weights = dag_avg_deg(
    num_nodes, deg, w_min=0.5, w_max=2, return_ordering=False, random_state=seed
)
edge_idcs = np.flatnonzero(weights)
to_neg = edge_idcs[default_rng(seed).choice([True, False], len(edge_idcs))]
weights[np.unravel_index(to_neg, (num_nodes, num_nodes))] *= -1
model = LGANM(weights, means=(-2, 2), variances=(0.5, 2), random_state=seed)
targets = intervention_targets(num_nodes, num_intervs, 1, random_state=seed)
obs_dataset = model.sample(samp_size)
interv_datasets = {
    str(idx): model.sample(samp_size, shift_interventions={target[0]: (2, 1)})
    for idx, target in enumerate(targets)
}
data_dict = {"obs": obs_dataset} | interv_datasets
np.savez(snakemake.output[0], weights=weights, targets=targets, **data_dict)
