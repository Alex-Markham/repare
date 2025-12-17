import numpy as np
from numpy.random import default_rng
from sempler import LGANM
from sempler.generators import dag_avg_deg, intervention_targets

seed = int(snakemake.wildcards.seed)
num_nodes = int(snakemake.wildcards.num_nodes)
samp_size = int(snakemake.wildcards.samp_size)
num_intervs = int(snakemake.params.num_intervs)
edge_probability = float(snakemake.params.edge_probability)
graph_family = str(snakemake.params.graph_family)

deg = edge_probability * (num_nodes - 1)
weights = dag_avg_deg(
    num_nodes,
    deg,
    w_min=0.5,
    w_max=2.0,
    return_ordering=False,
    random_state=seed,
)
edge_idcs = np.flatnonzero(weights)
if edge_idcs.size:
    to_neg = edge_idcs[default_rng(seed).choice([True, False], len(edge_idcs))]
    weights[np.unravel_index(to_neg, (num_nodes, num_nodes))] *= -1

model = LGANM(weights, means=(-2, 2), variances=(0.5, 2), random_state=seed)
targets = intervention_targets(
    num_nodes,
    num_intervs,
    1,
    replace=False,
    random_state=seed,
)

intervention_type = getattr(snakemake.params, "intervention_type", "soft").lower()


def sample_intervention(target):
    node = target[0]
    if intervention_type in {"hard", "do"}:
        return model.sample(samp_size, do_interventions={node: (100, 0.1)})
    if intervention_type in {"shift", "soft"}:
        return model.sample(samp_size, shift_interventions={node: (2.0, 1.0)})
    raise ValueError(f"Unknown intervention_type '{intervention_type}'")


obs_dataset = model.sample(samp_size)
interv_datasets = {
    str(idx): sample_intervention(target) for idx, target in enumerate(targets)
}

metadata = {
    "graph_family": graph_family,
    "edge_probability": edge_probability,
    "num_nodes": num_nodes,
    "samp_size": samp_size,
    "seed": seed,
    "num_intervs": num_intervs,
}

data_dict = {"obs": obs_dataset, "weights": weights, "targets": targets, **metadata}
data_dict |= interv_datasets

np.savez(snakemake.output.dataset, **data_dict)
