import pickle

import networkx as nx
import numpy as np
import pandas as pd
from repare.repare import _get_totally_ordered_partition
from sklearn.metrics import adjusted_rand_score

density = float(snakemake.wildcards.density)
samp_size = int(snakemake.wildcards.samp_size)
seed = int(snakemake.wildcards.seed)
model = pickle.load(open(snakemake.input.model, "rb"))
data = np.load(
    snakemake.input.data,
    allow_pickle=True,
)
weights = data["weights"]
targets = data["targets"]

true_dag = nx.DiGraph(weights.astype(bool))
num_nodes = true_dag.number_of_nodes()

target_des_masks = {}
for idx, target in enumerate(targets):
    mask = np.zeros(num_nodes, dtype=bool)
    descendants = nx.descendants(true_dag, target[0])
    mask[list(descendants) + [target[0]]] = True
    target_des_masks[str(idx)] = mask

ordered_masks = {str(idx): target_des_masks[str(idx)] for idx in range(len(targets))}
true_partition = _get_totally_ordered_partition(ordered_masks)
true_labels = np.zeros(num_nodes, dtype=int)
for label, part in enumerate(true_partition):
    true_labels[list(part)] = label
est_labels = np.zeros(len(true_dag))
for label, part in enumerate(model.dag.nodes):
    est_labels[list(part)] = label
ar_index = adjusted_rand_score(true_labels, est_labels)


def _is_adj(pa, ch):
    for atom in pa:
        for chatom in ch:
            if true_dag.has_edge(atom, chatom):
                return True
    return False


true_edge_est_partition = nx.create_empty_copy(model.dag)
node_list = list(true_edge_est_partition.nodes)
for idx, pa in enumerate(node_list[:-1]):
    for ch in node_list[idx + 1 :]:
        if _is_adj(pa, ch):
            true_edge_est_partition.add_edge(pa, ch)

true_positive = sum(
    (1 for edge in model.dag.edges if edge in true_edge_est_partition.edges)
)
try:
    precision = true_positive / len(model.dag.edges)
except ZeroDivisionError:
    precision = 1
try:
    recall = true_positive / len(true_edge_est_partition.edges)
except ZeroDivisionError:
    recall = 1
try:
    f_score = 2 * (precision * recall) / (precision + recall)
except ZeroDivisionError:
    f_score = 0


results = {
    "density": density,
    "samp_size": samp_size,
    "seed": seed,
    "fscore": f_score,
    "ari": ar_index,
}
pd.DataFrame([results]).to_csv(snakemake.output[0], index=False)
