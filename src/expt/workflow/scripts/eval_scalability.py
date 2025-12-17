import json
import pickle

import networkx as nx
import numpy as np
import pandas as pd
from repare.repare import _get_totally_ordered_partition
from sklearn.metrics import adjusted_rand_score

model = pickle.load(open(snakemake.input.model, "rb"))
data = np.load(snakemake.input.data, allow_pickle=True)
with open(snakemake.input.fit_metadata, "r", encoding="utf-8") as f:
    fit_metadata = json.load(f)

weights = data["weights"]
targets = data["targets"]
graph_family = str(data["graph_family"])
edge_probability = float(data["edge_probability"])
num_nodes = int(data["num_nodes"])
num_intervs = int(data["num_intervs"])
samp_size = int(data["samp_size"])
seed = int(data["seed"])

true_dag = nx.DiGraph(weights.astype(bool))

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

est_labels = np.zeros(num_nodes, dtype=int)
for label, part in enumerate(model.dag.nodes):
    est_labels[list(part)] = label

ari = adjusted_rand_score(true_labels, est_labels)


def _is_adj(pas, chs):
    for pa in pas:
        for ch in chs:
            if true_dag.has_edge(pa, ch):
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
precision = true_positive / len(model.dag.edges) if model.dag.edges else 1
recall = (
    true_positive / len(true_edge_est_partition.edges)
    if true_edge_est_partition.edges
    else 1
)
try:
    fscore = 2 * (precision * recall) / (precision + recall)
except ZeroDivisionError:
    fscore = 0

results = {
    "graph_family": graph_family,
    "edge_probability": edge_probability,
    "num_nodes": num_nodes,
    "samp_size": samp_size,
    "num_intervs": num_intervs,
    "seed": seed,
    "method": snakemake.params.method,
    "method_label": snakemake.params.method_label,
    "precision": precision,
    "recall": recall,
    "fscore": fscore,
    "ari": ari,
    "runtime_sec": float(fit_metadata.get("runtime_sec", np.nan)),
    "score": float(getattr(model, "score", np.nan)),
}

pd.DataFrame([results]).to_csv(snakemake.output.metrics, index=False)
