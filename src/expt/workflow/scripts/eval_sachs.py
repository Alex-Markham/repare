import pickle

import networkx as nx
import numpy as np
import pandas as pd
from repare.repare import _get_totally_ordered_partition
from sklearn.metrics import adjusted_rand_score

# load inputs and params
model_path = snakemake.input[0]
obs_idx = int(snakemake.wildcards.obs_idx)
output_path = snakemake.output[0]

with open(model_path, "rb") as f:
    model = pickle.load(f)

# ground truth
true_edges = [
    ("raf", "mek"),
    ("pka", "akt"),
    ("pka", "erk"),
    ("pka", "jnk"),
    ("pka", "mek"),
    ("pka", "p38"),
    ("pka", "raf"),
    ("plc", "pip2"),
    ("plc", "pkc"),
    ("pip3", "akt"),
    ("pip3", "pip2"),
    ("pip3", "plc"),
    ("pip2", "pkc"),
    ("pkc", "jnk"),
    ("pkc", "mek"),
    ("pkc", "p38"),
    ("pkc", "pka"),
    ("pkc", "raf"),
    ("mek", "erk"),
    ("erk", "akt"),
]
true_dag = nx.DiGraph()
true_dag.add_edges_from(true_edges)
labels = [
    "raf",
    "mek",
    "plc",
    "pip2",
    "pip3",
    "erk",
    "akt",
    "pka",
    "pkc",
    "p38",
    "jnk",
]
label_dict = {label: idx for idx, label in enumerate(labels)}
nx.relabel_nodes(true_dag, label_dict, copy=False)


# evaluate
targets = [0, 2, 4, 7, 8, 9]
targets.remove(obs_idx)

target_des_masks = {idx: np.zeros(len(true_dag), bool) for idx in range(len(targets))}
for idx in target_des_masks:
    target = targets[idx]
    des = list(true_dag.successors(target)) + [target]
    target_des_masks[idx][des] = True

true_partition = _get_totally_ordered_partition(target_des_masks)

true_labels = np.zeros(len(true_dag))
for label, part in enumerate(true_partition):
    true_labels[list(part)] = label

est_labels = np.zeros(len(true_dag))
for label, part in enumerate(model.dag.nodes):
    est_labels[list(part)] = label

ari = adjusted_rand_score(true_labels, est_labels)


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

fscore = 2 * (precision * recall) / (precision + recall)

num_nodes = model.dag.number_of_nodes()
num_edges = model.dag.number_of_edges()


# save outputs
df = pd.DataFrame(
    [
        {
            "reference": labels[obs_idx],
            "ari": ari,
            "fscore": fscore,
            "num_parts": num_nodes,
            "num_edges": num_edges,
        }
    ]
)
df.to_csv(output_path, index=False)
