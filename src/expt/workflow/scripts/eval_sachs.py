import pickle

import networkx as nx
import numpy as np
import pandas as pd
from repare.repare import _get_totally_ordered_partition
from sklearn.metrics import adjusted_rand_score


def evaluate_model(model, true_dag, labels, obs_idx):
    # Identify intervention targets except observational index
    targets = [0, 2, 4, 7, 8, 9]
    targets.remove(obs_idx)

    # Create descendant masks for each target in true DAG
    target_des_masks = {}
    for idx, target in enumerate(targets):
        mask = np.zeros(len(true_dag), dtype=bool)
        descendants = nx.descendants(true_dag, target)
        mask[list(descendants) + [target]] = True
        target_des_masks[str(idx)] = mask

    # Get ground truth partition
    ordered_masks = {
        str(idx): target_des_masks[str(idx)] for idx in range(len(targets))
    }
    true_partition = _get_totally_ordered_partition(ordered_masks)

    # Create true labels for each node in the graph
    true_labels = np.zeros(len(true_dag))
    for label, part in enumerate(true_partition):
        true_labels[list(part)] = label

    # Create estimated labels from model partition (list of node sets)
    est_labels = np.zeros(len(true_dag))
    for label, part in enumerate(model.dag.nodes):
        est_labels[list(part)] = label

    # Compute ARI
    ari = adjusted_rand_score(true_labels, est_labels)

    # Helper to check adjacency in true DAG between node sets
    def _is_adj(pa, ch):
        for atom in pa:
            for chatom in ch:
                if true_dag.has_edge(atom, chatom):
                    return True
        return False

    # Construct DAG from true edges that appear in model partition edges
    true_edge_est_partition = nx.create_empty_copy(model.dag)
    node_list = list(true_edge_est_partition.nodes)
    for idx, pa in enumerate(node_list[:-1]):
        for ch in node_list[idx + 1 :]:
            if _is_adj(pa, ch):
                true_edge_est_partition.add_edge(pa, ch)

    # Compute precision, recall, and fscore
    true_positive = sum(
        (1 for edge in model.dag.edges if edge in true_edge_est_partition.edges)
    )
    precision = (
        true_positive / len(model.dag.edges) if len(model.dag.edges) > 0 else 1.0
    )
    recall = (
        true_positive / len(true_edge_est_partition.edges)
        if len(true_edge_est_partition.edges) > 0
        else 1.0
    )
    fscore = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "reference": labels[obs_idx],
        "ari": ari,
        "fscore": fscore,
        "num_parts": model.dag.number_of_nodes(),
        "num_edges": model.dag.number_of_edges(),
    }


def eval_synthsachs():
    model_in = snakemake.input[0]
    out_csv = snakemake.output[0]
    obs_idx = int(snakemake.wildcards.obs_idx)

    with open(model_in, "rb") as f:
        model = pickle.load(f)

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

    true_dag = nx.DiGraph()
    true_dag.add_edges_from(true_edges)
    label_dict = {label: idx for idx, label in enumerate(labels)}
    nx.relabel_nodes(true_dag, label_dict, copy=False)

    results = evaluate_model(model, true_dag, labels, obs_idx)

    df = pd.DataFrame([results])
    df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    eval_synthsachs()
