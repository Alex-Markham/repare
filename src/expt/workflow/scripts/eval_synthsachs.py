import pickle

import networkx as nx
import numpy as np
import pandas as pd
from repare.repare import _get_totally_ordered_partition
from sklearn.metrics import adjusted_rand_score


def evaluate_model(model, true_dag, labels, reference, target_names):
    # Map names to indices
    label_to_idx = {name: i for i, name in enumerate(labels)}

    # Translate target names into indices, excluding reference
    targets = [label_to_idx[name] for name in target_names if name != reference]

    # Create descendant masks for each target in the true DAG
    target_des_masks = {
        idx: np.zeros(len(true_dag), bool) for idx in range(len(targets))
    }
    for idx, target in enumerate(targets):
        des = list(true_dag.successors(target)) + [target]
        target_des_masks[idx][des] = True

    # Get ground truth partition
    true_partition = _get_totally_ordered_partition(target_des_masks)

    # Assign cluster labels to nodes based on true partition
    true_labels = np.zeros(len(true_dag), dtype=int)
    for label, part in enumerate(true_partition):
        true_labels[list(part)] = label

    # Assign estimated labels from model partition
    est_labels = np.zeros(len(true_dag), dtype=int)
    for label, part in enumerate(model.dag.nodes):
        est_labels[list(part)] = label

    # Compute ARI
    ari = adjusted_rand_score(true_labels, est_labels)

    # Optional edge F1
    learned_edges = set(model.dag.edges)
    true_edges_used = set()
    node_list = list(model.dag.nodes)
    for i, pa in enumerate(node_list[:-1]):
        for ch in node_list[i + 1 :]:
            if any(true_dag.has_edge(atom, chatom) for atom in pa for chatom in ch):
                true_edges_used.add((pa, ch))

    tp = len(learned_edges & true_edges_used)
    precision = tp / len(learned_edges) if len(learned_edges) > 0 else 1.0
    recall = tp / len(true_edges_used) if len(true_edges_used) > 0 else 1.0
    fscore = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "reference": reference,
        "ari": ari,
        "fscore": fscore,
        "num_parts": model.dag.number_of_nodes(),
        "num_edges": model.dag.number_of_edges(),
        "samp_size": snakemake.wildcards.samp_size,
        "seed": snakemake.wildcards.seed,
    }


def eval_synthsachs():
    model_in = snakemake.input[0]
    out_csv = snakemake.output[0]
    reference = snakemake.params.reference

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

    # Define interventions by their names instead of indices
    target_names = [
        "raf",
        "plc",
        "pip3",
        "pka",
        "pkc",
        "p38",
    ]  # instead of [0,2,4,7,8,9]

    true_dag = nx.DiGraph()
    true_dag.add_edges_from(true_edges)
    label_dict = {label: idx for idx, label in enumerate(labels)}
    nx.relabel_nodes(true_dag, label_dict, copy=False)

    results = evaluate_model(model, true_dag, labels, reference, target_names)
    pd.DataFrame([results]).to_csv(out_csv, index=False)


if __name__ == "__main__":
    eval_synthsachs()
