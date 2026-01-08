#!/usr/bin/env python3
"""Run RePaRe grid (grouped or ungrouped mode)."""

import json
import pickle
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from gnies.scores.gnies_score import GnIESScore
from repare.repare import PartitionDagModelIvn, _get_totally_ordered_partition
from sklearn.metrics import adjusted_rand_score

matplotlib.use("Agg")


def partition_edge_metrics(model_dag, true_graph):
    """Precision/recall/F1 for partitions."""
    true_edge_partition = nx.create_empty_copy(model_dag)
    node_list = list(true_edge_partition.nodes)
    for i, pa in enumerate(node_list[:-1]):
        for ch in node_list[i + 1 :]:
            if any(true_graph.has_edge(u, v) for u in pa for v in ch):
                true_edge_partition.add_edge(pa, ch)
    tp = sum(1 for edge in model_dag.edges if edge in true_edge_partition.edges)
    precision = tp / len(model_dag.edges) if model_dag.edges else 1.0
    recall = tp / len(true_edge_partition.edges) if true_edge_partition.edges else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def build_data_dict(blocks, group_targets):
    """Build input dict for RePaRe."""
    data = {"obs": (blocks["obs"], set(), "obs")}
    for label, targets in group_targets.items():
        data[label] = (blocks[label], targets, "soft")
    return data


def save_dag_plot(model, feature_cols, path):
    """Save DAG as PNG."""
    labeled_dag = nx.relabel_nodes(
        model.dag,
        {node: tuple(feature_cols[idx] for idx in node) for node in model.dag.nodes},
        copy=True,
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    pos = nx.spring_layout(labeled_dag, seed=0)
    nx.draw_networkx(
        labeled_dag,
        pos=pos,
        ax=ax,
        node_color="#8fbcd4",
        edgecolors="#1f4b73",
        linewidths=1.0,
        font_size=8,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def labeled_summary(model, feature_cols):
    """Extract labeled nodes and edges."""
    labeled = []
    for idx, part in enumerate(model.dag.nodes):
        labels = tuple(feature_cols[i] for i in part)
        labeled.append((idx, labels))
    edges = []
    node_to_idx = {node: idx for idx, node in enumerate(model.dag.nodes)}
    for u, v in model.dag.edges:
        edges.append((node_to_idx[u], node_to_idx[v]))
    return labeled, edges


def ground_truth_partition(target_dict, parts, true_dag_full):
    """Compute ground-truth partition labels."""
    ordered_masks = {}
    for order, label in enumerate(sorted(target_dict)):
        atom_union = set().union(*[parts[idx] for idx in target_dict[label]])
        closure = set(atom_union)
        for idx in list(atom_union):
            closure.update(nx.descendants(true_dag_full, idx))
        mask = np.zeros(len(parts), dtype=bool)
        for part_idx, atoms in enumerate(parts):
            if atoms & closure:
                mask[part_idx] = True
        ordered_masks[str(order)] = mask
    partition = _get_totally_ordered_partition(ordered_masks)
    labels = np.zeros(len(parts), dtype=int)
    for label, block in enumerate(partition):
        labels[list(block)] = label
    return partition, labels


def run_repare_grid(
    blocks,
    partition_parts,
    group_targets,
    true_graph,
    true_labels,
    true_dag_full,
    single_env_labels,
    single_env_targets,
    alphas,
    betas,
    feature_cols,
    mode="grouped",
):
    """Run RePaRe grid search."""
    records = []
    models = {}

    # Setup targets
    if mode == "grouped":
        targets = group_targets
        labels = true_labels
    else:  # ungrouped
        targets = {
            label: {next(iter(single_env_targets[label]))}
            for label in single_env_labels
        }
        _, labels = ground_truth_partition(targets, partition_parts, true_dag_full)

    # Setup GnIES score
    env_labels = list(targets)
    gnies_data = [blocks["obs"]]
    for label in env_labels:
        gnies_data.append(blocks[label])
    intervention_union = set().union(*targets.values()) if targets else set()
    gnies_score_class = GnIESScore(
        gnies_data, intervention_union, lmbda=0.0, centered=True
    )

    # Grid search
    for alpha in alphas:
        for beta in betas:
            start = time.perf_counter()
            model = PartitionDagModelIvn().fit(
                build_data_dict(blocks, targets),
                alpha=float(alpha),
                beta=float(beta),
                assume="gaussian",
                refine_test="ks",
            )
            fit_time = time.perf_counter() - start

            est_labels = np.zeros(len(partition_parts), dtype=int)
            for label, block in enumerate(model.dag.nodes):
                est_labels[list(block)] = label

            ari = adjusted_rand_score(labels, est_labels)
            edge_stats = partition_edge_metrics(model.dag, true_graph)

            expanded_adj = model.expand_coarsened_dag(fully_connected=True)
            score_value = -float(gnies_score_class.full_score(expanded_adj))

            records.append(
                {
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "ari": float(ari),
                    "score": score_value,
                    "fit_time": fit_time,
                    "num_parts": model.dag.number_of_nodes(),
                    "num_edges": model.dag.number_of_edges(),
                    **edge_stats,
                }
            )
            models[(float(alpha), float(beta))] = model

    df = pd.DataFrame(records)
    score_row = min(records, key=lambda r: (r["score"], -r["ari"]))
    oracle_row = max(
        records, key=lambda r: (r["ari"], r["f1"], r["precision"], -r["score"])
    )

    score_model = models[(score_row["alpha"], score_row["beta"])]
    oracle_model = models[(oracle_row["alpha"], oracle_row["beta"])]

    score_parts, score_edges = labeled_summary(score_model, feature_cols)
    oracle_parts, oracle_edges = labeled_summary(oracle_model, feature_cols)

    return (
        df,
        score_model,
        oracle_model,
        score_row,
        oracle_row,
        score_parts,
        score_edges,
        oracle_parts,
        oracle_edges,
    )


def main():
    # Load preprocessed data
    with open(snakemake.input.blocks, "rb") as f:
        blocks = pickle.load(f)
    with open(snakemake.input.grouptargets, "rb") as f:
        group_targets = pickle.load(f)
    with open(snakemake.input.partition, "rb") as f:
        partition_parts = pickle.load(f)
    with open(snakemake.input.truegraph, "rb") as f:
        true_graph = pickle.load(f)
    with open(snakemake.input.truelabels, "rb") as f:
        true_labels = pickle.load(f)
    with open(snakemake.input.truedagfull, "rb") as f:
        true_dag_full = pickle.load(f)
    with open(snakemake.input.singleenvlabels, "rb") as f:
        single_env_labels = pickle.load(f)
    with open(snakemake.input.singleenvtargets, "rb") as f:
        single_env_targets = pickle.load(f)
    with open(snakemake.input.features, "r") as f:
        feature_cols = json.load(f)["feature_cols"]

    alphas = [float(a) for a in snakemake.params.alphas]
    betas = [float(b) for b in snakemake.params.betas]
    mode = snakemake.params.mode

    # Run grid
    (
        df,
        score_model,
        oracle_model,
        score_row,
        oracle_row,
        score_parts,
        score_edges,
        oracle_parts,
        oracle_edges,
    ) = run_repare_grid(
        blocks,
        partition_parts,
        group_targets,
        true_graph,
        true_labels,
        true_dag_full,
        single_env_labels,
        single_env_targets,
        alphas,
        betas,
        feature_cols,
        mode,
    )

    # Save metrics CSV
    df.to_csv(snakemake.output.metrics_csv, index=False)

    # Save DAGs
    save_dag_plot(score_model, feature_cols, Path(snakemake.output.score_dag))
    save_dag_plot(oracle_model, feature_cols, Path(snakemake.output.oracle_dag))

    # Save models
    with open(snakemake.output.score_model, "wb") as f:
        pickle.dump(score_model, f)
    with open(snakemake.output.oracle_model, "wb") as f:
        pickle.dump(oracle_model, f)

    # Save score/oracle params with structure for summary
    score_params = {
        "alpha": score_row["alpha"],
        "beta": score_row["beta"],
        "ari": score_row["ari"],
        "score": score_row["score"],
        "fit_time": score_row["fit_time"],
        "num_parts": score_row["num_parts"],
        "num_edges": score_row["num_edges"],
        "precision": score_row["precision"],
        "recall": score_row["recall"],
        "f1": score_row["f1"],
        "parts": score_parts,
        "edges": score_edges,
    }

    oracle_params = {
        "alpha": oracle_row["alpha"],
        "beta": oracle_row["beta"],
        "ari": oracle_row["ari"],
        "score": oracle_row["score"],
        "fit_time": oracle_row["fit_time"],
        "num_parts": oracle_row["num_parts"],
        "num_edges": oracle_row["num_edges"],
        "precision": oracle_row["precision"],
        "recall": oracle_row["recall"],
        "f1": oracle_row["f1"],
        "parts": oracle_parts,
        "edges": oracle_edges,
    }

    with open(snakemake.output.score_params, "w") as f:
        json.dump(score_params, f, indent=2, default=float)
    with open(snakemake.output.oracle_params, "w") as f:
        json.dump(oracle_params, f, indent=2, default=float)


if __name__ == "__main__":
    main()
