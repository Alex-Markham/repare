#!/usr/bin/env python3
"""Run GnIES."""

import json
import pickle
import time

import gnies
import networkx as nx
import numpy as np


def skeleton_metrics(est_graph, true_graph):
    """Precision/recall/F1 for skeleton (undirected)."""
    est_edges = {frozenset(edge) for edge in est_graph.edges}
    true_edges = {frozenset(edge) for edge in true_graph.edges}
    tp = len(est_edges & true_edges)
    precision = tp / len(est_edges) if est_edges else 1.0
    recall = tp / len(true_edges) if true_edges else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def subsample_env(arr, max_rows=2000, seed=0):
    """Subsample environment if too large."""
    if arr.shape[0] <= max_rows:
        return arr
    rng = np.random.default_rng(seed)
    take = rng.choice(arr.shape[0], size=max_rows, replace=False)
    return arr[take]


def main():
    # Load data
    with open(snakemake.input.blocks, "rb") as f:
        blocks = pickle.load(f)
    with open(snakemake.input.singleenvdata, "rb") as f:
        single_env_data = pickle.load(f)
    with open(snakemake.input.singleenvtargets, "rb") as f:
        single_env_targets = pickle.load(f)
    with open(snakemake.input.singleenvlabels, "rb") as f:
        single_env_labels = pickle.load(f)
    with open(snakemake.input.truegraph, "rb") as f:
        true_graph = pickle.load(f)

    # Prepare data
    data = [subsample_env(blocks["obs"], seed=0)]
    for idx, env in enumerate(single_env_data, start=1):
        data.append(subsample_env(env, seed=idx))

    # Run GnIES
    start = time.perf_counter()
    score, adj, targets = gnies.fit(
        data=data,
        known_targets=set(),
        approach="greedy",
        center=True,
        ges_iterate=False,
        phases=["forward"],
        ges_phases=["forward"],
    )
    runtime = time.perf_counter() - start

    # Build graph
    graph = nx.DiGraph()
    graph.add_nodes_from(range(adj.shape[0]))
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i, j]:
                graph.add_edge(i, j)

    stats = skeleton_metrics(graph, true_graph)

    # Save metrics
    metrics = {
        "method": "GnIES",
        "score": float(score),
        "runtime_sec": runtime,
        "estimated_targets": sorted(targets),
        **stats,
    }

    with open(snakemake.output.metrics, "w") as f:
        json.dump(metrics, f, indent=2, default=float)


if __name__ == "__main__":
    main()
