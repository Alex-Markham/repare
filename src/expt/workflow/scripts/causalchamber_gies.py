#!/usr/bin/env python3
"""Run GIES."""

import json
import pickle
import time
from pathlib import Path

import gies
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

matplotlib.use("Agg")


def graph_edge_metrics(est_graph, true_graph):
    """Precision/recall/F1 for edges."""
    tp = sum(1 for edge in est_graph.edges if true_graph.has_edge(*edge))
    precision = tp / len(est_graph.edges) if est_graph.edges else 1.0
    recall = tp / len(true_graph.edges) if true_graph.edges else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def save_dag_plot(graph, path):
    """Save graph as PNG."""
    fig, ax = plt.subplots(figsize=(10, 7))
    pos = nx.spring_layout(graph, seed=0)
    nx.draw_networkx(
        graph,
        pos=pos,
        ax=ax,
        node_color="#8fbcd4",
        edgecolors="#1f4b73",
        linewidths=1.0,
        font_size=8,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


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

    # Run GIES
    data = [blocks["obs"]]
    data.extend(single_env_data)
    interventions = [
        [],
        *[
            sorted(t)
            for t in [single_env_targets[label] for label in single_env_labels]
        ],
    ]

    start = time.perf_counter()
    adj, score = gies.fit_bic(data, interventions)
    runtime = time.perf_counter() - start

    # Build graph
    graph = nx.DiGraph()
    graph.add_nodes_from(range(adj.shape[0]))
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i, j]:
                graph.add_edge(i, j)

    stats = graph_edge_metrics(graph, true_graph)

    # Save metrics
    metrics = {
        "method": "GIES",
        "score": float(score),
        "runtime_sec": runtime,
        **stats,
    }

    with open(snakemake.output.metrics, "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    # Save plot
    save_dag_plot(graph, Path(snakemake.output.dag))


if __name__ == "__main__":
    main()
