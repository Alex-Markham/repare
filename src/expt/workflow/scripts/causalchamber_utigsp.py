#!/usr/bin/env python3
"""Run UT-IGSP grid."""

import pickle

import networkx as nx
import numpy as np
import pandas as pd
from causal_chambers import ut_igsp


def gaussian_bic_score(data, graph):
    """BIC for Gaussian graphical model."""
    if data.size == 0 or graph.number_of_nodes() == 0:
        return float("inf")
    n_samples = data.shape[0]
    total_ll = 0.0
    total_params = 0
    eps = 1e-12
    for node in sorted(graph.nodes):
        parents = list(graph.predecessors(node))
        y = data[:, node]
        if parents:
            X = data[:, parents]
            X_aug = np.column_stack([np.ones(n_samples), X])
            beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
            resid = y - X_aug @ beta
            params = len(parents) + 1
        else:
            resid = y - y.mean()
            params = 1
        sigma2 = max(float(np.mean(resid**2)), eps)
        total_ll += -0.5 * n_samples * (np.log(2 * np.pi * sigma2) + 1)
        total_params += params
    return float(-2 * total_ll + total_params * np.log(n_samples))


def graph_edge_metrics(est_graph, true_graph):
    """Precision/recall/F1 for edges."""
    tp = sum(1 for edge in est_graph.edges if true_graph.has_edge(*edge))
    precision = tp / len(est_graph.edges) if est_graph.edges else 1.0
    recall = tp / len(true_graph.edges) if true_graph.edges else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def main():
    # Load data
    with open(snakemake.input.blocks, "rb") as f:
        blocks = pickle.load(f)
    with open(snakemake.input.singleenvdata, "rb") as f:
        single_env_data = pickle.load(f)
    with open(snakemake.input.truegraph, "rb") as f:
        true_graph = pickle.load(f)

    # UT-IGSP alphas
    ut_alphas = [0.0001, 0.001, 0.01, 0.1]

    records = []
    data_list = [blocks["obs"], *single_env_data]

    for alpha_ci in ut_alphas:
        for alpha_inv in ut_alphas:
            import time

            start = time.perf_counter()
            dag_matrix, _ = ut_igsp.fit(
                data_list,
                alpha_ci=alpha_ci,
                alpha_inv=alpha_inv,
                test="gauss",
                obs_idx=0,
            )
            runtime = time.perf_counter() - start

            # Build graph
            graph = nx.DiGraph()
            graph.add_nodes_from(range(dag_matrix.shape[0]))
            for i in range(dag_matrix.shape[0]):
                for j in range(dag_matrix.shape[1]):
                    if dag_matrix[i, j]:
                        graph.add_edge(i, j)

            bic = gaussian_bic_score(blocks["obs"], graph)
            edge_stats = graph_edge_metrics(graph, true_graph)

            records.append(
                {
                    "alpha_ci": alpha_ci,
                    "alpha_inv": alpha_inv,
                    "bic": bic,
                    "runtime_sec": runtime,
                    **edge_stats,
                }
            )

    df = pd.DataFrame(records)
    df.to_csv(snakemake.output.ut_metrics, index=False)


if __name__ == "__main__":
    main()
