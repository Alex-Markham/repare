import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

import gies
import gnies
from causal_chambers import ut_igsp
from causalchamber.datasets import Dataset
from causalchamber.ground_truth import main as gt
from repare.repare import PartitionDagModelIvn, _get_totally_ordered_partition

OBS_EXPERIMENT = "uniform_reference"
RGB_EXPERIMENTS = ["uniform_red_strong", "uniform_green_strong", "uniform_blue_strong"]
POL_EXPERIMENTS = ["uniform_pol_1_strong", "uniform_pol_2_strong"]
SINGLE_TARGET_EXPERIMENTS = {
    "red": "uniform_red_strong",
    "green": "uniform_green_strong",
    "blue": "uniform_blue_strong",
    "pol_1": "uniform_pol_1_strong",
    "pol_2": "uniform_pol_2_strong",
}
REFERENCE_VARIABLES = [
    "red",
    "green",
    "blue",
    "current",
    "ir_1",
    "ir_2",
    "ir_3",
    "vis_1",
    "vis_2",
    "vis_3",
    "pol_1",
    "pol_2",
    "angle_1",
    "angle_2",
    "l_11",
    "l_12",
    "l_21",
    "l_22",
    "l_31",
    "l_32",
]


def _load_matrix(dataset, exp_name, feature_cols):
    df = dataset.get_experiment(name=exp_name).as_pandas_dataframe()
    return df[feature_cols].to_numpy(dtype=float)


def _stack(dataset, experiments, feature_cols):
    return np.vstack([_load_matrix(dataset, exp, feature_cols) for exp in experiments])


def drop_constant_features(blocks, feature_cols, aggregate_sets):
    mask = blocks["obs"].std(axis=0) > 0
    if mask.all():
        return blocks, feature_cols, aggregate_sets
    feature_cols = [name for name, keep in zip(feature_cols, mask) if keep]
    aggregate_sets = [atoms for atoms, keep in zip(aggregate_sets, mask) if keep]
    blocks = {key: value[:, mask] for key, value in blocks.items()}
    return blocks, feature_cols, aggregate_sets


def build_true_graph(aggregate_sets, true_dag_full):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(aggregate_sets)))
    for i, src_atoms in enumerate(aggregate_sets):
        for j, dst_atoms in enumerate(aggregate_sets):
            if i == j:
                continue
            if any(true_dag_full.has_edge(u, v) for u in src_atoms for v in dst_atoms):
                graph.add_edge(i, j)
    return graph


def descendant_mask(atom_indices, aggregate_sets, true_dag_full):
    closure = set(atom_indices)
    for idx in list(atom_indices):
        closure.update(nx.descendants(true_dag_full, idx))
    mask = np.zeros(len(aggregate_sets), dtype=bool)
    for agg_idx, atoms in enumerate(aggregate_sets):
        if atoms & closure:
            mask[agg_idx] = True
    return mask


def ground_truth_partition(target_dict, aggregate_sets, true_dag_full):
    ordered_masks = {}
    for order, label in enumerate(sorted(target_dict)):
        atom_union = set().union(*[aggregate_sets[idx] for idx in target_dict[label]])
        ordered_masks[str(order)] = descendant_mask(atom_union, aggregate_sets, true_dag_full)
    partition = _get_totally_ordered_partition(ordered_masks)
    labels = np.zeros(len(aggregate_sets), dtype=int)
    for label, block in enumerate(partition):
        labels[list(block)] = label
    return partition, labels


def prepare_dataset(dataset_name, root, chamber, configuration):
    dataset = Dataset(dataset_name, root=root, download=True)
    full_feature_cols = gt.variables(chamber, configuration)
    full_idx = {name: i for i, name in enumerate(full_feature_cols)}
    feature_cols = [col for col in REFERENCE_VARIABLES if col in full_idx]
    if not feature_cols:
        raise ValueError("No overlapping variables between reference list and dataset columns.")

    base_blocks = {
        "obs": _load_matrix(dataset, OBS_EXPERIMENT, feature_cols),
        "rgb": _stack(dataset, RGB_EXPERIMENTS, feature_cols),
        "pol": _stack(dataset, POL_EXPERIMENTS, feature_cols),
    }
    for label, exp_name in SINGLE_TARGET_EXPERIMENTS.items():
        base_blocks[label] = _load_matrix(dataset, exp_name, feature_cols)

    aggregate_sets = [{full_idx[col]} for col in feature_cols]
    blocks, feature_cols, aggregate_sets = drop_constant_features(
        base_blocks, feature_cols, aggregate_sets
    )

    name_to_idx = {name: i for i, name in enumerate(feature_cols)}
    group_spec = {
        "rgb": {"red", "green", "blue"},
        "pol": {"pol_1", "pol_2"},
    }
    group_targets = {}
    for label, names in group_spec.items():
        idxs = {name_to_idx[name] for name in names if name in name_to_idx}
        if not idxs:
            continue
        group_targets[label] = idxs

    true_dag_full = nx.DiGraph()
    true_dag_full.add_edges_from((full_idx[u], full_idx[v]) for u, v in gt.edges(chamber, configuration))
    true_graph = build_true_graph(aggregate_sets, true_dag_full)
    _, true_labels = ground_truth_partition(group_targets, aggregate_sets, true_dag_full)

    single_env_labels = [label for label in SINGLE_TARGET_EXPERIMENTS if label in name_to_idx]
    single_env_targets = {
        label: {name_to_idx[label]} for label in single_env_labels if label in name_to_idx
    }
    single_env_data = [blocks[label] for label in single_env_labels]

    return dict(
        blocks=blocks,
        feature_cols=feature_cols,
        aggregate_sets=aggregate_sets,
        group_targets=group_targets,
        true_graph=true_graph,
        true_labels=true_labels,
        true_dag_full=true_dag_full,
        name_to_idx=name_to_idx,
        single_env_labels=single_env_labels,
        single_env_targets=single_env_targets,
        single_env_data=single_env_data,
    )


def partition_edge_metrics(model_dag, true_graph):
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
    return dict(precision=precision, recall=recall, f1=f1)


def graph_edge_metrics(est_graph, true_graph):
    tp = sum(1 for edge in est_graph.edges if true_graph.has_edge(*edge))
    precision = tp / len(est_graph.edges) if est_graph.edges else 1.0
    recall = tp / len(true_graph.edges) if true_graph.edges else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(precision=precision, recall=recall, f1=f1)


def skeleton_metrics(est_graph, true_graph):
    est_edges = {frozenset(edge) for edge in est_graph.edges}
    true_edges = {frozenset(edge) for edge in true_graph.edges}
    tp = len(est_edges & true_edges)
    precision = tp / len(est_edges) if est_edges else 1.0
    recall = tp / len(true_edges) if true_edges else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(precision=precision, recall=recall, f1=f1)


def build_data_dict(blocks, group_targets):
    data = {"obs": (blocks["obs"], set(), "obs")}
    for label, targets in group_targets.items():
        data[label] = (blocks[label], targets, "soft")
    return data


def run_repare_grid(blocks, aggregate_sets, group_targets, true_graph, true_labels, alphas, betas):
    records = []
    best = None
    best_model = None
    best_row = None
    for alpha in alphas:
        for beta in betas:
            model = PartitionDagModelIvn().fit(
                build_data_dict(blocks, group_targets),
                alpha=float(alpha),
                beta=float(beta),
                assume="gaussian",
                refine_test="ks",
            )
            est_labels = np.zeros(len(aggregate_sets), dtype=int)
            for label, block in enumerate(model.dag.nodes):
                est_labels[list(block)] = label
            ari = adjusted_rand_score(true_labels, est_labels)
            edge_stats = partition_edge_metrics(model.dag, true_graph)
            row = dict(alpha=float(alpha), beta=float(beta), ari=ari, **edge_stats)
            records.append(row)
            score = (edge_stats["f1"], edge_stats["precision"])
            if best is None or score > best:
                best = score
                best_model = model
                best_row = row
    return pd.DataFrame(records), best_model, best_row


def run_ut_grid(data_list, true_graph, alphas_ci, alphas_inv, test="gauss"):
    records = []
    best_graph = None
    best_params = None
    best_score = -np.inf
    for alpha_ci in alphas_ci:
        for alpha_inv in alphas_inv:
            dag_matrix, _ = ut_igsp.fit(
                data_list,
                alpha_ci=alpha_ci,
                alpha_inv=alpha_inv,
                test=test,
                obs_idx=0,
            )
            graph = nx.DiGraph()
            graph.add_nodes_from(range(dag_matrix.shape[0]))
            for i in range(dag_matrix.shape[0]):
                for j in range(dag_matrix.shape[1]):
                    if dag_matrix[i, j]:
                        graph.add_edge(i, j)
            edge_stats = graph_edge_metrics(graph, true_graph)
            row = dict(alpha_ci=alpha_ci, alpha_inv=alpha_inv, **edge_stats)
            records.append(row)
            if edge_stats["f1"] > best_score:
                best_score = edge_stats["f1"]
                best_graph = graph
                best_params = dict(alpha_ci=alpha_ci, alpha_inv=alpha_inv, **edge_stats)
    return pd.DataFrame(records), best_graph, best_params


def labeled_summary(model, feature_cols):
    labeled = []
    for idx, part in enumerate(model.dag.nodes):
        labels = tuple(feature_cols[i] for i in part)
        labeled.append((idx, labels))
    edges = []
    node_to_idx = {node: idx for idx, node in enumerate(model.dag.nodes)}
    for u, v in model.dag.edges:
        edges.append((node_to_idx[u], node_to_idx[v]))
    return labeled, edges


def save_dag_plot(model, feature_cols, path):
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


def run_gies(obs, env_data, env_targets, true_graph):
    data = [obs, *env_data]
    interventions = [[], *[sorted(t) for t in env_targets]]
    adj, score = gies.fit_bic(data, interventions)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(adj.shape[0]))
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i, j]:
                graph.add_edge(i, j)
    stats = graph_edge_metrics(graph, true_graph)
    return score, graph, stats


def _subsample_env(arr, max_rows=4000, seed=0):
    if arr.shape[0] <= max_rows:
        return arr
    rng = np.random.default_rng(seed)
    take = rng.choice(arr.shape[0], size=max_rows, replace=False)
    return arr[take]


def run_gnies(obs, env_data, env_targets, true_graph):
    data = [_subsample_env(obs, seed=0)]
    for idx, env in enumerate(env_data, start=1):
        data.append(_subsample_env(env, seed=idx))
    known_targets = set().union(*env_targets)
    score, adj, targets = gnies.fit(
        data=data,
        known_targets=known_targets,
        approach="greedy",
        center=False,
        ges_iterate=False,
        ges_phases=["forward", "backward"],
    )
    graph = nx.DiGraph()
    graph.add_nodes_from(range(adj.shape[0]))
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i, j]:
                graph.add_edge(i, j)
    stats = skeleton_metrics(graph, true_graph)
    return score, targets, graph, stats


def main():
    dataset_name = snakemake.params.dataset
    root = Path(snakemake.params.root)
    chamber = snakemake.params.chamber
    configuration = snakemake.params.configuration
    alphas = snakemake.params.alphas
    betas = snakemake.params.betas
    ut_alphas = snakemake.params.ut_alphas

    data = prepare_dataset(dataset_name, root, chamber, configuration)
    blocks = data["blocks"]
    feature_cols = data["feature_cols"]
    aggregate_sets = data["aggregate_sets"]
    group_targets = data["group_targets"]
    true_graph = data["true_graph"]
    true_labels = data["true_labels"]
    true_dag_full = data["true_dag_full"]
    name_to_idx = data["name_to_idx"]
    single_env_labels = data["single_env_labels"]
    single_env_targets = data["single_env_targets"]
    single_env_data = data["single_env_data"]

    metrics_df, best_model, best_params = run_repare_grid(
        blocks,
        aggregate_sets,
        group_targets,
        true_graph,
        true_labels,
        alphas,
        betas,
    )
    metrics_df.to_csv(snakemake.output.metrics, index=False)

    save_dag_plot(best_model, feature_cols, snakemake.output.dag)
    parts, edges = labeled_summary(best_model, feature_cols)

    method_records = []
    method_records.append(
        dict(
            method="RePaRe_grouped",
            metric_type="partition",
            ari=best_params["ari"],
            precision=best_params["precision"],
            recall=best_params["recall"],
            f1=best_params["f1"],
        )
    )

    # Ungrouped RePaRe
    ungrouped_data = {"obs": (blocks["obs"], set(), "obs")}
    for label in single_env_labels:
        idx = single_env_targets[label]
        ungrouped_data[label] = (blocks[label], idx, "soft")
    _, ungrouped_true_labels = ground_truth_partition(
        {label: single_env_targets[label] for label in single_env_labels},
        aggregate_sets,
        true_dag_full,
    )
    alpha = best_params["alpha"]
    beta = best_params["beta"]
    ungrouped_model = PartitionDagModelIvn().fit(
        ungrouped_data,
        alpha=alpha,
        beta=beta,
        assume="gaussian",
        refine_test="ks",
    )
    est_labels = np.zeros(len(aggregate_sets), dtype=int)
    for label, block in enumerate(ungrouped_model.dag.nodes):
        est_labels[list(block)] = label
    ungrouped_ari = adjusted_rand_score(ungrouped_true_labels, est_labels)
    ungrouped_stats = partition_edge_metrics(ungrouped_model.dag, true_graph)
    method_records.append(
        dict(
            method="RePaRe_ungrouped",
            metric_type="partition",
            ari=ungrouped_ari,
            precision=ungrouped_stats["precision"],
            recall=ungrouped_stats["recall"],
            f1=ungrouped_stats["f1"],
        )
    )
    ungrouped_parts, ungrouped_edges = labeled_summary(ungrouped_model, feature_cols)

    # GIES
    gies_score, gies_graph, gies_stats = run_gies(
        blocks["obs"],
        single_env_data,
        [single_env_targets[label] for label in single_env_labels],
        true_graph,
    )
    method_records.append(
        dict(
            method="GIES",
            metric_type="edges",
            ari=np.nan,
            precision=gies_stats["precision"],
            recall=gies_stats["recall"],
            f1=gies_stats["f1"],
        )
    )

    # GnIES
    gnies_score, gnies_targets, gnies_graph, gnies_stats = run_gnies(
        blocks["obs"],
        single_env_data,
        [single_env_targets[label] for label in single_env_labels],
        true_graph,
    )
    method_records.append(
        dict(
            method="GnIES",
            metric_type="skeleton",
            ari=np.nan,
            precision=gnies_stats["precision"],
            recall=gnies_stats["recall"],
            f1=gnies_stats["f1"],
        )
    )

    # UT-IGSP
    ut_df, ut_graph, ut_best = run_ut_grid(
        [blocks["obs"], *single_env_data],
        true_graph,
        alphas_ci=ut_alphas,
        alphas_inv=ut_alphas,
        test="gauss",
    )
    method_records.append(
        dict(
            method="UT-IGSP",
            metric_type="edges",
            ari=np.nan,
            precision=ut_best["precision"],
            recall=ut_best["recall"],
            f1=ut_best["f1"],
        )
    )

    pd.DataFrame(method_records).to_csv(snakemake.output.method_metrics, index=False)

    summary_lines = []
    summary_lines.append("Best grouped RePaRe hyperparameters:")
    summary_lines.append(json.dumps(best_params, indent=2))
    summary_lines.append("")
    summary_lines.append("Partition nodes:")
    for idx, labels in parts:
        summary_lines.append(f"  Node {idx}: {labels}")
    summary_lines.append("Edges (u -> v):")
    for u, v in edges:
        summary_lines.append(f"  {u} -> {v}")

    summary_lines.append("")
    summary_lines.append(f"Ungrouped RePaRe (alpha={alpha}, beta={beta}):")
    summary_lines.append(f"  ARI: {ungrouped_ari}")
    summary_lines.append(f"  Edge precision/recall/F1: {json.dumps(ungrouped_stats)}")
    summary_lines.append("  Nodes:")
    for idx, labels in ungrouped_parts:
        summary_lines.append(f"    Node {idx}: {labels}")
    summary_lines.append("  Edges (u -> v):")
    for u, v in ungrouped_edges:
        summary_lines.append(f"    {u} -> {v}")

    summary_lines.append("")
    summary_lines.append("GIES:")
    summary_lines.append(f"  Score: {gies_score}")
    summary_lines.append(f"  Precision/Recall/F1: {json.dumps(gies_stats)}")
    summary_lines.append(f"  Edges: {list(gies_graph.edges())}")

    summary_lines.append("")
    summary_lines.append("GnIES:")
    summary_lines.append(f"  Score: {gnies_score}")
    summary_lines.append(f"  Estimated targets: {sorted(gnies_targets)}")
    summary_lines.append(f"  Skeleton precision/recall/F1: {json.dumps(gnies_stats)}")

    summary_lines.append("")
    summary_lines.append("UT-IGSP:")
    summary_lines.append(f"  Best params: {json.dumps(ut_best)}")
    summary_lines.append(f"  Edges: {list(ut_graph.edges())}")
    summary_lines.append("")

    with open(snakemake.output.summary, "w") as f:
        f.write("\n".join(summary_lines))

    ut_df.to_csv(snakemake.output.ut_metrics, index=False)


if __name__ == "__main__":
    main()
