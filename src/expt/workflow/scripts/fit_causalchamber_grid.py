import json
import pickle
import shutil
import time
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
from gnies.scores.gnies_score import GnIESScore
from causal_chambers import ut_igsp
from causalchamber.datasets import Dataset
from causalchamber.ground_truth import main as gt
from repare.repare import PartitionDagModelIvn, _get_totally_ordered_partition

OBS_EXPERIMENT = "uniform_reference"
RGB_EXPERIMENTS = ["uniform_red_strong", "uniform_green_strong", "uniform_blue_strong"]
POL_EXPERIMENTS = ["uniform_pol_1_strong", "uniform_pol_2_strong"]
DEFAULT_SINGLE_TARGET_EXPERIMENTS = {
    "red": "uniform_red_strong",
    "green": "uniform_green_strong",
    "blue": "uniform_blue_strong",
    "pol_1": "uniform_pol_1_strong",
    "pol_2": "uniform_pol_2_strong",
}
ALL_SINGLE_TARGET_EXPERIMENTS = {
    "red": "uniform_red_strong",
    "green": "uniform_green_strong",
    "blue": "uniform_blue_strong",
    "current": "uniform_v_c_strong",
    "ir_1": "uniform_t_ir_1_strong",
    "ir_2": "uniform_t_ir_2_strong",
    "ir_3": "uniform_t_ir_3_strong",
    "vis_1": "uniform_t_vis_1_strong",
    "vis_2": "uniform_t_vis_2_strong",
    "vis_3": "uniform_t_vis_3_strong",
    "pol_1": "uniform_pol_1_strong",
    "pol_2": "uniform_pol_2_strong",
    "angle_1": "uniform_v_angle_1_strong",
    "angle_2": "uniform_v_angle_2_strong",
    "l_11": "uniform_l_11_mid",
    "l_12": "uniform_l_12_mid",
    "l_21": "uniform_l_21_mid",
    "l_22": "uniform_l_22_mid",
    "l_31": "uniform_l_31_mid",
    "l_32": "uniform_l_32_mid",
}
TARGET_MODE_MAP = {
    "default": DEFAULT_SINGLE_TARGET_EXPERIMENTS,
    "all": ALL_SINGLE_TARGET_EXPERIMENTS,
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


def drop_constant_features(blocks, feature_cols, partition_parts):
    mask = blocks["obs"].std(axis=0) > 0
    if mask.all():
        return blocks, feature_cols, partition_parts
    feature_cols = [name for name, keep in zip(feature_cols, mask) if keep]
    partition_parts = [atoms for atoms, keep in zip(partition_parts, mask) if keep]
    blocks = {key: value[:, mask] for key, value in blocks.items()}
    return blocks, feature_cols, partition_parts


def format_hparam(value):
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


def ensure_clean_dir(path):
    if path is None:
        return None
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def gaussian_bic_score(data, graph):
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


def persist_model_artifacts(model, feature_cols, root, metrics, alpha, beta):
    if root is None:
        return
    sel = metrics.get("selection")
    suffix = f"_{sel}" if sel else ""
    run_dir = root / f"alpha={format_hparam(alpha)}_beta={format_hparam(beta)}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_dag_plot(model, feature_cols, run_dir / "dag.png")
    with open(run_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)


def build_true_graph(parts, true_dag_full):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(parts)))
    for i, src_atoms in enumerate(parts):
        for j, dst_atoms in enumerate(parts):
            if i == j:
                continue
            if any(true_dag_full.has_edge(u, v) for u in src_atoms for v in dst_atoms):
                graph.add_edge(i, j)
    return graph


def descendant_mask(atom_indices, parts, true_dag_full):
    closure = set(atom_indices)
    for idx in list(atom_indices):
        closure.update(nx.descendants(true_dag_full, idx))
    mask = np.zeros(len(parts), dtype=bool)
    for part_idx, atoms in enumerate(parts):
        if atoms & closure:
            mask[part_idx] = True
    return mask


def ground_truth_partition(target_dict, parts, true_dag_full):
    ordered_masks = {}
    for order, label in enumerate(sorted(target_dict)):
        atom_union = set().union(*[parts[idx] for idx in target_dict[label]])
        ordered_masks[str(order)] = descendant_mask(atom_union, parts, true_dag_full)
    partition = _get_totally_ordered_partition(ordered_masks)
    labels = np.zeros(len(parts), dtype=int)
    for label, block in enumerate(partition):
        labels[list(block)] = label
    return partition, labels


def prepare_dataset(dataset_name, root, chamber, configuration, single_target_experiments):
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    dataset = Dataset(dataset_name, root=root_path, download=True)
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
    for label, exp_name in single_target_experiments.items():
        base_blocks[label] = _load_matrix(dataset, exp_name, feature_cols)

    partition_parts = [{full_idx[col]} for col in feature_cols]
    blocks, feature_cols, partition_parts = drop_constant_features(
        base_blocks, feature_cols, partition_parts
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
    true_graph = build_true_graph(partition_parts, true_dag_full)
    _, true_labels = ground_truth_partition(group_targets, partition_parts, true_dag_full)

    single_env_labels = [label for label in single_target_experiments if label in name_to_idx]
    single_env_targets = {label: {name_to_idx[label]} for label in single_env_labels}
    single_env_data = [blocks[label] for label in single_env_labels]

    return dict(
        blocks=blocks,
        feature_cols=feature_cols,
        partition_parts=partition_parts,
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


def run_repare_grid(
    blocks,
    partition_parts,
    group_targets,
    true_graph,
    true_labels,
    alphas,
    betas,
    grid_root=None,
    feature_cols=None,
):
    records = []
    models = {}
    env_labels = list(group_targets)
    gnies_data = [blocks["obs"]]
    for label in env_labels:
        gnies_data.append(blocks[label])
    intervention_union = set().union(*group_targets.values()) if group_targets else set()
    gnies_score_class = GnIESScore(gnies_data, intervention_union, lmbda=0.0, centered=True)
    for alpha in alphas:
        for beta in betas:
            start = time.perf_counter()
            model = PartitionDagModelIvn().fit(
                build_data_dict(blocks, group_targets),
                alpha=float(alpha),
                beta=float(beta),
                assume="gaussian",
                refine_test="ks",
            )
            fit_time = time.perf_counter() - start
            model.fit_runtime_sec = fit_time
            est_labels = np.zeros(len(partition_parts), dtype=int)
            for label, block in enumerate(model.dag.nodes):
                est_labels[list(block)] = label
            ari = adjusted_rand_score(true_labels, est_labels)
            edge_stats = partition_edge_metrics(model.dag, true_graph)
            # Score expanded DAG using GnIES full score.
            expanded_adj = model.expand_coarsened_dag()
            gnies_value = float(gnies_score_class.full_score(expanded_adj))
            model.score = gnies_value
            row = dict(
                alpha=float(alpha),
                beta=float(beta),
                ari=ari,
                score=gnies_value,
                fit_time=fit_time,
                num_parts=model.dag.number_of_nodes(),
                num_edges=model.dag.number_of_edges(),
                **edge_stats,
            )
            records.append(row)
            models[(row["alpha"], row["beta"])] = model
            persist_model_artifacts(model, feature_cols, grid_root, row, row["alpha"], row["beta"])
    df = pd.DataFrame(records)
    if not records:
        raise RuntimeError("No RePaRe models were fitted.")
    score_row = min(records, key=lambda r: (r["score"]))
    oracle_row = max(records, key=lambda r: (r["ari"], r["f1"], r["precision"]))
    score_model = models[(score_row["alpha"], score_row["beta"])]
    oracle_model = models[(oracle_row["alpha"], oracle_row["beta"])]
    return df, score_model, oracle_model, score_row, oracle_row


def run_ut_grid(data_list, obs_data, true_graph, alphas_ci, alphas_inv, test="gauss"):
    records = []
    best_bic = None
    best_bic_graph = None
    best_bic_row = None
    best_oracle = None
    best_oracle_graph = None
    best_oracle_row = None
    for alpha_ci in alphas_ci:
        for alpha_inv in alphas_inv:
            start = time.perf_counter()
            dag_matrix, _ = ut_igsp.fit(
                data_list,
                alpha_ci=alpha_ci,
                alpha_inv=alpha_inv,
                test=test,
                obs_idx=0,
            )
            runtime = time.perf_counter() - start
            graph = nx.DiGraph()
            graph.add_nodes_from(range(dag_matrix.shape[0]))
            for i in range(dag_matrix.shape[0]):
                for j in range(dag_matrix.shape[1]):
                    if dag_matrix[i, j]:
                        graph.add_edge(i, j)
            bic = gaussian_bic_score(obs_data, graph)
            edge_stats = graph_edge_metrics(graph, true_graph)
            row = dict(
                alpha_ci=float(alpha_ci),
                alpha_inv=float(alpha_inv),
                bic=bic,
                runtime_sec=runtime,
                **edge_stats,
            )
            records.append(row)
            if best_bic is None or bic < best_bic:
                best_bic = bic
                best_bic_graph = graph
                best_bic_row = row
            oracle_key = (edge_stats["f1"], edge_stats["precision"], -bic)
            if best_oracle is None or oracle_key > best_oracle:
                best_oracle = oracle_key
                best_oracle_graph = graph
                best_oracle_row = row
    return pd.DataFrame(records), best_bic_graph, best_oracle_graph, best_bic_row, best_oracle_row


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
    start = time.perf_counter()
    adj, score = gies.fit_bic(data, interventions)
    runtime = time.perf_counter() - start
    graph = nx.DiGraph()
    graph.add_nodes_from(range(adj.shape[0]))
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i, j]:
                graph.add_edge(i, j)
    stats = graph_edge_metrics(graph, true_graph)
    return score, graph, stats, runtime


def _subsample_env(arr, max_rows=2000, seed=0):
    if arr.shape[0] <= max_rows:
        return arr
    rng = np.random.default_rng(seed)
    take = rng.choice(arr.shape[0], size=max_rows, replace=False)
    return arr[take]


def run_gnies(obs, env_data, env_targets, true_graph):
    data = [_subsample_env(obs, seed=0)]
    for idx, env in enumerate(env_data, start=1):
        data.append(_subsample_env(env, seed=idx))
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
    graph = nx.DiGraph()
    graph.add_nodes_from(range(adj.shape[0]))
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i, j]:
                graph.add_edge(i, j)
    stats = skeleton_metrics(graph, true_graph)
    return score, targets, graph, stats, runtime


def main():
    dataset_name = snakemake.params.dataset
    root = Path(snakemake.params.root)
    chamber = snakemake.params.chamber
    configuration = snakemake.params.configuration
    alphas = snakemake.params.alphas
    betas = snakemake.params.betas
    ut_alphas = snakemake.params.ut_alphas
    target_mode = getattr(snakemake.params, "target_mode", "default")
    include_grouped = getattr(snakemake.params, "include_grouped", True)
    single_target_map = TARGET_MODE_MAP.get(target_mode, DEFAULT_SINGLE_TARGET_EXPERIMENTS)
    grid_dir = getattr(snakemake.output, "grid_dir", None)
    grid_root = ensure_clean_dir(Path(grid_dir)) if grid_dir else None

    data = prepare_dataset(dataset_name, root, chamber, configuration, single_target_map)
    blocks = data["blocks"]
    feature_cols = data["feature_cols"]
    partition_parts = data["partition_parts"]
    group_targets = data["group_targets"]
    true_graph = data["true_graph"]
    true_labels = data["true_labels"]
    true_dag_full = data["true_dag_full"]
    name_to_idx = data["name_to_idx"]
    single_env_labels = data["single_env_labels"]
    single_env_targets = data["single_env_targets"]
    single_env_data = data["single_env_data"]

    grouped_df = None
    method_records = []
    if include_grouped:
        grouped_df, score_model, oracle_model, score_params, oracle_params = run_repare_grid(
            blocks,
            partition_parts,
            group_targets,
            true_graph,
            true_labels,
            alphas,
            betas,
            grid_root=grid_root,
            feature_cols=feature_cols,
        )
        save_dag_plot(score_model, feature_cols, snakemake.output.dag)
        score_parts, score_edges = labeled_summary(score_model, feature_cols)
        oracle_parts, oracle_edges = labeled_summary(oracle_model, feature_cols)
        for label, params in (("score", score_params), ("oracle", oracle_params)):
            method_records.append(
                dict(
                    method="RePaRe_grouped",
                    selection=label,
                    metric_type="partition",
                    ari=params["ari"],
                    precision=params["precision"],
                    recall=params["recall"],
                    f1=params["f1"],
                    runtime_sec=params["fit_time"],
                    score=params["score"],
                    alpha=params["alpha"],
                    beta=params["beta"],
                )
            )

    # Ungrouped RePaRe
    ungrouped_data = {"obs": (blocks["obs"], set(), "obs")}
    for label in single_env_labels:
        idx = single_env_targets[label]
        ungrouped_data[label] = (blocks[label], idx, "soft")
    _, ungrouped_true_labels = ground_truth_partition(
        {label: single_env_targets[label] for label in single_env_labels},
        partition_parts,
        true_dag_full,
    )
    (
        ungrouped_df,
        ungrouped_score_model,
        ungrouped_oracle_model,
        ungrouped_score_params,
        ungrouped_oracle_params,
    ) = run_repare_grid(
        blocks,
        partition_parts,
        {label: single_env_targets[label] for label in single_env_labels},
        true_graph,
        ungrouped_true_labels,
        alphas,
        betas,
        grid_root=None,
        feature_cols=feature_cols,
    )
    untranslated = [
        ("score", ungrouped_score_model, ungrouped_score_params),
        ("oracle", ungrouped_oracle_model, ungrouped_oracle_params),
    ]
    ungrouped_plots = {}
    for sel, model, params in untranslated:
        stats = partition_edge_metrics(model.dag, true_graph)
        method_records.append(
            dict(
                method="RePaRe_ungrouped",
                selection=sel,
                metric_type="partition",
                ari=params["ari"],
                precision=stats["precision"],
                recall=stats["recall"],
                f1=stats["f1"],
                runtime_sec=params["fit_time"],
                score=params["score"],
                alpha=params["alpha"],
                beta=params["beta"],
            )
        )
        ungrouped_plots[sel] = labeled_summary(model, feature_cols)

    if not include_grouped:
        save_dag_plot(ungrouped_score_model, feature_cols, snakemake.output.dag)

    # GIES
    gies_score, gies_graph, gies_stats, gies_runtime = run_gies(
        blocks["obs"],
        single_env_data,
        [single_env_targets[label] for label in single_env_labels],
        true_graph,
    )
    method_records.append(
        dict(
            method="GIES",
            selection="score",
            metric_type="edges",
            ari=np.nan,
            precision=gies_stats["precision"],
            recall=gies_stats["recall"],
            f1=gies_stats["f1"],
            runtime_sec=gies_runtime,
            score=gies_score,
        )
    )

    # GnIES
    gnies_score, gnies_targets, gnies_graph, gnies_stats, gnies_runtime = run_gnies(
        blocks["obs"],
        single_env_data,
        [single_env_targets[label] for label in single_env_labels],
        true_graph,
    )
    method_records.append(
        dict(
            method="GnIES",
            selection="score",
            metric_type="skeleton",
            ari=np.nan,
            precision=gnies_stats["precision"],
            recall=gnies_stats["recall"],
            f1=gnies_stats["f1"],
            runtime_sec=gnies_runtime,
            score=gnies_score,
        )
    )

    # UT-IGSP
    ut_df, ut_score_graph, ut_oracle_graph, ut_score_params, ut_oracle_params = run_ut_grid(
        [blocks["obs"], *single_env_data],
        blocks["obs"],
        true_graph,
        alphas_ci=ut_alphas,
        alphas_inv=ut_alphas,
        test="gauss",
    )
    for label, params in (("score", ut_score_params), ("oracle", ut_oracle_params)):
        method_records.append(
            dict(
                method="UT-IGSP",
                selection=label,
                metric_type="edges",
                ari=np.nan,
                precision=params["precision"],
                recall=params["recall"],
                f1=params["f1"],
                runtime_sec=params["runtime_sec"],
                score=params.get("bic", np.nan),
                alpha_ci=params["alpha_ci"],
                alpha_inv=params["alpha_inv"],
            )
        )

    pd.DataFrame(method_records).to_csv(snakemake.output.method_metrics, index=False)

    summary_lines = []
    summary_lines.append(f"Target mode: {target_mode}")
    summary_lines.append("")
    if include_grouped:
        summary_lines.append("Grouped RePaRe (score-selected) hyperparameters:")
        summary_lines.append(json.dumps(score_params, indent=2, default=float))
        summary_lines.append("Partition nodes (score-selected):")
        for idx, labels in score_parts:
            summary_lines.append(f"  Node {idx}: {labels}")
        summary_lines.append("Edges (u -> v):")
        for u, v in score_edges:
            summary_lines.append(f"  {u} -> {v}")
        summary_lines.append("")
        summary_lines.append("Grouped RePaRe (oracle-selected) hyperparameters:")
        summary_lines.append(json.dumps(oracle_params, indent=2, default=float))
        summary_lines.append("Partition nodes (oracle-selected):")
        for idx, labels in oracle_parts:
            summary_lines.append(f"  Node {idx}: {labels}")
        summary_lines.append("Edges (u -> v):")
        for u, v in oracle_edges:
            summary_lines.append(f"  {u} -> {v}")
    else:
        summary_lines.append("Grouped RePaRe skipped for this configuration.")

    summary_lines.append("")
    for label, (parts_summary, edges_summary) in ungrouped_plots.items():
        summary_lines.append("")
        summary_lines.append(f"Ungrouped RePaRe ({label}-selected):")
        params = ungrouped_score_params if label == "score" else ungrouped_oracle_params
        summary_lines.append(json.dumps(params, indent=2, default=float))
        summary_lines.append("  Nodes:")
        for idx, labels in parts_summary:
            summary_lines.append(f"    Node {idx}: {labels}")
        summary_lines.append("  Edges (u -> v):")
        for u, v in edges_summary:
            summary_lines.append(f"    {u} -> {v}")

    summary_lines.append("")
    summary_lines.append("GIES:")
    summary_lines.append(f"  Score: {gies_score}")
    summary_lines.append(f"  Precision/Recall/F1: {json.dumps(gies_stats)}")
    summary_lines.append(f"  Runtime (s): {gies_runtime}")
    summary_lines.append(f"  Edges: {list(gies_graph.edges())}")

    summary_lines.append("")
    summary_lines.append("GnIES:")
    summary_lines.append(f"  Score: {gnies_score}")
    summary_lines.append(f"  Estimated targets: {sorted(gnies_targets)}")
    summary_lines.append(f"  Skeleton precision/recall/F1: {json.dumps(gnies_stats)}")
    summary_lines.append(f"  Runtime (s): {gnies_runtime}")

    summary_lines.append("")
    summary_lines.append("UT-IGSP (score-selected):")
    summary_lines.append(json.dumps(ut_score_params, indent=2, default=float))
    summary_lines.append(f"  Edges: {list(ut_score_graph.edges())}")
    summary_lines.append("")
    summary_lines.append("UT-IGSP (oracle-selected):")
    summary_lines.append(json.dumps(ut_oracle_params, indent=2, default=float))
    summary_lines.append(f"  Edges: {list(ut_oracle_graph.edges())}")
    summary_lines.append("")

    with open(snakemake.output.summary, "w") as f:
        f.write("\n".join(summary_lines))

    ut_df.to_csv(snakemake.output.ut_metrics, index=False)

    grid_metrics_df = grouped_df if include_grouped and grouped_df is not None else ungrouped_df
    grid_metrics_df.to_csv(snakemake.output.metrics, index=False)


if __name__ == "__main__":
    main()
