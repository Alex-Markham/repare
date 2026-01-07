#!/usr/bin/env python3
"""Prepare dataset from CausalChamber."""

import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
from causalchamber.datasets import Dataset
from causalchamber.ground_truth import main as gt

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


def load_matrix(dataset, exp_name, feature_cols):
    """Load experiment data."""
    df = dataset.get_experiment(name=exp_name).as_pandas_dataframe()
    return df[feature_cols].to_numpy(dtype=float)


def stack_experiments(dataset, experiments, feature_cols):
    """Stack multiple experiments."""
    return np.vstack([load_matrix(dataset, exp, feature_cols) for exp in experiments])


def drop_constant_features(blocks, feature_cols, partition_parts):
    """Remove features with zero variance."""
    mask = blocks["obs"].std(axis=0) > 0
    if mask.all():
        return blocks, feature_cols, partition_parts
    feature_cols = [name for name, keep in zip(feature_cols, mask) if keep]
    partition_parts = [atoms for atoms, keep in zip(partition_parts, mask) if keep]
    blocks = {key: value[:, mask] for key, value in blocks.items()}
    return blocks, feature_cols, partition_parts


def build_true_graph(parts, true_dag_full):
    """Build graph from partition."""
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
    """Get descendant mask."""
    closure = set(atom_indices)
    for idx in list(atom_indices):
        closure.update(nx.descendants(true_dag_full, idx))
    mask = np.zeros(len(parts), dtype=bool)
    for part_idx, atoms in enumerate(parts):
        if atoms & closure:
            mask[part_idx] = True
    return mask


def ground_truth_partition(target_dict, parts, true_dag_full):
    """Compute ground-truth partition labels."""
    from repare.repare import _get_totally_ordered_partition

    ordered_masks = {}
    for order, label in enumerate(sorted(target_dict)):
        atom_union = set().union(*[parts[idx] for idx in target_dict[label]])
        ordered_masks[str(order)] = descendant_mask(atom_union, parts, true_dag_full)
    partition = _get_totally_ordered_partition(ordered_masks)
    labels = np.zeros(len(parts), dtype=int)
    for label, block in enumerate(partition):
        labels[list(block)] = label
    return partition, labels


def prepare_dataset(
    dataset_name, root, chamber, configuration, single_target_experiments
):
    """Prepare and return all data structures."""
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    dataset = Dataset(dataset_name, root=root_path, download=True)

    full_feature_cols = gt.variables(chamber, configuration)
    full_idx = {name: i for i, name in enumerate(full_feature_cols)}
    feature_cols = [col for col in REFERENCE_VARIABLES if col in full_idx]

    if not feature_cols:
        raise ValueError("No overlapping variables.")

    base_blocks = {
        "obs": load_matrix(dataset, OBS_EXPERIMENT, feature_cols),
        "rgb": stack_experiments(dataset, RGB_EXPERIMENTS, feature_cols),
        "pol": stack_experiments(dataset, POL_EXPERIMENTS, feature_cols),
    }

    for label, exp_name in single_target_experiments.items():
        base_blocks[label] = load_matrix(dataset, exp_name, feature_cols)

    partition_parts = [{full_idx[col]} for col in feature_cols]
    blocks, feature_cols, partition_parts = drop_constant_features(
        base_blocks, feature_cols, partition_parts
    )

    name_to_idx = {name: i for i, name in enumerate(feature_cols)}
    group_spec = {"rgb": {"red", "green", "blue"}, "pol": {"pol_1", "pol_2"}}
    group_targets = {}
    for label, names in group_spec.items():
        idxs = {name_to_idx[name] for name in names if name in name_to_idx}
        if idxs:
            group_targets[label] = idxs

    true_dag_full = nx.DiGraph()
    true_dag_full.add_edges_from(
        (full_idx[u], full_idx[v]) for u, v in gt.edges(chamber, configuration)
    )
    true_graph = build_true_graph(partition_parts, true_dag_full)
    _, true_labels = ground_truth_partition(
        group_targets, partition_parts, true_dag_full
    )

    single_env_labels = [
        label for label in single_target_experiments if label in name_to_idx
    ]
    single_env_targets = {label: {name_to_idx[label]} for label in single_env_labels}
    single_env_data = [blocks[label] for label in single_env_labels]

    return {
        "blocks": blocks,
        "feature_cols": feature_cols,
        "partition_parts": partition_parts,
        "group_targets": group_targets,
        "true_graph": true_graph,
        "true_labels": true_labels,
        "true_dag_full": true_dag_full,
        "name_to_idx": name_to_idx,
        "single_env_labels": single_env_labels,
        "single_env_targets": single_env_targets,
        "single_env_data": single_env_data,
    }


def main():
    dataset_name = snakemake.params.dataset
    root = snakemake.params.root
    chamber = snakemake.params.chamber
    configuration = snakemake.params.configuration
    target_mode = snakemake.params.target_mode

    single_target_map = TARGET_MODE_MAP.get(
        target_mode, DEFAULT_SINGLE_TARGET_EXPERIMENTS
    )
    data = prepare_dataset(
        dataset_name, root, chamber, configuration, single_target_map
    )

    with open(snakemake.output.blocks, "wb") as f:
        pickle.dump(data["blocks"], f)
    with open(snakemake.output.features, "w") as f:
        json.dump({"feature_cols": data["feature_cols"]}, f)
    with open(snakemake.output.partition, "wb") as f:
        pickle.dump(data["partition_parts"], f)
    with open(snakemake.output.grouptargets, "wb") as f:
        pickle.dump(data["group_targets"], f)
    with open(snakemake.output.truegraph, "wb") as f:
        pickle.dump(data["true_graph"], f)
    with open(snakemake.output.truelabels, "wb") as f:
        pickle.dump(data["true_labels"], f)
    with open(snakemake.output.truedagfull, "wb") as f:
        pickle.dump(data["true_dag_full"], f)
    with open(snakemake.output.nametoidx, "wb") as f:
        pickle.dump(data["name_to_idx"], f)
    with open(snakemake.output.singleenvlabels, "wb") as f:
        pickle.dump(data["single_env_labels"], f)
    with open(snakemake.output.singleenvtargets, "wb") as f:
        pickle.dump(data["single_env_targets"], f)
    with open(snakemake.output.singleenvdata, "wb") as f:
        pickle.dump(data["single_env_data"], f)


if __name__ == "__main__":
    main()
