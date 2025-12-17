import os
import yaml
from snakemake.io import directory

WORKFLOW_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(WORKFLOW_DIR, "config/scalability.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as _cfg:
    config = yaml.safe_load(_cfg)

GRAPH_FAMILY = config.get("graph_family", "erdos_renyi")
EDGE_PROB = config.get("edge_probability", 0.3)
NUM_INTERVS = config.get("num_intervs", 5)
INTERVENTION_TYPE = config.get("intervention_type", "soft")
NUM_NODES = config.get("num_nodes", [])
SAMPLE_SIZES = config.get("sample_sizes", [])
SEEDS = config.get("seeds", [])
METHODS = config.get("methods", {})
METHOD_NAMES = list(METHODS.keys())


def _dataset_path(w):
    return (
        f"results/scalability/data/graph={GRAPH_FAMILY}_p={EDGE_PROB}"
        f"/num_nodes={w.num_nodes}/samp_size={w.samp_size}/seed={w.seed}/dataset.npz"
    )


def _model_path(w):
    return (
        f"results/scalability/models/{w.method}/graph={GRAPH_FAMILY}_p={EDGE_PROB}"
        f"/num_nodes={w.num_nodes}/samp_size={w.samp_size}/seed={w.seed}/model.pkl"
    )


def _fit_metadata_path(w):
    return (
        f"results/scalability/models/{w.method}/graph={GRAPH_FAMILY}_p={EDGE_PROB}"
        f"/num_nodes={w.num_nodes}/samp_size={w.samp_size}/seed={w.seed}/fit_metadata.json"
    )


def _metrics_path(w):
    return (
        f"results/scalability/metrics/{w.method}/graph={GRAPH_FAMILY}_p={EDGE_PROB}"
        f"/num_nodes={w.num_nodes}/samp_size={w.samp_size}/seed={w.seed}/metrics.csv"
    )


def _aggregated_path(w):
    return (
        f"results/scalability/aggregated/{w.method}_graph={GRAPH_FAMILY}_p={EDGE_PROB}.csv"
    )


def _plot_path(w):
    return (
        f"results/scalability/plots/{w.method}_graph={GRAPH_FAMILY}_p={EDGE_PROB}.pdf"
    )


rule all:
    input:
        expand(_plot_path, method=METHOD_NAMES),


rule generate_scalability_data:
    output:
        dataset=_dataset_path,
    params:
        num_intervs=NUM_INTERVS,
        graph_family=GRAPH_FAMILY,
        edge_probability=EDGE_PROB,
        intervention_type=INTERVENTION_TYPE,
    script:
        "scripts/gen_scalability.py"


rule fit_scalability_model:
    input:
        data=_dataset_path,
    output:
        model=_model_path,
        metadata=_fit_metadata_path,
    params:
        method=lambda wildcards: wildcards.method,
        method_config=lambda wildcards: METHODS[wildcards.method],
    script:
        "scripts/fit_scalability.py"


rule eval_scalability_model:
    input:
        data=_dataset_path,
        model=_model_path,
        fit_metadata=_fit_metadata_path,
    output:
        metrics=_metrics_path,
    params:
        method=lambda wildcards: wildcards.method,
        method_label=lambda wildcards: METHODS[wildcards.method].get(
            "label", wildcards.method
        ),
    script:
        "scripts/eval_scalability.py"


rule aggregate_scalability_metrics:
    input:
        lambda wildcards: expand(
            _metrics_path,
            method=[wildcards.method],
            num_nodes=NUM_NODES,
            samp_size=SAMPLE_SIZES,
            seed=SEEDS,
        ),
    output:
        summary=_aggregated_path,
    params:
        method=lambda wildcards: wildcards.method,
    script:
        "scripts/aggregate_scalability.py"


rule plot_scalability:
    input:
        summary=_aggregated_path,
    output:
        plot=_plot_path,
    params:
        method=lambda wildcards: wildcards.method,
        method_label=lambda wildcards: METHODS[wildcards.method].get(
            "label", wildcards.method
        ),
    script:
        "scripts/plot_scalability.py"
