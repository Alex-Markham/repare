# ruff: noqa: E402
from snakemake.io import temp

cfg = config.get("scalability", {})

GRAPH_FAMILY = str(cfg.get("graph_family", "erdos_renyi"))
EDGE_PROBABILITY = float(cfg.get("edge_probability", 0.15))
NUM_INTERVS = int(cfg.get("num_intervs", 5))
INTERVENTION_TYPE = str(cfg.get("intervention_type", "soft"))
NODE_COUNTS = [int(n) for n in cfg.get("num_nodes", [10])]
SAMPLE_SIZES = [int(s) for s in cfg.get("sample_sizes", [10])]
SEEDS = [int(seed) for seed in cfg.get("seeds", list(range(20)))]

DEFAULT_METHODS = {
    "RePaRe": {
        "label": "RePaRe",
        "alpha": 0.01,
        "beta": 0.01,
        "assume": "gaussian",
        "refine_test": "ttest",
        "intervention_type": INTERVENTION_TYPE,
    }
}

METHOD_CONFIGS = cfg.get("methods") or DEFAULT_METHODS
METHODS = list(METHOD_CONFIGS.keys())
METHOD_LABELS = {
    name: cfg.get("label", name) for name, cfg in METHOD_CONFIGS.items()
}
METHOD_PARAMS = {
    name: {k: v for k, v in cfg.items() if k != "label"}
    for name, cfg in METHOD_CONFIGS.items()
}

GRAPH_BASE_DIR = (
    "results/scalability/"
    f"graph={GRAPH_FAMILY}/edge_prob={EDGE_PROBABILITY}"
)
SCALABILITY_BASE = (
    GRAPH_BASE_DIR + "/num_nodes={num_nodes}/samp_size={samp_size}/seed={seed}"
)
DATASET_PATH = SCALABILITY_BASE + "/dataset.npz"
MODEL_DIR = SCALABILITY_BASE + "/method={method}"
MODEL_PATH = MODEL_DIR + "/model.pkl"
FIT_METADATA_PATH = MODEL_DIR + "/fit_metadata.json"
METRICS_PATH = MODEL_DIR + "/metrics.csv"
SUMMARY_PATH = (
    "results/scalability/summary/"
    f"graph={GRAPH_FAMILY}_edge_prob={EDGE_PROBABILITY}_method={{method}}.csv"
)
PLOT_PATH = (
    "results/scalability/plots/"
    f"graph={GRAPH_FAMILY}_edge_prob={EDGE_PROBABILITY}_method={{method}}.pdf"
)
PLOT_RUNTIME_NODES_PATH = (
    "results/scalability/plots/"
    f"graph={GRAPH_FAMILY}_edge_prob={EDGE_PROBABILITY}_method={{method}}_runtime_vs_nodes.pdf"
)
CLEANUP_TOKEN = (
    "results/scalability/clean/"
    f"cleanup_graph={GRAPH_FAMILY}_edge_prob={EDGE_PROBABILITY}.done"
)


rule scalability_cleanup:
    output:
        token=temp(CLEANUP_TOKEN),
    run:
        import shutil
        from pathlib import Path

        base = Path(GRAPH_BASE_DIR)
        if base.exists():
            shutil.rmtree(base)
        base.mkdir(parents=True, exist_ok=True)
        Path(output.token).parent.mkdir(parents=True, exist_ok=True)
        Path(output.token).touch()


rule scalability_data_generation:
    input:
        cleanup=CLEANUP_TOKEN,
    output:
        dataset=temp(DATASET_PATH),
    params:
        num_intervs=NUM_INTERVS,
        graph_family=GRAPH_FAMILY,
        edge_probability=EDGE_PROBABILITY,
        intervention_type=INTERVENTION_TYPE,
    script:
        "../scripts/gen_scalability.py"


rule scalability_model_fitting:
    input:
        data=DATASET_PATH,
    output:
        model=temp(MODEL_PATH),
        metadata=temp(FIT_METADATA_PATH),
    params:
        method=lambda wildcards: wildcards.method,
        method_config=lambda wildcards: METHOD_PARAMS[wildcards.method],
    script:
        "../scripts/fit_scalability.py"


rule scalability_evaluation:
    input:
        data=DATASET_PATH,
        model=MODEL_PATH,
        fit_metadata=FIT_METADATA_PATH,
    output:
        metrics=temp(METRICS_PATH),
    params:
        method=lambda wildcards: wildcards.method,
        method_label=lambda wildcards: METHOD_LABELS[wildcards.method],
    script:
        "../scripts/eval_scalability.py"


rule scalability_aggregation:
    input:
        lambda wildcards: expand(
            METRICS_PATH,
            method=[wildcards.method],
            num_nodes=NODE_COUNTS,
            samp_size=SAMPLE_SIZES,
            seed=SEEDS,
        ),
    output:
        summary=SUMMARY_PATH,
    script:
        "../scripts/aggregate_scalability.py"


rule scalability_plot:
    input:
        summary=SUMMARY_PATH,
    output:
        sample_plot=PLOT_PATH,
        runtime_nodes_plot=PLOT_RUNTIME_NODES_PATH,
    params:
        method_label=lambda wildcards: METHOD_LABELS[wildcards.method],
    script:
        "../scripts/plot_scalability.py"


rule scalability_all:
    input:
        expand(PLOT_PATH, method=METHODS),
        expand(PLOT_RUNTIME_NODES_PATH, method=METHODS)


def _default_existing_plot_path(summary_path: str, runtime: bool = False) -> str:
    plot_path = summary_path.replace("/summary/", "/plots/", 1)
    suffix = "_runtime_vs_nodes" if runtime else ""
    return plot_path.replace(".csv", f"{suffix}.pdf")


_PLOT_EXISTING_CFG = config.get("plot_existing_summary")
if _PLOT_EXISTING_CFG:
    _existing_summary = _PLOT_EXISTING_CFG["summary"]
    _existing_sample_plot = _PLOT_EXISTING_CFG.get(
        "sample_plot", _default_existing_plot_path(_existing_summary)
    )
    _existing_runtime_plot = _PLOT_EXISTING_CFG.get(
        "runtime_plot",
        _default_existing_plot_path(_existing_summary, runtime=True),
    )
    _existing_label = _PLOT_EXISTING_CFG.get("label", "existing summary")

    rule scalability_plot_existing_summary:
        input:
            summary=_existing_summary,
        output:
            sample_plot=_existing_sample_plot,
            runtime_nodes_plot=_existing_runtime_plot,
        params:
            method_label=_existing_label,
        script:
            "../scripts/plot_scalability.py"
