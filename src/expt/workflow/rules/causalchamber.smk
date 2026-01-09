from snakemake.io import directory

base_path = "results/causalchamber/"
chain_base_path = "results/causalchamber_chain/"

rule fit_causalchamber_grid:
    output:
        dag=base_path + "dag.png",
        summary=base_path + "dag_summary.txt",
        metrics=base_path + "grid_metrics.csv",
        method_metrics=base_path + "method_metrics.csv",
        ut_metrics=base_path + "ut_grid_metrics.csv",
        grid_dir=directory(base_path + "grid_runs"),
    params:
        dataset="lt_interventions_standard_v1",
        root="data/causalchamber",
        chamber="lt",
        configuration="standard",
        alphas=[0.0001, 0.001, 0.01, 0.1],
        betas=[0.0001, 0.001, 0.01, 0.1],
        ut_alphas=[0.0001, 0.001, 0.01, 0.1],
        target_mode="default",
        include_grouped=True,
        score_mode="gnies",
        intra_part_connectivity=True,
    script:
        "../scripts/fit_causalchamber_grid.py"

rule fit_causalchamber_grid_no_edges:
    output:
        dag=base_path + "no_edges/dag.png",
        summary=base_path + "no_edges/dag_summary.txt",
        metrics=base_path + "no_edges/grid_metrics.csv",
        method_metrics=base_path + "no_edges/method_metrics.csv",
        ut_metrics=base_path + "no_edges/ut_grid_metrics.csv",
        grid_dir=directory(base_path + "no_edges/grid_runs"),
    params:
        dataset="lt_interventions_standard_v1",
        root="data/causalchamber",
        chamber="lt",
        configuration="standard",
        alphas=[0.0001, 0.001, 0.01, 0.1],
        betas=[0.0001, 0.001, 0.01, 0.1],
        ut_alphas=[0.0001, 0.001, 0.01, 0.1],
        target_mode="default",
        include_grouped=True,
        score_mode="gnies",
        intra_part_connectivity=False,
    script:
        "../scripts/fit_causalchamber_grid.py"


rule causal_chamber_chain:
    output:
        dag=chain_base_path + "dag.png",
        summary=chain_base_path + "dag_summary.txt",
        metrics=chain_base_path + "grid_metrics.csv",
        method_metrics=chain_base_path + "method_metrics.csv",
        ut_metrics=chain_base_path + "ut_grid_metrics.csv",
        grid_dir=directory(chain_base_path + "grid_runs"),
    params:
        dataset="lt_interventions_standard_v1",
        root="data/causalchamber",
        chamber="lt",
        configuration="standard",
        alphas=[0.0001, 0.001, 0.01, 0.1],
        betas=[0.0001, 0.001, 0.01, 0.1],
        ut_alphas=[0.0001, 0.001, 0.01, 0.1],
        target_mode="default",
        include_grouped=True,
        score_mode="chain",
    script:
        "../scripts/fit_causalchamber_grid.py"
        
rule causal_chamber_chain_no_edges:
    output:
        dag=chain_base_path + "no_edges/dag.png",
        summary=chain_base_path + "no_edges/dag_summary.txt",
        metrics=chain_base_path + "no_edges/grid_metrics.csv",
        method_metrics=chain_base_path + "no_edges/method_metrics.csv",
        ut_metrics=chain_base_path + "no_edges/ut_grid_metrics.csv",
        grid_dir=directory(chain_base_path + "no_edges/grid_runs"),
    params:
        dataset="lt_interventions_standard_v1",
        root="data/causalchamber",
        chamber="lt",
        configuration="standard",
        alphas=[0.0001, 0.001, 0.01, 0.1],
        betas=[0.0001, 0.001, 0.01, 0.1],
        ut_alphas=[0.0001, 0.001, 0.01, 0.1],
        target_mode="default",
        include_grouped=True,
        score_mode="chain",
        intra_part_connectivity=False,
    script:
        "../scripts/fit_causalchamber_grid.py"