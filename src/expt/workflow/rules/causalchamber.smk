from snakemake.io import directory

base_path = "results/causalchamber/"
base_path_all = "results/causalchamber_all/"


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
    script:
        "../scripts/fit_causalchamber_grid.py"


rule fit_causalchamber_grid_all:
    output:
        dag=base_path_all + "dag.png",
        summary=base_path_all + "dag_summary.txt",
        metrics=base_path_all + "grid_metrics.csv",
        method_metrics=base_path_all + "method_metrics.csv",
        ut_metrics=base_path_all + "ut_grid_metrics.csv",
        grid_dir=directory(base_path_all + "grid_runs"),
    params:
        dataset="lt_interventions_standard_v1",
        root="data/causalchamber",
        chamber="lt",
        configuration="standard",
        alphas=[0.0001, 0.001, 0.01, 0.1],
        betas=[0.0001, 0.001, 0.01, 0.1],
        ut_alphas=[0.0001, 0.001, 0.01, 0.1],
        target_mode="all",
        include_grouped=False,
    script:
        "../scripts/fit_causalchamber_grid.py"


rule causalchamber_all:
    input:
        base_path + "dag.png",
        base_path + "dag_summary.txt",
        base_path + "grid_metrics.csv",
        base_path + "method_metrics.csv",
        base_path + "ut_grid_metrics.csv",
        base_path + "grid_runs",
        base_path_all + "dag.png",
        base_path_all + "dag_summary.txt",
        base_path_all + "grid_metrics.csv",
        base_path_all + "method_metrics.csv",
        base_path_all + "ut_grid_metrics.csv",
        base_path_all + "grid_runs",
