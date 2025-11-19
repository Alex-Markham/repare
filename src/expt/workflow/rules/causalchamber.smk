base_path = "results/causalchamber/"


rule fit_causalchamber_grid:
    output:
        dag=base_path + "dag.png",
        summary=base_path + "dag_summary.txt",
        metrics=base_path + "grid_metrics.csv",
        method_metrics=base_path + "method_metrics.csv",
        ut_metrics=base_path + "ut_grid_metrics.csv",
    params:
        dataset="lt_interventions_standard_v1",
        root="data/causalchamber",
        chamber="lt",
        configuration="standard",
        alphas=[0.0001, 0.001, 0.01, 0.1],
        betas=[0.0001, 0.001, 0.01, 0.1],
        ut_alphas=[0.0001, 0.001, 0.01, 0.1],
    script:
        "../scripts/fit_causalchamber_grid.py"


rule causalchamber_all:
    input:
        base_path + "dag.png",
        base_path + "dag_summary.txt",
        base_path + "grid_metrics.csv",
        base_path + "method_metrics.csv",
        base_path + "ut_grid_metrics.csv",
