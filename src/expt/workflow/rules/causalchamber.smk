from snakemake.io import directory

base_path = "results/causalchamber/"

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
        
