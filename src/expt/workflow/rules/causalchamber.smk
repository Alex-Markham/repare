"""CausalChamber"""

from snakemake.io import directory

ALPHAS = [0.0001, 0.001, 0.01, 0.1]
BETAS = [0.0001, 0.001, 0.01, 0.1]
DATASET = "lt_interventions_standard_v1"
CHAMBER = "lt"
CONFIGURATION = "standard"

BASE = "results/causalchamber/"

PREP_FILES = {
    "blocks": BASE + "preprocessed/blocks.pkl",
    "features": BASE + "preprocessed/features.json",
    "partition": BASE + "preprocessed/partition_parts.pkl",
    "grouptargets": BASE + "preprocessed/grouptargets.pkl",
    "truegraph": BASE + "preprocessed/truegraph.pkl",
    "truelabels": BASE + "preprocessed/truelabels.pkl",
    "truedagfull": BASE + "preprocessed/truedagfull.pkl",
    "nametoidx": BASE + "preprocessed/nametoidx.pkl",
    "singleenvlabels": BASE + "preprocessed/singleenvlabels.pkl",
    "singleenvtargets": BASE + "preprocessed/singleenvtargets.pkl",
    "singleenvdata": BASE + "preprocessed/singleenvdata.pkl",
}


rule causalchamber_prepare:
    output:
        blocks=BASE + "preprocessed/blocks.pkl",
        features=BASE + "preprocessed/features.json",
        partition=BASE + "preprocessed/partition_parts.pkl",
        grouptargets=BASE + "preprocessed/grouptargets.pkl",
        truegraph=BASE + "preprocessed/truegraph.pkl",
        truelabels=BASE + "preprocessed/truelabels.pkl",
        truedagfull=BASE + "preprocessed/truedagfull.pkl",
        nametoidx=BASE + "preprocessed/nametoidx.pkl",
        singleenvlabels=BASE + "preprocessed/singleenvlabels.pkl",
        singleenvtargets=BASE + "preprocessed/singleenvtargets.pkl",
        singleenvdata=BASE + "preprocessed/singleenvdata.pkl",
    params:
        dataset=DATASET,
        chamber=CHAMBER,
        configuration=CONFIGURATION,
        root="data/causalchamber",
        target_mode="default",
        include_grouped=True,
    script:
        "../scripts/causalchamber_prepare.py"


rule causalchamber_repare_grouped:
    input:
        **PREP_FILES,
    output:
        metrics_csv=BASE + "repare_grouped/metrics.csv",
        score_dag=BASE + "repare_grouped/score_dag.png",
        oracle_dag=BASE + "repare_grouped/oracle_dag.png",
        score_model=temp(BASE + "repare_grouped/score_model.pkl"),
        oracle_model=temp(BASE + "repare_grouped/oracle_model.pkl"),
        score_params=BASE + "repare_grouped/score_params.json",
        oracle_params=BASE + "repare_grouped/oracle_params.json",
    params:
        alphas=ALPHAS,
        betas=BETAS,
        mode="grouped",
    script:
        "../scripts/causalchamber_repare.py"


rule causalchamber_repare_ungrouped:
    input:
        **PREP_FILES,
    output:
        metrics_csv=BASE + "repare_ungrouped/metrics.csv",
        score_dag=BASE + "repare_ungrouped/score_dag.png",
        oracle_dag=BASE + "repare_ungrouped/oracle_dag.png",
        score_model=temp(BASE + "repare_ungrouped/score_model.pkl"),
        oracle_model=temp(BASE + "repare_ungrouped/oracle_model.pkl"),
        score_params=BASE + "repare_ungrouped/score_params.json",
        oracle_params=BASE + "repare_ungrouped/oracle_params.json",
    params:
        alphas=ALPHAS,
        betas=BETAS,
        mode="ungrouped",
    script:
        "../scripts/causalchamber_repare.py"


rule causalchamber_gies:
    input:
        **PREP_FILES,
    output:
        metrics=BASE + "gies/metrics.json",
        dag=BASE + "gies/dag.png",
    script:
        "../scripts/causalchamber_gies.py"


rule causalchamber_gnies:
    input:
        **PREP_FILES,
    output:
        metrics=BASE + "gnies/metrics.json",
    script:
        "../scripts/causalchamber_gnies.py"


rule causalchamber_utigsp:
    input:
        **PREP_FILES,
    output:
        ut_metrics=BASE + "utigsp/ut_metrics.csv",
    script:
        "../scripts/causalchamber_utigsp.py"


rule causalchamber_aggregate:
    input:
        grouped_metrics=BASE + "repare_grouped/metrics.csv",
        grouped_score_dag=BASE + "repare_grouped/score_dag.png",
        grouped_oracle_dag=BASE + "repare_grouped/oracle_dag.png",
        grouped_score_model=BASE + "repare_grouped/score_model.pkl",
        grouped_oracle_model=BASE + "repare_grouped/oracle_model.pkl",
        grouped_score_params=BASE + "repare_grouped/score_params.json",
        grouped_oracle_params=BASE + "repare_grouped/oracle_params.json",
        ungrouped_metrics=BASE + "repare_ungrouped/metrics.csv",
        ungrouped_score_dag=BASE + "repare_ungrouped/score_dag.png",
        ungrouped_oracle_dag=BASE + "repare_ungrouped/oracle_dag.png",
        ungrouped_score_model=BASE + "repare_ungrouped/score_model.pkl",
        ungrouped_oracle_model=BASE + "repare_ungrouped/oracle_model.pkl",
        ungrouped_score_params=BASE + "repare_ungrouped/score_params.json",
        ungrouped_oracle_params=BASE + "repare_ungrouped/oracle_params.json",
        gies=BASE + "gies/metrics.json",
        gnies=BASE + "gnies/metrics.json",
        ut=BASE + "utigsp/ut_metrics.csv",
    output:
        grid_metrics=BASE + "grid_metrics.csv",
        dag=BASE + "dag.png",
        grid_dir=directory(BASE + "grid_runs"),
        dag_summary="results/causalchamber_dags.txt",
        method_metrics="results/causalchamber_summary.csv",
    script:
        "../scripts/causalchamber_aggregate.py"
