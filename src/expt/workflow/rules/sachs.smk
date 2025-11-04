base_path = "results/sachs/"


rule prepare_sachs:
    input:
        "src/expt/resources/sachs.interventional.txt",
    output:
        base_path + "dataset.pkl",
    script:
        "../scripts/prepare_sachs.py"


rule fit_sachs:
    input:
        base_path + "dataset.pkl",
    output:
        dag=base_path + "obs_idx={obs_idx}/dag.png",
        model=base_path + "obs_idx={obs_idx}/model.pkl",
    params:
        alpha=0.05,
        beta=0.1,
        assume=None,
    script:
        "../scripts/fit_sachs.py"


rule eval_sachs:
    input:
        model=base_path + "obs_idx={obs_idx}/model.pkl",
    output:
        base_path + "obs_idx={obs_idx}/eval.csv",
    script:
        "../scripts/eval_sachs.py"


rule aggregate_sachs:
    input:
        expand(base_path + "obs_idx={obs_idx}/eval.csv", obs_idx=[0, 2, 4, 7, 8, 9]),
    output:
        base_path + "results.csv",
    script:
        "../scripts/aggregate.py"


rule sachs_results_to_latex:
    input:
        base_path + "results.csv",
    output:
        "results/sachs.tex",
    script:
        "../scripts/make_table.py"
