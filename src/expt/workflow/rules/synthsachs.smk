base_path = "results/synthsachs/"
extended_path = base_path + "samp_size={samp_size}/seed={seed}"


rule synthesize_ground_truth:
    input:
        "resources/sachs.interventional.txt",
    output:
        base_path + "synthetic_ground_truth.pkl",
    script:
        "synthesize_ground_truth.py"


rule synthesize_data:
    input:
        base_path + "synthetic_ground_truth.pkl",
    output:
        extended_path + "dataset.npz",
    script:
        "../scripts/synthesize_sachs_data.py"


rule fit_synthsachs:
    input:
        extended_path + "dataset.npz",
    output:
        model=extended_path + "model.pkl",
        dag=extended_path + "dag.png",
    params:
        reference="pip3",
        alpha=0.05,
        beta=0.1,
        assume=None,
        refine_test="ks",
    script:
        "../scripts/fit_synthsachs.py"


rule eval_synthsachs:
    input:
        extended_path + "model.pkl",
    output:
        extended_path + "eval.csv",
    params:
        reference="pip3",
    script:
        "../scripts/eval_synthsachs.py"


rule aggregate_synthsachs:
    input:
        expand(extended_path + "eval.csv", samp_size=[100, 500, 1000], seed=range(5)),
    output:
        base_path + "results.csv",
    script:
        "../scripts/aggregate.py"


rule plot_synthsachs:
    input:
        base_path + "results.csv",
    output:
        "results/synthsachs_plot.pdf",
    script:
        "../scripts/plot_synthsachs.py"
