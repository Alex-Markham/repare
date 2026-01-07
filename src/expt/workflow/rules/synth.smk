base_path = "results/synth/graph={graph}/num_nodes={num_nodes}/num_intervs={num_intervs}/density={density}/samp_size={samp_size}/seed={seed}/"


rule generate:
    output:
        base_path + "dataset.npz",
    params:
        intervention_type="soft",
    script:
        "../scripts/generate.py"


rule fit:
    input:
        data=base_path + "dataset.npz",
    output:
        base_path + "model.pkl",
    params:
        alpha=0.0001,
        beta=0.0001,
        assume="gaussian",
        refine_test="ttest",
    script:
        "../scripts/fit.py"


rule evaluate:
    input:
        data=base_path + "dataset.npz",
        model=base_path + "model.pkl",
    output:
        base_path + "metrics.csv",
    script:
        "../scripts/evaluate.py"


rule collect:
    input:
        expand(
            base_path + "metrics.csv",
            num_nodes=[10],
            seed=range(2),
            density=[0.2, 0.5, 0.8],
            samp_size=[
                10,
                100,
                200,
                500,
                1000,
                2000,
                5000,
                10000,
                20000,
                50000,
                100000,
            ],
            allow_missing=True,
        ),
    output:
        results="results/{graph}_results_ivn={num_intervs}.csv",
    script:
        "../scripts/collect.py"


rule plot:
    input:
        rules.collect.output["results"],
    output:
        ari="results/{graph}_ari_ivn={num_intervs}.pdf",
        fscore="results/{graph}_fscore_ivn={num_intervs}.pdf",
    script:
        "../scripts/plot.py"
