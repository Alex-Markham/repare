base_path = "results/synth/density={density}/samp_size={samp_size}/seed={seed}/"


rule data_generation_synth:
    output:
        base_path + "dataset.npz",
    params:
        num_nodes=10,
        num_intervs=2,
    script:
        "../scripts/gen_synth.py"


rule model_fitting_synth:
    input:
        data=base_path + "dataset.npz",
    output:
        base_path + "model.pkl",
    params:
        alpha=0.01,
        beta=0.01,
        assume="gaussian",
    script:
        "../scripts/fit_synth.py"


rule evaluation_synth:
    input:
        data=base_path + "dataset.npz",
        model=base_path + "model.pkl",
    output:
        base_path + "metrics.csv",
    script:
        "../scripts/eval_synth.py"


rule aggregation_synth:
    input:
        expand(
            base_path + "metrics.csv",
            seed=range(10),
            # density=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            density=[0.2, 0.5, 0.8],
            samp_size=[
                100,
                200,
                500,
                1000,
                # 2000,
                # 5000,
                # 10000,
                # 20000,
                # 50000,
                # 100000,
                # 200000,
                # 500000,
                # 1000000,
            ],
        ),
    output:
        "results/synth/results.csv",
    script:
        "../scripts/aggregate.py"


rule plot_synth:
    input:
        "results/synth/results.csv",
    output:
        "results/fscores.pdf",
        "results/ari.pdf",
    script:
        "../scripts/plot_synth.py"
