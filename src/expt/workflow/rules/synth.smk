synth_base_path = "results/synth/num_intervs={num_intervs}/density={density}/samp_size={samp_size}/seed={seed}/"


rule data_generation_synth:
    output:
        synth_base_path + "dataset.npz",
    params:
        num_nodes=10,
        num_intervs=lambda wildcards: wildcards.num_intervs,
        intervention_type="soft",
    script:
        "../scripts/gen_synth.py"


rule model_fitting_synth:
    input:
        data=synth_base_path + "dataset.npz",
    output:
        synth_base_path + "model.pkl",
    params:
        alpha=0.0001,
        beta=0.0001,
        assume="gaussian",
        refine_test="ttest",
    script:
        "../scripts/fit_synth.py"


rule evaluation_synth:
    input:
        data=synth_base_path + "dataset.npz",
        model=synth_base_path + "model.pkl",
    output:
        synth_base_path + "metrics.csv",
    script:
        "../scripts/eval_synth.py"


rule aggregation_synth:
    input:
        lambda wildcards, bp=synth_base_path: expand(
            bp + "metrics.csv",
            num_intervs=[wildcards.num_intervs],
            seed=range(10),
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
        ),
    output:
        "results/synth/results_num_intervs_{num_intervs}.csv",
    script:
        "../scripts/aggregate.py"


rule plot_synth:
    input:
        "results/synth/results_num_intervs_{num_intervs}.csv",
    output:
        "results/fscores_num_intervs_{num_intervs}.pdf",
        "results/ari_num_intervs_{num_intervs}.pdf",
    script:
        "../scripts/plot_synth.py"


rule synth_all:
    input:
        expand("results/fscores_num_intervs_{num_intervs}.pdf", num_intervs=[2, 5, 8]),
        expand("results/ari_num_intervs_{num_intervs}.pdf", num_intervs=[2, 5, 8])
