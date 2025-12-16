scale_free_base_path = "results/scale_free/num_intervs={num_intervs}/density={density}/samp_size={samp_size}/seed={seed}/"


rule data_generation_synth_scale_free:
    output:
        scale_free_base_path + "dataset.npz",
    params:
        num_nodes=10,
        num_intervs=lambda wildcards: wildcards.num_intervs,
        intervention_type="soft",
    script:
        "../scripts/gen_synth_scale_free.py"


rule model_fitting_synth_scale_free:
    input:
        data=scale_free_base_path + "dataset.npz",
    output:
        scale_free_base_path + "model.pkl",
    params:
        alpha=0.0001,
        beta=0.0001,
        assume="gaussian",
        refine_test="ttest",
    script:
        "../scripts/fit_synth_scale_free.py"


rule evaluation_synth_scale_free:
    input:
        data=scale_free_base_path + "dataset.npz",
        model=scale_free_base_path + "model.pkl",
    output:
        scale_free_base_path + "metrics.csv",
    script:
        "../scripts/eval_synth_scale_free.py"


rule aggregation_synth_scale_free:
    input:
        lambda wildcards, bp=scale_free_base_path: expand(
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
        "results/scale_free/results_num_intervs_{num_intervs}.csv",
    script:
        "../scripts/aggregate_scale_free.py"


rule plot_synth_scale_free:
    input:
        "results/scale_free/results_num_intervs_{num_intervs}.csv",
    output:
        "results/scale_free/fscores_num_intervs_{num_intervs}.pdf",
        "results/scale_free/ari_num_intervs_{num_intervs}.pdf",
    script:
        "../scripts/plot_synth_scale_free.py"


rule scale_free_all:
    input:
        expand("results/scale_free/fscores_num_intervs_{num_intervs}.pdf", num_intervs=[2, 5, 8]),
        expand("results/scale_free/ari_num_intervs_{num_intervs}.pdf", num_intervs=[2, 5, 8])
