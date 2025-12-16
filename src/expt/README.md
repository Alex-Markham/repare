add some intsructions here, maybe also the snakemake badge, and consider if it's worth publishing the workflow: [more info](https://snakemake.readthedocs.io/en/stable/project_info/citations.html)

## RePaRe failure sweep

The workflow in `src/expt/workflow/repare_failure.smk` generates Erdos–Rényi datasets (edge probability 0.3) via `sempler`, fits RePaRe (or any future method registered in `scripts/fit_repare_failure.py`), evaluates ARI/F-score/runtime, aggregates all results, and plots degradation curves. It never touches the legacy workflows; new code lives in the `scripts/` helpers above.

Run the sweep with:

```bash
snakemake -s src/expt/workflow/repare_failure.smk -j4
```

Datasets land in `results/repare_failure/data/.../dataset.npz` and contain `obs`, intervention datasets, `weights`, `targets`, plus metadata (`graph_family`, `edge_probability`, `num_nodes`, `num_intervs`, `samp_size`, `seed`). Any algorithm runner can reuse these files by adding a new entry to the `METHOD_REGISTRY` in `scripts/fit_repare_failure.py`, guaranteeing fair comparisons on identical data.
