# RePaRe #

This repo provides an implementation of the recursive partition refinement (RePaRe) algorithm and experimental results from the preprint *Coarsening Causal DAG Models* ([arXiv:2601.10531](https://arxiv.org/abs/2601.10531) [stat.ML]).

The interventional coarsening implementation is provided by the `PartitionDagModelIvn` class in [`src/repare/repare.py`](src/repare/repare.py).

Experiments are organized into a Snakemake workflow, with the [`src/expt/workflow/Snakefile`](src/expt/workflow/Snakefile) entry point.

Dependencies, versioning, and installation can all be handled by `uv`, with the included `pyproject.toml` and `uv.lock` containing all necessary information.
The pinned `requirements.txt` with hashes is provided for users of other package managers.

After [installing uv](https://docs.astral.sh/uv/getting-started/):
- `uv run pytest tests/` can be run as a quick check from project root
- `uv run snakemake results/er_ari_ivn=2.pdf --cores all --forceall` can be run from `src/expt/` to reproduce Figure 1(a); it runs 330 experiments, taking about 10 minutes depending on the cpu.
- `uv run snakemake all --cores all --forceall` can be run from `src/expt/` to reproduce all experiments; it takes less than an hour, depending on cpu, and the first call will download the third-party `causalchambers` dataset.

The first time `uv run <...>` is called, it will download and install all dependencies.
Decrease the number of cores (e.g., `10` instead of `all`) as needed.
All snakemake outputs are saved in `src/expt/results/`.

# Citing #

If you find the code helpful, please cite it using the following bibtex:
```
@InProceedings{madaleno2026,
  title = 	     {Coarsening Causal {DAG} Models},
  author =       {Francisco Madaleno and Pratik Misra and Alex Markham},
  booktitle = 	 {Proceedings of the Fifth Conference on Causal Learning and Reasoning},
  year = 	     {2026},
  editor = 	     {Bijan Mazaheri and Niels Richard Hansen},
  series = 	     {Proceedings of Machine Learning Research},
  month = 	     {Apr},
  publisher =    {PMLR},
}

```

# Contact #

Feel free to [raise an issue](https://github.com/Alex-Markham/repare/issues/new/choose) or [email me](mailto:alex.markham@causal.dev) with any questions about reproducing the experimental results, modifying the code to your problem, or applying RePaRe to your data!
