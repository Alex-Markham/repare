# RePaRe #

This repo provides an implementation of the recursive partition refinement (RePaRe) algorithm and experimental results, suitable for double-blind submission.
Included license files retain third‑party names only to comply with upstream license terms and do not violate anonymity for review.

The interventional coarsening implementation is provided by the `PartitionDagModelIvn` class in (`src/repare/repare.py`)[src/repare/repare.py].

Experiments are organized into a Snakemake workflow, with the (`src/expt/workflow/Snakefile`)[src/expt/workflow/Snakefile] entry point.

Dependencies, versioning, and installation can all be handled by `uv`, with the included `pyproject.toml` and `uv.lock` containing all necessary information.

After (installing uv)[https://docs.astral.sh/uv/getting-started/]:
- `uv run pytest tests/` can be run as a quick check from project root
- `uv run snakemake synth_all --cores 10` can be run from `src/expt/` to quickly produce the plots from Figure 2
- `uv run snakemake all --cores 10` can be run from `src/expt/` to reproduce all experiments.

The first time `uv run..` is called, it will install all dependencies.
In/decrease the number of cores used as needed.
All snakemake outputs are saved in `src/expt/results/`.

