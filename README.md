# RePaRe #

This repo provides an implementation of the recursive partition refinement (RePaRe) algorithm and experimental results, suitable for double-blind submission.
Included license files retain third‑party names only to comply with upstream license terms and do not violate anonymity for review.

The interventional coarsening implementation is provided by the `PartitionDagModelIvn` class in [`src/repare/repare.py`](src/repare/repare.py).

Experiments are organized into a Snakemake workflow, with the [`src/expt/workflow/Snakefile`](src/expt/workflow/Snakefile) entry point.

Dependencies, versioning, and installation can all be handled by `uv`, with the included `pyproject.toml` and `uv.lock` containing all necessary information.

After [installing uv](https://docs.astral.sh/uv/getting-started/):
- `uv run pytest tests/` can be run as a quick check from project root
- `uv run snakemake results/ari_num_intervs_2.pdf --cores all` can be run from `src/expt/` to reproduce Figure 1(a); it runs 330 experiments, taking about 10 minutes depending on the cpu.
- `uv run snakemake all --cores all` can be run from `src/expt/` to reproduce all experiments; it takes less than an hour, depending on cpu, and the first call will download the third-party `causalchambers` dataset.

The first time `uv run..` is called, it will download and install all dependencies.
Decrease the number of cores (e.g., `10` instead of `all`) as needed.
All snakemake outputs are saved in `src/expt/results/`.

