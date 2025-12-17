import json
import pickle
import time
from typing import Any, Dict, Tuple

import numpy as np
from repare.repare import PartitionDagModelIvn


def _method_registry():
    registry = {}

    def register(name):
        def decorator(func):
            registry[name] = func
            return func

        return decorator

    return register, registry


register_method, METHOD_REGISTRY = _method_registry()


def _load_dataset(path: str) -> Dict[str, Any]:
    return np.load(path, allow_pickle=True)


def _map_intervention_type(value: str) -> str:
    lowered = str(value).lower()
    if lowered in {"shift", "soft"}:
        return "soft"
    if lowered in {"hard", "do"}:
        return "hard"
    return lowered


@register_method("RePaRe")
def _fit_repare(dataset, method_cfg: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    alpha = float(method_cfg.get("alpha", 0.0001))
    beta = float(method_cfg.get("beta", 0.0001))
    assume_param = method_cfg.get("assume")
    assume = None if assume_param in (None, "None", "none") else assume_param
    refine_param = method_cfg.get("refine_test", "ttest")
    refine_test = ("ks" if refine_param is None else str(refine_param)).lower()
    intervention_type = _map_intervention_type(
        method_cfg.get("intervention_type", "soft")
    )

    data_dict = {"obs": (dataset["obs"], set(), "obs")}
    targets = dataset["targets"]
    for idx, target in enumerate(targets):
        tgt = set(np.atleast_1d(target).astype(int))
        data_dict[str(idx)] = (dataset[str(idx)], tgt, intervention_type)

    model = PartitionDagModelIvn()
    start = time.perf_counter()
    model.fit(data_dict, alpha, beta, assume, refine_test=refine_test)
    runtime_sec = time.perf_counter() - start

    metadata = {
        "alpha": alpha,
        "beta": beta,
        "assume": assume,
        "refine_test": refine_test,
        "intervention_type": intervention_type,
        "runtime_sec": runtime_sec,
    }
    return model, metadata


def main():
    method_name = snakemake.params.method
    method_cfg = snakemake.params.method_config
    dataset = _load_dataset(snakemake.input.data)
    fit_fn = METHOD_REGISTRY.get(method_name)
    if fit_fn is None:
        raise ValueError(f"Unknown method '{method_name}'.")
    model, metadata = fit_fn(dataset, method_cfg)

    with open(snakemake.output.model, "wb") as f:
        pickle.dump(model, f)

    metadata = {"method": method_name, **metadata}
    with open(snakemake.output.metadata, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


main()
