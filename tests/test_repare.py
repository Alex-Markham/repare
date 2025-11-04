import networkx as nx
import numpy as np
from numpy.random import default_rng as rng
from repare.repare import (
    PartitionDagModelIvn,
    PartitionDagModelOracle,
)
from sempler import LGANM
from sempler.generators import dag_avg_deg, intervention_targets


def test_fit_oracle_basic():
    true_order = [3, 1, 2, 4]
    true_adj = {1: [2], 3: [2], 2: [4]}

    model = PartitionDagModelOracle()
    model.fit(true_order, true_adj)

    assert len(model.dag.edges) == 3
    for u in true_adj.keys():
        for v in true_adj[u]:
            assert model.dag.has_edge((u,), (v,))


def test_fit_ivn_chain_gaussian():
    num_samples = 400

    def sample_env(seed, shift_target=None, shift=3.0):
        rng = np.random.default_rng(seed)
        x0 = rng.normal(size=num_samples)
        if shift_target == 0:
            x0 = rng.normal(loc=shift, size=num_samples)
        e1 = rng.normal(scale=0.4, size=num_samples)
        if shift_target == 1:
            x1 = rng.normal(loc=shift, size=num_samples)
        else:
            x1 = 0.8 * x0 + e1
        e2 = rng.normal(scale=0.4, size=num_samples)
        if shift_target == 2:
            x2 = rng.normal(loc=shift, size=num_samples)
        else:
            x2 = 0.6 * x1 + e2
        return np.column_stack([x0, x1, x2])

    data_dict = {
        "obs": (sample_env(0), set(), "obs"),
        "do_x0": (sample_env(1, shift_target=0), {0}, "hard"),
        "do_x1": (sample_env(2, shift_target=1), {1}, "hard"),
        "do_x2": (sample_env(3, shift_target=2), {2}, "hard"),
    }

    model = PartitionDagModelIvn()
    model.fit(data_dict, assume="gaussian")

    assert set(model.dag.nodes) == {(0,), (1,), (2,)}
    expected_edges = {((0,), (1,)), ((1,), (2,))}
    assert set(model.dag.edges) == expected_edges


def test_fit_oracle_extensive():
    digraph = nx.binomial_graph(100, 0.5, directed=True)
    true_order = list(rng(0).permutation(range(100)))
    s = np.argsort(true_order)

    to_rm = [(u, v) for u, v in digraph.edges if s[u] > s[v]]
    digraph.remove_edges_from(to_rm)
    true_adj = {k: [kk for kk in v.keys()] for k, v in digraph.adj.items()}

    model = PartitionDagModelOracle()
    model.fit(true_order, true_adj)


def test_intervention():
    seed = 0
    num_nodes = 20
    num_intervs = 5
    density = 0.1
    deg = density * (num_nodes - 1)

    weights = dag_avg_deg(
        num_nodes, deg, w_min=0.5, w_max=2, return_ordering=False, random_state=seed
    )

    edge_idcs = np.flatnonzero(weights)
    to_neg = edge_idcs[rng(seed).choice((True, False), len(edge_idcs))]
    weights[np.unravel_index(to_neg, (num_nodes, num_nodes))] *= -1

    model = LGANM(weights, means=(-2, 2), variances=(0.5, 2), random_state=seed)
    targets = intervention_targets(num_nodes, num_intervs, 1, random_state=seed)

    obs_dataset = model.sample(1000)
    # mu = obs_dataset.mean(0)
    # var = obs_dataset.std(0)
    # obs_dataset -= mu
    # obs_dataset /= var
    interv_datasets = {}
    for idx, target in enumerate(targets):
        target_nodes = set(target)
        sample = model.sample(1000, shift_interventions={target[0]: (2, 1)})
        interv_datasets[idx] = (sample, target_nodes, "soft")
    # interv_datasets = {k: v - mu for k, v in interv_datasets.items()}
    # interv_datasets = {k: v / var for k, v in interv_datasets.items()}

    data_dict = {"obs": (obs_dataset, set(), "obs")}
    data_dict.update(interv_datasets)
    model = PartitionDagModelIvn()
    model.fit(data_dict)
