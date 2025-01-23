import networkx as nx
import numpy as np
from numpy.random import default_rng as rng
from repare.repare import PartitionDagModel
from scipy.stats import kstest
from sempler import LGANM
from sempler.generators import dag_avg_deg, intervention_targets


def test_fit_basic():
    true_order = [3, 1, 2, 4]
    true_adj = {1: [2], 3: [2], 2: [4]}

    model = PartitionDagModel()
    model.fit(true_order, true_adj)

    assert len(model.dag.edges) == 3
    for u in true_adj.keys():
        for v in true_adj[u]:
            assert model.dag.has_edge((u,), (v,))


def test_fit_extensive():
    digraph = nx.binomial_graph(100, 0.5, directed=True)
    true_order = list(rng(0).permutation(range(100)))
    s = np.argsort(true_order)
    to_rm = [(u, v) for u, v in digraph.edges if s[u] > s[v]]
    digraph.remove_edges_from(to_rm)
    true_adj = {k: [kk for kk in v.keys()] for k, v in digraph.adj.items()}

    model = PartitionDagModel()
    model.fit(true_order, true_adj)


def test_intervention():
    seed = 0
    num_nodes = 20
    num_intervs = 5
    density = 0.3
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
    interv_datasets = {
        idx: model.sample(1000, shift_interventions={target[0]: (2, 1)})
        for idx, target in enumerate(targets)
    }

    alpha = 0.05

    ks_results = {
        idx: kstest(obs_dataset, interv_dataset).pvalue < alpha
        for idx, interv_dataset in interv_datasets.items()
    }  # <alpha means reject the null, so 0 -> 1 (in PO, not in causal dag)
