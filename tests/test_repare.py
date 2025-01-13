import networkx as nx
import numpy as np
from numpy.random import default_rng as rng

# from repare.cdci import causal_score
from repare.repare import PartitionDagModel
from sempler import LGANM

# from sklearn.decomposition import PCA


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

    return


# def test_cdci():
#     """quick start idea for refine test:
#     - refine should only return u,v such that u->v, v->u, or u _|_ v
#     - CDCI (along with independence test in the third case) determines whether one of the three cases applies
#     - can generalize CDCI random vectors? or otherwise use PCA to project down
#     """
#     seed = 0

#     # {0->1}->2
#     case_1 = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]]).astype(float)
#     num_edges = int(case_1.sum())
#     weights = rng(seed).uniform(-2, -0.5, num_edges)
#     weights[rng(seed).choice((True, False), num_edges)] *= -1
#     case_1[case_1.astype(bool)] = weights
#     case_1 = LGANM(case_1, means=(-2, 2), variances=(0.5, 2), random_state=seed)

#     dataset = case_1.sample(1000)
#     dataset -= dataset.mean(0)
#     dataset /= dataset.std(0)
#     part_1 = dataset[:, :2]
#     part_2 = dataset[:, 2]

#     # assert part_1 -> part_2
#     pca = PCA(n_components=1)
#     part_1 = pca.fit_transform(part_1).flatten()
#     case_1 = causal_score("CTV", part_1, part_2)
#     for meth in "CCS CHD CKL CKM CTV".split():
#         causal_score(meth, raw[:, 0], raw[:, 1])

#     # 0->{1<-2}
#     case_2 = np.array([[0, 1, 1], [0, 0, 0], [0, 1, 0]])
#     num_edges = int(case_2.sum())
#     weights = rng(seed).uniform(-2, -0.5, num_edges)
#     weights[rng(seed).choice((True, False), num_edges)] *= -1
#     case_2[case_2.astype(bool)] = weights
#     case_2 = LGANM(case_2, means=(-2, 2), variances=(0, 2), random_state=seed)

#     dataset = case_2.sample(1000)
#     dataset -= dataset.mean(0)
#     dataset /= dataset.std(0)
#     part_1 = dataset[:, :2]
#     part_2 = dataset[:, 2]

#     # assert part_1 /-> part_2
#     pca = PCA(n_components=1)
#     part_1 = pca.fit_transform(part_1).flatten()

#     # 0->{1<-{2,3,4}}
#     case_3 = np.array(
#         [
#             [0, 1, 1, 1, 1],
#             [0, 0, 0, 0, 0],
#             [0, 1, 0, 0, 0],
#             [0, 0, 1, 0, 0],
#             [0, 0, 0, 1, 0],
#         ]
#     )
#     num_edges = int(case_3.sum())
#     weights = rng(seed).uniform(-2, -0.5, num_edges)
#     weights[rng(seed).choice((True, False), num_edges)] *= -1
#     case_3[case_3.astype(bool)] = weights
#     case_3 = LGANM(case_3, means=(-2, 2), variances=(0, 2), random_state=seed)

#     dataset = case_3.sample(1000)
#     dataset -= dataset.mean(0)
#     dataset /= dataset.std(0)
#     part_1 = dataset[:, :2]
#     part_2 = dataset[:, 2:]

#     # assert part_1 /-> part_2
#     pca = PCA(n_components=1)
#     part_1 = pca.fit_transform(part_1).flatten()
#     pca = PCA(n_components=1)
#     part_2 = pca.fit_transform(part_2).flatten()

#     # doesn't seem to work :(
