import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.random import default_rng as rng
from repare.repare import (
    PartitionDagModelIvn,
    PartitionDagModelOracle,
    _get_totally_ordered_partition,
)
from scipy.stats import kstest
from sempler import LGANM
from sempler.generators import dag_avg_deg, intervention_targets
from sklearn.metrics import adjusted_rand_score
from tigramite.independence_tests.cmiknn import CMIknn


def test_fit_oracle_basic():
    true_order = [3, 1, 2, 4]
    true_adj = {1: [2], 3: [2], 2: [4]}

    model = PartitionDagModelOracle()
    model.fit(true_order, true_adj)

    assert len(model.dag.edges) == 3
    for u in true_adj.keys():
        for v in true_adj[u]:
            assert model.dag.has_edge((u,), (v,))


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
    interv_datasets = {
        idx: model.sample(1000, shift_interventions={target[0]: (2, 1)})
        for idx, target in enumerate(targets)
    }
    # interv_datasets = {k: v - mu for k, v in interv_datasets.items()}
    # interv_datasets = {k: v / var for k, v in interv_datasets.items()}

    data_dict = {"obs": obs_dataset} | interv_datasets
    model = PartitionDagModelIvn()
    model.fit(data_dict)


def test_cmiknn():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 100))
    y = x[0] + x[1]  # rng.standard_normal((1, 100))
    xy = np.vstack((x, y))
    tester = CMIknn()
    test_stat_val = tester.get_dependence_measure(xy, np.array([0, 0, 1]))
    # p_val = tester.get_shuffle_significance(xy, np.array([0, 0, 1]), test_stat_val)
    return test_stat_val > 0.1


def eval(num_nodes, density, samp_size, num_intervs, seed, alpha, mu):
    # generate data
    deg = density * (num_nodes - 1)
    weights = dag_avg_deg(
        num_nodes, deg, w_min=0.5, w_max=2, return_ordering=False, random_state=seed
    )

    edge_idcs = np.flatnonzero(weights)
    to_neg = edge_idcs[rng(seed).choice((True, False), len(edge_idcs))]
    weights[np.unravel_index(to_neg, (num_nodes, num_nodes))] *= -1

    model = LGANM(weights, means=(-2, 2), variances=(0.5, 2), random_state=seed)
    targets = intervention_targets(num_nodes, num_intervs, 1, random_state=seed)

    obs_dataset = model.sample(samp_size)
    interv_datasets = {
        idx: model.sample(samp_size, shift_interventions={target[0]: (2, 1)})
        for idx, target in enumerate(targets)
    }
    data_dict = {"obs": obs_dataset} | interv_datasets

    # estimate model
    model = PartitionDagModelIvn()
    model.fit(data_dict, alpha, mu)

    # extract ground truth
    true_dag = nx.DiGraph(weights.astype(bool))
    target_des_masks = {
        idx: np.zeros(len(true_dag), bool) for idx in range(len(targets))
    }
    # target_des = [list(true_dag.successors(target[0])) + target for target in targets]
    for idx in target_des_masks:
        target = targets[idx][0]
        des = list(true_dag.successors(target)) + [target]
        target_des_masks[idx][des] = True
    true_partition = _get_totally_ordered_partition(target_des_masks)
    true_labels = np.zeros(len(true_dag))
    for label, part in enumerate(true_partition):
        true_labels[list(part)] = label

    # evaluate
    est_labels = np.zeros(len(true_dag))
    for label, part in enumerate(model.dag.nodes):
        est_labels[list(part)] = label
    ar_index = adjusted_rand_score(true_labels, est_labels)

    def _is_adj(pa, ch):
        for atom in pa:
            for chatom in ch:
                if true_dag.has_edge(atom, chatom):
                    return True
        return False

    true_edge_est_partition = nx.create_empty_copy(model.dag)
    node_list = list(true_edge_est_partition.nodes)
    for idx, pa in enumerate(node_list[:-1]):
        for ch in node_list[idx + 1 :]:
            if _is_adj(pa, ch):
                true_edge_est_partition.add_edge(pa, ch)

    true_positive = sum(
        (1 for edge in model.dag.edges if edge in true_edge_est_partition.edges)
    )
    try:
        precision = true_positive / len(model.dag.edges)
    except ZeroDivisionError:
        precision = 1
    try:
        recall = true_positive / len(true_edge_est_partition.edges)
    except ZeroDivisionError:
        recall = 1
    f_score = 2 * (precision * recall) / (precision + recall)

    return ar_index, f_score
    # consider adding some eval related to downstream task (e.g., causal inference)


def initial_eval_results():
    seeds = range(10)
    num_nodes = 10
    samp_size = 1000
    num_intervs = 3
    densities = [0.2, 0.4, 0.6, 0.8]
    alpha = 0.01
    mu = 0.1

    cols = [
        "seed",
        "density",
        "samp_size",
        "num_nodes",
        "num_intervs",
        "alpha",
        "mu",
        "alg",
        "fscore",
        "ari",
    ]
    results_df = pd.DataFrame(columns=cols)
    for seed in seeds:
        print(seed)
        for density in densities:
            ar_index, f_score = eval(
                num_nodes, density, samp_size, num_intervs, seed, alpha, mu
            )
            df_idx = len(results_df)
            results_df.loc[df_idx] = [
                seed,
                density,
                samp_size,
                num_nodes,
                num_intervs,
                alpha,
                mu,
                "repare",
                f_score,
                ar_index,
            ]
    results_df.to_csv("initial_results.csv")


def plot():
    results_df = pd.read_csv("initial_results.csv")

    sns.set_context("paper", font_scale=1.5)

    fplot = sns.catplot(
        data=results_df,
        x="density",
        y="fscore",
        hue="alg",
        kind="box",
        row="num_nodes",
        legend=False,
        palette="Set2",
    )
    fplot.set_titles("")
    fplot.set_ylabels("F-score ↑")
    fplot.tight_layout()
    fplot.figure.savefig("fscores.pdf")

    ariplot = sns.catplot(
        data=results_df,
        x="density",
        y="ari",
        hue="alg",
        kind="box",
        row="num_nodes",
        legend=False,
        palette="Set2",
    )
    ariplot.set_titles("")
    ariplot.set_ylabels("adjusted Rand index ↑")
    ariplot.tight_layout()
    ariplot.figure.savefig("ari.pdf")


# plot()


def sachs(alpha=0.05, mu=0.1, obs_idx=0):
    # downloaded from https://www.bnlearn.com/book-crc/code/sachs.interventional.txt.gz
    pooled = pd.read_csv("sachs.interventional.txt", sep=" ")

    # p38 (idx 9) is lowest ivn in causal order, so use that as obs reference
    # using ground truth from https://github.com/cmu-phil/example-causal-datasets/blob/main/real/sachs/ground.truth/sachs.2005.ground.truth.graph.txt

    # Split the DataFrame based on the last column
    ivn_idcs = pooled["INT"].unique()

    data_dict = {}
    for idx in ivn_idcs:
        data_dict[idx] = pooled[pooled["INT"] == idx].copy()
    data_dict["obs"] = data_dict.pop(obs_idx)
    data_dict = {k: v.drop("INT", axis=1) for k, v in data_dict.items()}

    # run repare
    model = PartitionDagModelIvn()
    model.fit(data_dict, alpha, mu, disc=True)

    # save dag fig
    fig = plt.figure()
    # layout = nx.circular_layout
    layout = nx.kamada_kawai_layout
    nx.draw(model.dag, pos=layout(model.dag), ax=fig.add_subplot())
    nx.draw_networkx_labels(model.dag, pos=layout(model.dag))
    fig.savefig(f"sach_dag_{obs_idx}.png")

    return model


def partition_sachs(alpha=0.01, obs_idx=9):
    pooled = pd.read_csv("sachs.interventional.txt", sep=" ")
    ivn_idcs = pooled["INT"].unique()
    data_dict = {}
    for idx in ivn_idcs:
        data_dict[idx] = pooled[pooled["INT"] == idx].copy()
    data_dict["obs"] = data_dict.pop(obs_idx)
    data_dict = {k: v.drop("INT", axis=1) for k, v in data_dict.items()}

    obs = data_dict.pop("obs", None)
    ks_results = {
        idx: kstest(obs, post_ivn).pvalue < alpha for idx, post_ivn in data_dict.items()
    }  # <alpha means reject the null, so 0 -> 1 (in PO, not in causal dag)
    print(ks_results)


def eval_sachs(model, obs_idx):
    true_edges = [
        ("raf", "mek"),
        ("pka", "akt"),
        ("pka", "erk"),
        ("pka", "jnk"),
        ("pka", "mek"),
        ("pka", "p38"),
        ("pka", "raf"),
        ("plc", "pip2"),
        ("plc", "pkc"),
        ("pip3", "akt"),
        ("pip3", "pip2"),
        ("pip3", "plc"),
        ("pip2", "pkc"),
        ("pkc", "jnk"),
        ("pkc", "mek"),
        ("pkc", "p38"),
        ("pkc", "pka"),
        ("pkc", "raf"),
        ("mek", "erk"),
        ("erk", "akt"),
    ]
    true_dag = nx.DiGraph()
    true_dag.add_edges_from(true_edges)
    labels = [
        "raf",
        "mek",
        "plc",
        "pip2",
        "pip3",
        "erk",
        "akt",
        "pka",
        "pkc",
        "p38",
        "jnk",
    ]
    label_dict = {label: idx for idx, label in enumerate(labels)}
    nx.relabel_nodes(true_dag, label_dict, copy=False)

    targets = [0, 2, 4, 7, 8, 9]
    targets.remove(obs_idx)

    # extract ground truth
    target_des_masks = {
        idx: np.zeros(len(true_dag), bool) for idx in range(len(targets))
    }
    # target_des = [list(true_dag.successors(target[0])) + target for target in targets]
    for idx in target_des_masks:
        target = targets[idx]
        des = list(true_dag.successors(target)) + [target]
        target_des_masks[idx][des] = True
    true_partition = _get_totally_ordered_partition(target_des_masks)
    true_labels = np.zeros(len(true_dag))
    for label, part in enumerate(true_partition):
        true_labels[list(part)] = label

    # evaluate
    est_labels = np.zeros(len(true_dag))
    for label, part in enumerate(model.dag.nodes):
        est_labels[list(part)] = label
    ar_index = adjusted_rand_score(true_labels, est_labels)

    def _is_adj(pa, ch):
        for atom in pa:
            for chatom in ch:
                if true_dag.has_edge(atom, chatom):
                    return True
        return False

    true_edge_est_partition = nx.create_empty_copy(model.dag)
    node_list = list(true_edge_est_partition.nodes)
    for idx, pa in enumerate(node_list[:-1]):
        for ch in node_list[idx + 1 :]:
            if _is_adj(pa, ch):
                true_edge_est_partition.add_edge(pa, ch)

    true_positive = sum(
        (1 for edge in model.dag.edges if edge in true_edge_est_partition.edges)
    )
    try:
        precision = true_positive / len(model.dag.edges)
    except ZeroDivisionError:
        precision = 1
    try:
        recall = true_positive / len(true_edge_est_partition.edges)
    except ZeroDivisionError:
        recall = 1
    f_score = 2 * (precision * recall) / (precision + recall)

    return ar_index, f_score


def grid_partition_sachs():
    for idx in [
        # 0,
        8,
        7,
        9,
        4,
        2,
    ]:
        model = sachs(alpha=0.05, mu=0.2, obs_idx=idx)
        print(
            f"\n\nobs is {idx}, resulting in {len(model.dag.nodes)} nodes and {len(model.dag.edges)} edges."  # with partition {model.dag.nodes} and edges {model.dag.edges}"
        )
        print(eval_sachs(model, idx))
