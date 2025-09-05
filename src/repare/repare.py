from collections import deque

import dcor
import networkx as nx
import numpy as np
from scipy.stats import chi2, chi2_contingency, kstest

from .utils import SimpleCanCorr


class PartitionDagModelOracle(object):
    """A probabilistic graphical model in the form of a directed acyclic graph over parts of a partition."""

    def __init__(self, rng=np.random.default_rng(0)) -> None:
        self.dag = nx.DiGraph()
        self.rng = rng

    def fit(self, order, adj) -> object:
        self.order = order
        self.adj = adj
        # always use lexicographical order for node partitions
        init_partition = [set(order)]  # trivial coarsening
        self.dag.add_node(tuple(sorted(order)))
        self.refinable = deque(init_partition)
        while len(self.refinable) > 0:
            self._recurse()
        return self

    def _refine(self):
        to_refine = self.refinable.popleft()
        u_len = self.rng.choice(range(1, len(to_refine)))  # rest go to v
        u = []
        for el in self.order:
            if el in to_refine:
                u.append(el)
            if len(u) == u_len:
                break
        u = set(u)
        v = set((el for el in to_refine if el not in u))
        if len(u) > 1:
            self.refinable.append(u)
        if len(v) > 1:
            self.refinable.append(v)
        return to_refine, u, v

    def _is_adj(self, pa, ch):
        for el_pa in pa:
            for el_ch in ch:
                if el_ch in self.adj[el_pa]:
                    return True
        return False

    def _recurse(self):
        to_refine, u, v = self._refine()
        if not u or not v:
            return
        self.dag.add_node(tuple(u))
        self.dag.add_node(tuple(v))
        for pa in self.dag.predecessors(tuple(to_refine)):
            for ch in (u, v):
                if self._is_adj(set(pa), ch):
                    self.dag.add_edge(pa, tuple(ch))
        for pa in (u, v):
            for ch in self.dag.successors(tuple(to_refine)):
                if self._is_adj(pa, set(ch)):
                    self.dag.add_edge(tuple(pa), ch)
        if self._is_adj(u, v):
            self.dag.add_edge(tuple(u), tuple(v))
        self.dag.remove_node(tuple(to_refine))


class PartitionDagModelIvn(PartitionDagModelOracle):
    def __init__(self, rng=np.random.default_rng(0)) -> None:
        super().__init__(rng)

    def fit(self, data_dict, alpha=0.05, beta=0.05, assume=None):
        self.assume = assume
        self.data_dict = data_dict
        self.beta = beta
        obs = data_dict.pop("obs")

        if self.assume is None:

            def p_val(post_ivn):
                res = kstest(obs, post_ivn)
                return res.pvalue

        # Gaussian LRT: vectorized over columns of post_ivn
        if self.assume == "gaussian":

            def p_val(post_ivn):
                num_feats = obs.shape[1]
                pvals = np.empty(num_feats)
                for j in range(num_feats):
                    x = obs[:, j]
                    y = post_ivn[:, j]

                    def ll(data):
                        m = data.size
                        mu = data.mean()
                        s2 = np.mean((data - mu) ** 2)
                        return -0.5 * m * (np.log(2 * np.pi * s2) + 1)

                    ll_sep = ll(x) + ll(y)
                    ll_null = ll(np.concatenate([x, y]))
                    LR = 2.0 * (ll_sep - ll_null)
                    pvals[j] = chi2.sf(LR, df=2)
                return pvals

        # Discrete chi2: vectorized over columns of post_ivn
        if self.assume == "discrete":

            def p_val(post_ivn):
                num_feats = obs.shape[1]
                pvals = np.empty(num_feats)
                for j in range(num_feats):
                    x = obs[:, j]
                    y = post_ivn[:, j]
                    vals = np.concatenate([x, y])
                    uniq, inv = np.unique(vals, return_inverse=True)
                    inv_x = inv[: x.size]
                    inv_y = inv[x.size :]
                    V = uniq.size
                    table = np.zeros((2, V), dtype=int)
                    for idx in inv_x:
                        table[0, idx] += 1
                    for idx in inv_y:
                        table[1, idx] += 1
                    _, pval, _, _ = chi2_contingency(table, correction=False)
                    pvals[j] = float(pval)
                return pvals

        results = {idx: p_val(post_ivn) < alpha for idx, post_ivn in data_dict.items()}
        self.partition = _get_totally_ordered_partition(results)
        init_partition = [set(range(obs.shape[1]))]  # trivial coarsening
        self.dag.add_node(tuple(range(obs.shape[1])))
        self.refinable = deque(init_partition)
        while len(self.refinable) > 0:
            self._recurse()
        return self

    def _refine(self):
        to_refine = self.refinable.pop()
        u = self.partition.popleft()
        v = to_refine - u
        if self.partition and v != self.partition[-1]:
            self.refinable.append(v)
        return to_refine, u, v

    def _is_adj(self, pa, ch):
        # Compute dependence between sets pa and ch
        xy_indices = list(pa.union(ch))

        # Separate data columns for pa and ch
        pa_mask = [idx in pa for idx in xy_indices]
        ch_mask = [idx in ch for idx in xy_indices]

        if self.assume is None:

            def p_val(x, y):
                test_result = dcor.independence.distance_correlation_t_test(x, y)
                return test_result.pvalue

        if self.assume == "gaussian":

            def p_val(x, y):
                # bug in statsmodels.multivariate.cancorr, so can't use the following
                # cc = CanCorr(y, x)
                # test_res = cc.corr_test()
                # return test_res.stats_mv.loc["Wilks' lambda", "Pr > F"]
                cca = SimpleCanCorr(x, y)
                result = cca.wilks_lambda_test()
                return result.loc[0, "Pr > F"]

        if self.assume == "discrete":
            """For paired samples X (n×p) and Y (n×q) where both are
            discrete (categorical), give a concise Python function
            that tests independence parametrically: encode
            multivariate categories as joint labels, build the
            contingency table, run scipy.stats.chi2_contingency
            (Pearson chi-square) to return the asymptotic p-value, and
            handle sparse tables by warning when expected counts < 5
            (or fall back to Fisher/exact/permutation). Include
            imports and a minimal example ."""
            p_val = 1
        p_vals = []
        for env in self.data_dict.values():
            xy_data = env[:, xy_indices]  # shape (samples x variables)
            x = xy_data[:, pa_mask]
            y = xy_data[:, ch_mask]
            if x.ndim == 1:
                x = x[:, None]
            if y.ndim == 1:
                y = y[:, None]
            p_vals.append(p_val(x, y))

        return max(p_vals) <= self.beta


def _get_totally_ordered_partition(ivn_biadj):
    num_atoms = len(next(iter(ivn_biadj.values())))
    partition = [list(range(num_atoms))]
    while len(ivn_biadj) > 0:
        idx, ivned = ivn_biadj.popitem()
        new_partition = []
        for part in partition:
            first_part = []
            second_part = []
            for atom in part:
                if not ivned[atom]:
                    first_part.append(atom)
                else:
                    second_part.append(atom)
            if len(first_part) > 0 and len(second_part) > 0:
                new_partition.append(first_part)
                new_partition.append(second_part)
            else:
                new_partition.append(part)
        partition = new_partition
    return deque((set(part) for part in partition))
