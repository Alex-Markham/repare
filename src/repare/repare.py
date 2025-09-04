from collections import deque

import dcor
import networkx as nx
import numpy as np
from scipy.stats import kstest


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

    def fit(self, data_dict, alpha=0.05, mu=0.1, normalize=True):
        # pool and standardize data; set mu for self._is_adj()
        pooled_data = np.vstack(list(data_dict.values()))
        if normalize:
            pooled_data -= pooled_data.mean(axis=0)
            pooled_data /= pooled_data.std(axis=0)
        self.pooled_data = pooled_data
        self.mu = mu
        obs = data_dict.pop("obs", None)
        if obs is None:
            raise ValueError("Observed data 'obs' key not found in data_dict")

        ks_results = {
            idx: kstest(obs, post_ivn).pvalue < alpha
            for idx, post_ivn in data_dict.items()
        }
        self.partition = _get_totally_ordered_partition(ks_results)
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
        # Compute distance covariance between sets pa and ch using dcor
        xy_indices = list(pa.union(ch))
        xy_data = self.pooled_data[:, xy_indices]  # shape (samples x variables)
        # Separate data columns for pa and ch
        pa_mask = [idx in pa for idx in xy_indices]
        ch_mask = [idx in ch for idx in xy_indices]
        x = xy_data[:, pa_mask]
        y = xy_data[:, ch_mask]
        # Compute the empirical distance covariance test statistic
        # test_result = dcor.independence.distance_covariance_test(
        #     x, y, num_resamples=100, random_state=0
        # )
        test_result = dcor.independence.distance_correlation_t_test(x, y)
        return test_result.pvalue > self.mu


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
