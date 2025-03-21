from collections import deque

import networkx as nx
import numpy as np
from scipy.stats import kstest
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.cmisymb import CMIsymb


class PartitionDagModelOracle(object):
    """A probabilistic graphical model in the form of a directed acyclic graph over parts of a partiton."""

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
    def __init__(rng=np.random.default_rng(0)) -> None:
        super().__init__(rng)

    def fit(self, data_dict, alpha=0.05, mu=0.1, disc=False):
        # pool and standardize data; set mu for self._is_adj()
        pooled_data = np.vstack(list(data_dict.values()))
        self.tester = CMIsymb() if disc else CMIknn()
        if not disc:
            pooled_data -= pooled_data.mean(0)
            pooled_data /= pooled_data.std(0)
        self.pooled_data = pooled_data
        self.mu = mu

        obs = data_dict.pop("obs", None)  # raise error if obs is None
        ks_results = {
            idx: kstest(obs, post_ivn).pvalue < alpha
            for idx, post_ivn in data_dict.items()
        }  # <alpha means reject the null, so 0 -> 1 (in PO, not in causal dag)
        self.partition = _get_totally_ordered_partition(ks_results)
        # total_order = list(chain(*partition))
        # partition_points = tuple(accumulate((len(part) for part in partition)))
        # print(total_order, partition_points)

        # always use lexicographical order for node partitions
        init_partition = [set(range(obs.shape[1]))]  # trivial coarsening
        self.dag.add_node(tuple(range(obs.shape[1])))
        self.refinable = deque(init_partition)
        while len(self.refinable) > 0:
            # print(self.refinable)
            self._recurse()
        return self

    def _refine(self):
        to_refine = self.refinable.pop()
        u = self.partition.popleft()
        v = to_refine - u
        if v != self.partition[-1]:
            self.refinable.append(v)
        return to_refine, u, v

    def _is_adj(self, pa, ch):
        if type(pa) is tuple or type(ch) is tuple:
            print(pa, ch)
        xy_idcs = list(pa.union(ch))
        xy_data = self.pooled_data.T[xy_idcs]
        xy_mask = np.array([atom in ch for atom in xy_idcs])
        test_stat_val = self.tester.get_dependence_measure(xy_data, xy_mask)
        # p_val = self.tester.get_shuffle_significance(xy_data, xy_mask, test_stat_val)
        # print(test_stat_val)
        return test_stat_val > self.mu
        # return p_val < self.mu

    # add explicit CI test
    # def _is_adj(self, pa, ch, cond):
    #     if type(pa) is tuple or type(ch) is tuple:
    #         print(pa, ch)
    #     xy_idcs = list(pa.union(ch))
    #     xy_data = self.pooled_data.T[xy_idcs]
    #     xy_mask = np.array([atom in ch for atom in xy_idcs])
    #     tester = CMIknn()
    #     test_stat_val = tester.get_dependence_measure(xy_data, xy_mask)
    #     # p_val = tester.get_shuffle_significance(xy_data, xy_mask, test_stat_val)
    #     # print(test_stat_val)
    #     return test_stat_val > self.mu
    #     # return p_val < 0.05

    # def _recurse(self):
    #     to_refine, u, v = self._refine()
    #     self.dag.add_node(tuple(u))
    #     self.dag.add_node(tuple(v))

    #     for pa in self.dag.predecessors(tuple(to_refine)):
    #         for ch in (u, v):
    #             if self._is_adj(set(pa), ch):
    #                 self.dag.add_edge(pa, tuple(ch))
    #     for pa in (u, v):
    #         for ch in self.dag.successors(tuple(to_refine)):
    #             if self._is_adj(pa, set(ch)):
    #                 self.dag.add_edge(tuple(pa), ch)
    #     if self._is_adj(u, v):
    #         self.dag.add_edge(tuple(u), tuple(v))
    #     self.dag.remove_node(tuple(to_refine))


def _get_totally_ordered_partition(ivn_biadj):
    num_atoms = len(ivn_biadj[0])
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
