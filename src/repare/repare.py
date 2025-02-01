import networkx as nx
import numpy as np
from scipy.stats import kstest


class PartitionDagModelOracle(object):
    """A probabilistic graphical model in the form of a directed acyclic graph over parts of a partiton."""

    def __init__(self, rng=np.random.default_rng(0)) -> None:
        self.dag = nx.DiGraph()
        self.rng = rng

    def fit(self, order, adj) -> object:
        self.order = order
        self.adj = adj

        # always use lexicograph order for node partitions
        self.dag.add_node(tuple(sorted(order)))  # trivial coarsening

        recurse = len(order) - 1
        while recurse:
            recurse -= 1
            self._recurse()
        return self

    def _refine(self):
        refinable_nodes = [node for node in self.dag.nodes if len(node) > 1]
        to_refine_idx = self.rng.choice(len(refinable_nodes))
        to_refine = tuple(refinable_nodes[to_refine_idx])
        u_len = self.rng.choice(range(1, len(to_refine)))  # rest go to v
        u = []
        for el in self.order:
            if el in to_refine:
                u.append(el)
            if len(u) == u_len:
                break
        u = tuple(sorted(u))
        v = tuple((el for el in to_refine if el not in u))
        return to_refine, u, v

    def _is_adj(self, pa, ch):
        for el_pa in pa:
            for el_ch in ch:
                if el_ch in self.adj[el_pa]:
                    return True
        return False

    def _recurse(self):
        to_refine, u, v = self._refine()

        self.dag.add_node(u)
        self.dag.add_node(v)

        for pa in self.dag.predecessors(to_refine):
            for ch in (u, v):
                if self._is_adj(pa, ch):
                    self.dag.add_edge(pa, ch)
        for pa in (u, v):
            for ch in self.dag.successors(to_refine):
                if self._is_adj(pa, ch):
                    self.dag.add_edge(pa, ch)
        if self._is_adj(u, v):
            self.dag.add_edge(u, v)
        self.dag.remove_node(to_refine)


class PartitionDagModelIvn(PartitionDagModelOracle):
    def __init__(rng=np.random.default_rng(0)) -> None:
        super.__init__(rng)

    def fit(self, data_dict, alpha=0.05):
        obs = data_dict.pop("obs", None)  # raise error if obs is None
        ks_results = {
            idx: kstest(obs, post_ivn).pvalue < alpha
            for idx, post_ivn in data_dict.items()
        }  # <alpha means reject the null, so 0 -> 1 (in PO, not in causal dag)

        ivn_biadj = {k: np.flatnonzero(v) for k, v in ks_results.items()}

        # recurse

    def _refine():
        to_refine, u, v = None
        return to_refine, u, v
