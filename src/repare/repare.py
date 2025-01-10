import networkx as nx
import numpy as np


class PartitionDagModel(object):
    """A probabilistic graphical model in the form of a directed acyclic graph over parts of a partiton."""

    def __init__(self, rng=np.random.default_rng(0)) -> None:
        self.dag = nx.DiGraph()
        self.rng = rng

    def fit(self, order, adj) -> object:
        # always use lexicograph order for node partitions
        self.dag.add_node(tuple(sorted(order)))  # trivial coarsening

        def _is_adj(pa, ch):
            for el_pa in pa:
                for el_ch in ch:
                    if el_ch in adj[el_pa]:
                        return True
            return False

        recurse = len(order) - 1
        while recurse:
            recurse -= 1

            to_refine, u, v = self._refine(order)

            self.dag.add_node(u)
            self.dag.add_node(v)
            for pa in self.dag.predecessors(to_refine):
                for ch in (u, v):
                    if _is_adj(pa, ch):
                        self.dag.add_edge(pa, ch)
            for pa in (u, v):
                for ch in self.dag.successors(to_refine):
                    if _is_adj(pa, ch):
                        self.dag.add_edge(pa, ch)
            if _is_adj(u, v):
                self.dag.add_edge(u, v)
            self.dag.remove_node(to_refine)

        # recursively select a partition to split
        #     split it according to order
        #     add edges according to adj
        return self

    def _refine(self, order):
        refinable_nodes = [node for node in self.dag.nodes if len(node) > 1]
        to_refine = tuple(self.rng.choice(refinable_nodes))
        u_len = self.rng.choice(range(1, len(to_refine)))  # rest go to v
        u = tuple(
            sorted(
                [
                    el
                    for size, el in enumerate(order)
                    if (el in to_refine) and (size < u_len)
                ]
            )
        )
        v = tuple((el for el in to_refine if el not in u))
        return to_refine, u, v
