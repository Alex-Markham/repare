import networkx as nx
import numpy as np


class PartitionDagModel(object):
    """A probabilistic graphical model in the form of a directed acyclic graph over parts of a partiton."""

    def __init__(self, rng=np.random.default_rng(0)) -> None:
        self.dag = nx.DiGraph()
        self.rng = rng

    def fit(self, order, adj) -> object:
        # always use lexicograph order for node partitions
        self.dag.add_node(sorted(order))  # trivial coarsening

        recurse = len(order)
        while recurse:
            recurse -= 1

            to_refine, u, v = self.refine(order)

            self.dag.add_node(u)
            self.dag.add_node(v)
            for pa in self.dag.predecessors(to_refine):
                for el in pa:
                    for el_u in u:
                        if el_u in adj[el]:
                            self.dag.add_edge(pa, u)
                            break
                    for el_v in v:
                        if el_v in adj[el]:
                            self.dag.add_edge(pa, v)
                            break
            for el_u in u:
                for el_v in v:
                    if el_v in adj[u]:
                        self.dag.add_edge(u, v)
                        break
            self.dag.remove_node(to_refine)

        # recursively select a partition to split
        #     split it according to order
        #     add edges according to adj
        return self

    def refine(self, order):
        refinable_nodes = [node for node in self.dag.nodes if len(node) > 1]
        to_refine = self.rng.choice(refinable_nodes)
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
