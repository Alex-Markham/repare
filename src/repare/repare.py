import networkx as nx


class PartitionDagModel(object):
    """A probabilistic graphical model in the form of a directed acyclic graph over parts of a partiton."""

    def __init__(self) -> None:
        self.dag = nx.DiGraph()

    def fit(self, order, adj) -> object:
        # always use lexicograph order for node partitions
        self.dag.add_node(sorted(order))  # trivial coarsening

        # recursively select a partition to split
        #     split it according to order
        #     add edges according to adj
        return self
