from repare.repare import PartitionDagModel


def test_fit():
    true_order = [3, 1, 2, 4]
    true_adj = {1: [2], 3: [2], 2: [4]}

    model = PartitionDagModel()
    model.fit(true_order, true_adj)
