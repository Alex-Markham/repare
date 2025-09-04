import pickle

import networkx as nx
import pandas as pd
from sempler import DRFNet


def synthesize_ground_truth():
    data_file = snakemake.input[0]
    output_file = snakemake.output[0]

    # Define the Sachs ground truth edges and nodes, as you provided
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

    # Build directed graph
    true_dag = nx.DiGraph()
    true_dag.add_edges_from(true_edges)
    nx.relabel_nodes(true_dag, label_dict, copy=False)

    # Get adjacency matrix (NumPy array)
    adjacency = nx.to_numpy_array(true_dag, nodelist=range(len(labels)))

    # Load interventional data split by "INT" column
    pooled = pd.read_csv(data_file, sep=" ")
    ivn_idcs = pooled["INT"].unique()
    data_list = [
        pooled[pooled["INT"] == idx].drop("INT", axis=1).values for idx in ivn_idcs
    ]

    # monkey patch, since sempler implicitly requires rpy2~=3.4.1, which implicitly requires pandas<2.0, which fails to install
    if not hasattr(pd.DataFrame, "iteritems"):
        pd.DataFrame.iteritems = pd.DataFrame.items

    # Fit DRFNet with fixed true DAG structure on interventional datasets simultaneously
    network = DRFNet(adjacency, data_list)

    # Save the fitted DRFNet model
    with open(output_file, "wb") as f:
        pickle.dump(network, f)


if __name__ == "__main__":
    synthesize_ground_truth()
