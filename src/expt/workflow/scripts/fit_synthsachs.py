import pickle

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from repare.repare import PartitionDagModelIvn


def fit_synthsachs():
    input_file = snakemake.input[0]
    output_model = snakemake.output.model
    output_dag = snakemake.output.dag
    reference = snakemake.params.reference
    alpha = float(snakemake.params.alpha)
    beta = float(snakemake.params.beta)
    assume_param = snakemake.params.assume
    assume = None if assume_param in (None, "None", "none") else assume_param
    refine_param = getattr(snakemake.params, "refine_test", "ks")
    refine_test = ("ks" if refine_param is None else str(refine_param)).lower()

    # Load DataFrame with 'INT' column marking intervention indices
    df = pd.read_pickle(input_file)

    # Build dictionary of data arrays keyed by intervention index as string
    ivn_idcs = df["INT"].unique()
    data_dict = {}
    for ivn in ivn_idcs:
        array = df[df["INT"] == ivn].drop("INT", axis=1).to_numpy()
        if ivn == reference:
            data_dict["obs"] = (array, set(), "obs")
        else:
            data_dict[str(ivn)] = (array, {int(ivn)}, "hard")

    # Fit the PartitionDagModelIvn
    model = PartitionDagModelIvn()
    model.fit(data_dict, alpha, beta, assume, refine_test=refine_test)

    # Plot learned DAG
    fig = plt.figure()
    ax = fig.add_subplot()
    layout = nx.kamada_kawai_layout
    nx.draw(model.dag, pos=layout(model.dag), ax=ax, with_labels=True)
    nx.draw_networkx_labels(model.dag, pos=layout(model.dag))
    fig.savefig(output_dag)
    plt.close(fig)

    # Save the fitted model
    with open(output_model, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    fit_synthsachs()
