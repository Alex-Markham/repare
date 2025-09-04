import pickle

import pandas as pd


def synthesize_sachs_data():
    input_file = snakemake.input[0]  # Fitted DRFNet model pickle file
    output_file = snakemake.output[0]  # Output DataFrame pickle file
    samp_size = int(snakemake.wildcards["samp_size"])
    seed = int(snakemake.wildcards["seed"])

    with open(input_file, "rb") as f:
        drfnet = pickle.load(f)

    # monkey patch, since sempler implicitly requires rpy2~=3.4.1, which implicitly requires pandas<2.0, which fails to install
    if not hasattr(pd.DataFrame, "iteritems"):
        pd.DataFrame.iteritems = pd.DataFrame.items

    # Sample synthetic data: list of arrays, one per environment
    synthetic_samples = drfnet.sample(n=samp_size, random_state=seed)

    # Convert each array to DataFrame and add 'INT' column indicating intervention index
    dfs = []
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
    targets = [labels[i] for i in [0, 2, 4, 7, 8, 9]]
    for i, arr in enumerate(synthetic_samples):
        df = pd.DataFrame(arr, columns=labels)
        df["INT"] = targets[i]  # intervention index added to match real data format
        dfs.append(df)

    # Concatenate all environments into one DataFrame matching the real data style
    synthetic_df = pd.concat(dfs, ignore_index=True)

    # Save as pandas pickle to match expected input format
    synthetic_df.to_pickle(output_file)


if __name__ == "__main__":
    synthesize_sachs_data()
