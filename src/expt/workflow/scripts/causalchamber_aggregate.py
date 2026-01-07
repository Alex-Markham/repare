#!/usr/bin/env python3
"""Aggregate all method results (grouped+ungrouped RePaRe + GIES/GnIES/UT-IGSP)."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    # Load RePaRe metrics CSVs (for grid_metrics output only)
    grouped_df = pd.read_csv(snakemake.input.grouped_metrics)

    # Load RePaRe params (for method_metrics and summary)
    with open(snakemake.input.grouped_score_params, "r") as f:
        grouped_score_params = json.load(f)
    with open(snakemake.input.grouped_oracle_params, "r") as f:
        grouped_oracle_params = json.load(f)
    with open(snakemake.input.ungrouped_score_params, "r") as f:
        ungrouped_score_params = json.load(f)
    with open(snakemake.input.ungrouped_oracle_params, "r") as f:
        ungrouped_oracle_params = json.load(f)

    # Load other methods
    with open(snakemake.input.gies, "r") as f:
        gies_metrics = json.load(f)
    with open(snakemake.input.gnies, "r") as f:
        gnies_metrics = json.load(f)

    ut_df = pd.read_csv(snakemake.input.ut)

    # Build method_metrics
    method_records = []

    # Grouped RePaRe
    for label, params in [
        ("score", grouped_score_params),
        ("oracle", grouped_oracle_params),
    ]:
        method_records.append(
            {
                "method": "RePaRe_grouped",
                "selection": label,
                "metric_type": "partition",
                "ari": params["ari"],
                "precision": params["precision"],
                "recall": params["recall"],
                "f1": params["f1"],
                "runtime_sec": params["fit_time"],
                "score": params["score"],
                "alpha": params["alpha"],
                "beta": params["beta"],
            }
        )

    # Ungrouped RePaRe
    for label, params in [
        ("score", ungrouped_score_params),
        ("oracle", ungrouped_oracle_params),
    ]:
        method_records.append(
            {
                "method": "RePaRe_ungrouped",
                "selection": label,
                "metric_type": "partition",
                "ari": params["ari"],
                "precision": params["precision"],
                "recall": params["recall"],
                "f1": params["f1"],
                "runtime_sec": params["fit_time"],
                "score": params["score"],
                "alpha": params["alpha"],
                "beta": params["beta"],
            }
        )

    # GIES
    method_records.append(
        {
            "method": "GIES",
            "selection": "score",
            "metric_type": "edges",
            "ari": np.nan,
            "precision": gies_metrics["precision"],
            "recall": gies_metrics["recall"],
            "f1": gies_metrics["f1"],
            "runtime_sec": gies_metrics["runtime_sec"],
            "score": gies_metrics["score"],
            "alpha": np.nan,
            "beta": np.nan,
        }
    )

    # GnIES
    method_records.append(
        {
            "method": "GnIES",
            "selection": "score",
            "metric_type": "skeleton",
            "ari": np.nan,
            "precision": gnies_metrics["precision"],
            "recall": gnies_metrics["recall"],
            "f1": gnies_metrics["f1"],
            "runtime_sec": gnies_metrics["runtime_sec"],
            "score": gnies_metrics["score"],
            "alpha": np.nan,
            "beta": np.nan,
        }
    )

    # UT-IGSP score/oracle
    for label in ["score", "oracle"]:
        if label == "score":
            row = ut_df.loc[ut_df["bic"].idxmin()]
        else:
            row = ut_df.loc[(ut_df["f1"] * ut_df["precision"]).idxmax()]

        method_records.append(
            {
                "method": "UT-IGSP",
                "selection": label,
                "metric_type": "edges",
                "ari": np.nan,
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "runtime_sec": row["runtime_sec"],
                "score": row.get("bic", np.nan),
                "alpha": np.nan,
                "beta": np.nan,
                "alpha_ci": row["alpha_ci"],
                "alpha_inv": row["alpha_inv"],
            }
        )

    method_df = pd.DataFrame(method_records)
    method_df.to_csv(snakemake.output.method_metrics, index=False)

    # Grid metrics = grouped RePaRe
    grouped_df.to_csv(snakemake.output.grid_metrics, index=False)

    # Copy grouped score DAG as best
    shutil.copy(snakemake.input.grouped_score_dag, snakemake.output.dag)

    # Build detailed summary
    summary_lines = ["Target mode: default\n"]

    summary_lines.append("Grouped RePaRe (score-selected) hyperparameters:")
    summary_lines.append(
        json.dumps(
            {
                k: v
                for k, v in grouped_score_params.items()
                if k not in ["parts", "edges"]
            },
            indent=2,
            default=float,
        )
    )
    summary_lines.append("Partition nodes (score-selected):")
    for idx, labels in grouped_score_params["parts"]:
        summary_lines.append(f" Node {idx}: {labels}")
    summary_lines.append("Edges (u -> v):")
    for u, v in grouped_score_params["edges"]:
        summary_lines.append(f" {u} -> {v}")
    summary_lines.append("")

    summary_lines.append("Grouped RePaRe (oracle-selected) hyperparameters:")
    summary_lines.append(
        json.dumps(
            {
                k: v
                for k, v in grouped_oracle_params.items()
                if k not in ["parts", "edges"]
            },
            indent=2,
            default=float,
        )
    )
    summary_lines.append("Partition nodes (oracle-selected):")
    for idx, labels in grouped_oracle_params["parts"]:
        summary_lines.append(f" Node {idx}: {labels}")
    summary_lines.append("Edges (u -> v):")
    for u, v in grouped_oracle_params["edges"]:
        summary_lines.append(f" {u} -> {v}")
    summary_lines.append("")

    summary_lines.append("Ungrouped RePaRe (score-selected):")
    summary_lines.append(
        json.dumps(
            {
                k: v
                for k, v in ungrouped_score_params.items()
                if k not in ["parts", "edges"]
            },
            indent=2,
            default=float,
        )
    )
    summary_lines.append(" Nodes:")
    for idx, labels in ungrouped_score_params["parts"]:
        summary_lines.append(f" Node {idx}: {labels}")
    summary_lines.append(" Edges (u -> v):")
    for u, v in ungrouped_score_params["edges"]:
        summary_lines.append(f" {u} -> {v}")
    summary_lines.append("")

    summary_lines.append("Ungrouped RePaRe (oracle-selected):")
    summary_lines.append(
        json.dumps(
            {
                k: v
                for k, v in ungrouped_oracle_params.items()
                if k not in ["parts", "edges"]
            },
            indent=2,
            default=float,
        )
    )
    summary_lines.append(" Nodes:")
    for idx, labels in ungrouped_oracle_params["parts"]:
        summary_lines.append(f" Node {idx}: {labels}")
    summary_lines.append(" Edges (u -> v):")
    for u, v in ungrouped_oracle_params["edges"]:
        summary_lines.append(f" {u} -> {v}")
    summary_lines.append("")

    summary_lines.append("GIES:")
    summary_lines.append(f" Score: {gies_metrics['score']:.4f}")
    summary_lines.append(
        f" Precision/Recall/F1: {gies_metrics['precision']:.4f} / {gies_metrics['recall']:.4f} / {gies_metrics['f1']:.4f}"
    )
    summary_lines.append(f" Runtime (s): {gies_metrics['runtime_sec']:.2f}")
    summary_lines.append("")

    summary_lines.append("GnIES:")
    summary_lines.append(f" Score: {gnies_metrics['score']:.4f}")
    summary_lines.append(
        f" Estimated targets: {sorted(gnies_metrics.get('estimated_targets', []))}"
    )
    summary_lines.append(
        f" Skeleton precision/recall/F1: {gnies_metrics['precision']:.4f} / {gnies_metrics['recall']:.4f} / {gnies_metrics['f1']:.4f}"
    )
    summary_lines.append(f" Runtime (s): {gnies_metrics['runtime_sec']:.2f}")
    summary_lines.append("")

    ut_score = ut_df.loc[ut_df["bic"].idxmin()]
    ut_oracle = ut_df.loc[(ut_df["f1"] * ut_df["precision"]).idxmax()]

    summary_lines.append("UT-IGSP (score-selected):")
    summary_lines.append(
        json.dumps(
            {
                "alpha_ci": float(ut_score["alpha_ci"]),
                "alpha_inv": float(ut_score["alpha_inv"]),
                "bic": float(ut_score["bic"]),
            },
            indent=2,
        )
    )
    summary_lines.append("")

    summary_lines.append("UT-IGSP (oracle-selected):")
    summary_lines.append(
        json.dumps(
            {
                "alpha_ci": float(ut_oracle["alpha_ci"]),
                "alpha_inv": float(ut_oracle["alpha_inv"]),
                "bic": float(ut_oracle["bic"]),
            },
            indent=2,
        )
    )

    with open(snakemake.output.dag_summary, "w") as f:
        f.write("\n".join(summary_lines))

    Path(snakemake.output.grid_dir).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
