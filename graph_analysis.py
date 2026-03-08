# compare_params.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Pull truth from your shared module
from model import PARAMS_OPT, PARAM_ORDER

def compare_and_plot_params(csv_path: str, params_opt: dict | None = None, top_n: int | None = None):
    """
    Compare learned parameters (csv: columns ['name','value']) against reference ones
    and plot relative differences as a bar chart.
    """
    df = pd.read_csv(csv_path)

    # Use canonical reference from model unless caller overrides
    ref = PARAMS_OPT if params_opt is None else params_opt
    ref_df = pd.DataFrame({"name": list(ref.keys()), "ref_value": list(ref.values())})

    # Merge on name and warn if anything is missing/extra
    merged = pd.merge(df, ref_df, on="name", how="outer", indicator=True)
    missing_in_csv  = merged.loc[merged["_merge"] == "right_only", "name"].tolist()
    missing_in_ref  = merged.loc[merged["_merge"] == "left_only",  "name"].tolist()
    if missing_in_csv:
        print(f"[warn] Learned CSV missing params: {missing_in_csv}")
    if missing_in_ref:
        print(f"[warn] CSV has unknown params (not in reference): {missing_in_ref}")

    # Keep only rows present in both sides
    merged = merged.loc[merged["_merge"] == "both", ["name", "value", "ref_value"]]

    # Sort by canonical order if available
    order_pos = {k: i for i, k in enumerate(PARAM_ORDER)}
    merged["__ord"] = merged["name"].map(order_pos).fillna(1e9)
    merged.sort_values(["__ord"], inplace=True)
    merged.drop(columns="__ord", inplace=True)

    # Differences
    eps = 1e-12
    merged["abs_diff"]   = (merged["value"] - merged["ref_value"]).abs()
    merged["rel_diff_%"] = 100.0 * merged["abs_diff"] / (merged["ref_value"].abs() + eps)

    # Print a tidy table
    print(
        merged[["name", "value", "ref_value", "rel_diff_%"]]
        .to_string(index=False,
                   formatters={
                       "value": "{:.6g}".format,
                       "ref_value": "{:.6g}".format,
                       "rel_diff_%": "{:.2f}".format
                   })
    )
    print(f"\nAverage relative difference: {merged['rel_diff_%'].mean():.3f}%")

    # Top-N filter for the plot (largest deviations)
    plot_df = merged.sort_values("rel_diff_%", ascending=False)
    if top_n:
        plot_df = plot_df.head(top_n)

    # Bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(plot_df["name"], plot_df["rel_diff_%"], alpha=0.85)
    plt.xticks(rotation=60, ha="right", fontsize=9)
    plt.ylabel("Relative difference (%)")
    title_n = top_n if top_n else len(plot_df)
    plt.title(f"Parameter deviations (top {title_n})")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return merged

def plot_steady_state_heatmap(
    csv_path: str,
    states=("PSC", "QSC", "ASC", "SC_TAP", "Myo"),
    sort_by: str | None = "Myo",
    noise_tol: float = 1e-8,
    figsize=(8, 6),
    cmap="viridis",
):
    """
    Plot a normalized heatmap of steady states from a CSV.

    Parameters
    ----------
    csv_path : str
        Path to steady_state_samples.csv
    states : tuple
        State columns to include in the heatmap
    sort_by : str or None
        Column to sort rows by (e.g. "Myo"). Set to None to disable sorting.
    noise_tol : float
        Values with abs(x) < noise_tol are snapped to zero
    figsize : tuple
        Figure size
    cmap : str
        Matplotlib / seaborn colormap
    """

    # Load data
    df = pd.read_csv(csv_path)

    # Optional sorting (reveals structure very clearly)
    if sort_by is not None and sort_by in df.columns:
        df = df.sort_values(sort_by)

    # Extract states
    X = df[list(states)].copy()

    # Clean numerical noise
    X = X.clip(lower=0.0)
    X[X.abs() < noise_tol] = 0.0

    # Normalize per column (CRITICAL for interpretability)
    X_norm = (X - X.min()) / (X.max() - X.min() + 1e-12)

    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        X_norm,
        cmap=cmap,
        cbar_kws={"label": "Normalized value"},
        xticklabels=states,
        yticklabels=df["sample_id"] if "sample_id" in df.columns else False,
    )

    plt.xlabel("State")
    plt.ylabel("Seed (sample_id)")
    plt.title("Normalized steady states across initial conditions")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #csv_path = "/Users/erencimentepe/Desktop/VSCode Projects/Thesis/learned_params_torch.csv"
    #compare_and_plot_params(csv_path, top_n=None)
    plot_steady_state_heatmap(
        csv_path="/Users/erencimentepe/Desktop/VSCode Projects/Thesis/steady_state_samples.csv",
        sort_by=None#"Myo"
    )

