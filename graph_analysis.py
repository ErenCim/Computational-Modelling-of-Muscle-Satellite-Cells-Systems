import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compare_and_plot_params(csv_path: str, params_opt: dict, top_n: int = None):
    """
    Compare learned parameters to reference ones and plot relative differences as a bar chart.
    """
    df = pd.read_csv(csv_path)
    ref_df = pd.DataFrame({
        "name": list(params_opt.keys()),
        "ref_value": list(params_opt.values())
    })

    merged = pd.merge(df, ref_df, on="name", how="inner")
    merged["abs_diff"] = np.abs(merged["value"] - merged["ref_value"])
    merged["rel_diff_%"] = 100 * merged["abs_diff"] / (np.abs(merged["ref_value"]) + 1e-12)
    merged.sort_values("rel_diff_%", ascending=False, inplace=True)

    print(merged[["name", "value", "ref_value", "rel_diff_%"]].to_string(index=False,
          formatters={"value": "{:.6g}".format, "ref_value": "{:.6g}".format,
                      "rel_diff_%": "{:.2f}".format}))
    print(f"\nAverage relative difference: {merged['rel_diff_%'].mean():.3f}%")

    # pick top N if specified
    if top_n:
        subset = merged.head(top_n)
    else:
        subset = merged

    # --- bar chart only ---
    plt.figure(figsize=(10, 5))
    plt.bar(subset["name"], subset["rel_diff_%"], color="teal", alpha=0.85)
    plt.xticks(rotation=60, ha="right", fontsize=9)
    plt.ylabel("Relative difference (%)")
    plt.title(f"Parameter deviations (top {top_n if top_n else len(subset)})")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return merged

if __name__ == "__main__":
    # Suppose this is your CSV path and reference parameters
    csv_path = "/Users/erencimentepe/Desktop/VSCode Projects/Thesis/learned_params_torch.csv"

    # Your known optimal (paper) parameters from the script
    params_opt = dict(
        c_NMd=1.18e-05, c_Nin=1.38, c_Nout=3.10, c_M1Nd=9.17e-05,
        c_Min=2.81e+01, c_MM1=5.81e-05, c_Mout=2.31e+01, c_M1out=4.29e-02,
        c_M1M2=5.48e+02, c_M2inhib=4.58e-12, c_M2out=1.18e-1,
        c_QSCN=5.22e-06, c_QSCMd=2.09e-04, c_ASCM2=8.38e-03, c_ASCpro=4.28e-05,
        c_ASCdiff=7.24e-03, c_ASCout=1.55e-04, c_Mcout=3.73e-06, c_Mcfusion=4.01e+00,
        c_Mdout=9.81e-04, c_QSCself=9.63e-02, c_Mnself=9.39e-05, QSCmax=2700.0
    )

    compare_and_plot_params(csv_path, params_opt)