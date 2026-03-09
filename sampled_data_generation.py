from __future__ import annotations
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Get the necessary functions from the shared model.py file
from model import (
    STATE_ORDER, PARAMS_OPT, make_phi_prior, T_MAX,
    irregular_times, simulate_on_times, integrate,
    set_device_dtype, steady_state_from_y0,
)

# Basic config, the tensors will exist in the cpu and the gradients will be calculated to a float64 level precision
set_device_dtype("cpu", torch.float64)

CSV_PATH        = Path("/Users/erencimentepe/Desktop/VSCode Projects/Thesis/initial_conditions.csv")
OUT_SAMPLES_CSV = Path("/Users/erencimentepe/Desktop/VSCode Projects/Thesis/irregular_samples.csv")
SS_CSV = Path("/Users/erencimentepe/Desktop/VSCode Projects/Thesis/steady_state_samples.csv")
# Can set to 0 to skip plotting, currently will only plot the first one
PLOT_EVERY = 10
PLOT_VALUES = [3, 18, 19]

# Random Number Generator for irregular sampling times
RNG = np.random.default_rng(0)

"""
Currently observed states are all of the available states (N, M_d, etc.) but can select a partial list
If it is not necessary to graph all of the states. Sequence just specifies that this will be a sequnce of strings. 
"""
OBS_STATES: Sequence[str] = tuple(STATE_ORDER)

# Plotting helper function
def maybe_plot_dense(y0_vec: np.ndarray, theta_log: torch.Tensor, sample_id: int, num_pts: int = 400) -> None:
    # Ensure we are plotting every X PLOT_EVERY index 
    if PLOT_EVERY <= 0 or ((sample_id % PLOT_EVERY) != 0 and sample_id not in PLOT_VALUES):
        return
    # Plotting 400 samples (from num_pts) which ensure a smoother curve
    # T_MAX is usualyl 7, which represents the duration of the plot
    t_plot = np.linspace(0.0, T_MAX, num_pts, dtype=np.float64)

    # This is the initial conditions and the evenly distributed 400 time points that we are going to plot as tensors
    y0_t = torch.tensor(y0_vec, dtype=torch.float64)
    tt_t = torch.tensor(t_plot, dtype=torch.float64)

    # No need to store gradients since we are only going to be plotting
    with torch.no_grad():
        # Runs the ODE solver to generate the plots with the given initial conditions
        Y = integrate(y0_t, tt_t, theta_log).cpu().numpy().T
    for i, lab in enumerate[str](STATE_ORDER):
        plt.figure(figsize=(6, 4))
        plt.plot(t_plot, Y[i], label=lab)
        plt.xlabel("Time (days)"); plt.ylabel(lab)
        plt.title(f"{lab} (sample {sample_id})"); plt.grid(alpha=0.25)
        plt.tight_layout(); plt.show()

def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Initial-conditions CSV not found at '{CSV_PATH}'. "
            "Generate it first (including your baseline row if desired)."
        )

    # A tensor with the optimizied parameter values
    phi_prior = make_phi_prior(PARAMS_OPT)

    # Read initial conditions that were generated from the CSV file
    df_y0 = pd.read_csv(CSV_PATH)
    # Sanity check to ensure that the correct state parameters are included in the initial conditions
    missing = [c for c in STATE_ORDER if c not in df_y0.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    all_rows = []
    steady_rows = []
    for sample_id, row in df_y0.iterrows():
        # row[STATE_ORDER] extacts the columns values from row in the exact order of STATE_ORDER
        
        
        y0_vec = row[STATE_ORDER].to_numpy(dtype=np.float64)
        # Plotting of the generated curves, only plot the ones every PLOT_EVERY index
        maybe_plot_dense(y0_vec, phi_prior, int(sample_id))

        # Get irregular sampling times
        t_obs = irregular_times(T_MAX, RNG) 
        # Then simulate with those irregular times                    
        Y_obs = simulate_on_times(y0_vec, t_obs, phi_prior)      

        # Zip fairs the time and the state vectors (np array) together
        # Then, for each time and the corresponding state vector (with 10 states)
        # Create a row for the csv file, where there is the sample_id, time, and all 10 states
        for tt, vec in zip(t_obs, Y_obs):
            rec = {"sample_id": int(sample_id), "t": float(tt)}
            rec.update({s: float(v) for s, v in zip(OBS_STATES, vec)})
            all_rows.append(rec)
        """
        y0_vec = row[STATE_ORDER].to_numpy(dtype=np.float64)

        y_star, resid_max, ok = steady_state_from_y0(y0_vec, phi_prior)

        print(f"sample {sample_id}: ok={ok} resid_max={resid_max:.2e} y*={y_star}")

        rec = {"sample_id": int(sample_id), "resid_max": resid_max, "steady_ok": int(ok)}
        rec.update({s: float(v) for s, v in zip(STATE_ORDER, y_star)})
        steady_rows.append(rec)
        """

    # Saving the generated irregular conditions to a csv file
    samples_df = pd.DataFrame(steady_rows)
    samples_df.to_csv(OUT_SAMPLES_CSV, index=False)
    print(f"[OK] Saved irregular samples -> {OUT_SAMPLES_CSV} (rows={len(samples_df)})")

if __name__ == "__main__":
    main()
