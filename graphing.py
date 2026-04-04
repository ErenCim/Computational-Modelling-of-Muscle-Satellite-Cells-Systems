from __future__ import annotations
from pathlib import Path
from random import sample
from typing import Sequence
from scipy.interpolate import make_interp_spline
from scipy.interpolate import pchip_interpolate, interp1d
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

CSV_PATH        = Path("/Users/WangGroup_UofT/Desktop/Coding Projects/Thesis/initial_cond_density.csv")
OUT_SAMPLES_CSV = Path("/Users/WangGroup_UofT/Desktop/Coding Projects/Thesis/irregular_samples.csv")
DATA_CSV_PATH   = Path("/Users/WangGroup_UofT/Desktop/Coding Projects/Thesis/data_collection_final_combined.csv")
SS_CSV = Path("/Users/WangGroup_UofT/Desktop/Coding Projects/Thesis/steady_state_samples.csv")
# Can set to 0 to skip plotting, currently will only plot the first one
PLOT_EVERY = 10
PLOT_VALUES = [2]
DONT_PLOT = []#[0]

# Random Number Generator for irregular sampling times
RNG = np.random.default_rng(0)

"""
Currently observed states are all of the available states (N, M_d, etc.) but can select a partial list
If it is not necessary to graph all of the states. Sequence just specifies that this will be a sequnce of strings. 
"""
OBS_STATES: Sequence[str] = tuple(STATE_ORDER)

# Plotting helper function
def maybe_plot_dense(y0_vec: np.ndarray, theta_log: torch.Tensor, sample_id: int, 
                     density: float, num_pts: int = 400, color='blue', label=None) -> None:
    
    # Calculate which "Sample Group" this belongs to (0, 1, 2, 3...)
    group_id = sample_id // 4

    if PLOT_EVERY <= 0 or ((group_id % PLOT_EVERY) != 0 and group_id not in PLOT_VALUES) or sample_id in DONT_PLOT:
        return

    t_plot = np.linspace(0.0, T_MAX, num_pts, dtype=np.float64)
    y0_t = torch.tensor(y0_vec, dtype=torch.float64)
    tt_t = torch.tensor(t_plot, dtype=torch.float64)

    with torch.no_grad():
        Y = integrate(y0_t, tt_t, theta_log).cpu().numpy().T
    
    training_samples = pd.read_csv(DATA_CSV_PATH)

    for i, lab in enumerate(STATE_ORDER):
        fig_idx = (group_id * 10) + i
        plt.figure(fig_idx, figsize=(8, 6))
        
        plt.plot(t_plot, Y[i], label=label, color=color, linewidth=2, alpha=0.8)

        # Plot red dots ONLY for the 500 density baseline for THIS specific group
        if density == 500.0:
            filtered_data = training_samples[training_samples['sample_id'] == group_id]
            
            x = filtered_data['t'].values
            y = filtered_data[lab].values

            plt.scatter(x, y, color='red', s=30, zorder=5, label="Collected Data" if label else "")
            for xi, yi in zip(x, y):
                plt.annotate(f'{yi:.2f}', (xi, yi), textcoords="offset points", 
                             xytext=(0, 3), ha='center', fontsize=8, 
                             color='darkred', fontweight='bold')

        plt.xlabel("Time (days)")
        plt.ylabel(lab)
        plt.title(f"{lab} - Sample Group {group_id}")
        plt.grid(alpha=0.25)

def plot_total_stem_cells(y0_vec: np.ndarray, theta_log: torch.Tensor, sample_id: int, 
                          density: float, num_pts: int = 400, color='blue', label=None) -> None:
    
    group_id = sample_id // 4
    
    if PLOT_EVERY <= 0 or ((group_id % PLOT_EVERY) != 0 and group_id not in PLOT_VALUES) or sample_id in DONT_PLOT:
        return

    t_plot = np.linspace(0.0, T_MAX, num_pts, dtype=np.float64)
    y0_t = torch.tensor(y0_vec, dtype=torch.float64)
    tt_t = torch.tensor(t_plot, dtype=torch.float64)

    with torch.no_grad():
        Y = integrate(y0_t, tt_t, theta_log).cpu().numpy().T
    
    idx_psc = STATE_ORDER.index('PSC')
    idx_qsc = STATE_ORDER.index('QSC')
    idx_asc = STATE_ORDER.index('ASC')
    total_sc_model = Y[idx_psc] + Y[idx_qsc] + Y[idx_asc]

    plt.figure(1000 + group_id, figsize=(8, 6))
    
    plt.plot(t_plot, total_sc_model, label=label, color=color, linewidth=2.5, alpha=0.9)

    plt.xlabel("Time (days)")
    plt.ylabel("Total Count (PSC + QSC + ASC)")
    plt.title(f"Stem Cell Pool Dynamics - Sample Group {group_id}")
    plt.grid(True, which='both', linestyle='--', alpha=0.3)


if __name__ == "__main__":
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Initial-conditions CSV not found at '{CSV_PATH}'.")

    # A tensor with the optimized parameter values
    phi_prior = make_phi_prior(PARAMS_OPT)
    
    # Read initial conditions from the scaled CSV
    df_y0 = pd.read_csv(CSV_PATH)

    missing = [c for c in STATE_ORDER if c not in df_y0.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Color mapping for the different densities
    DENSITY_COLORS = {2000.0: 'gold', 1500.0: 'olive', 500.0: 'gray', 200.0: 'black'}

    # 1. Run the loop to add ALL lines to the figures
    for sample_id, row in df_y0.iterrows():
        y0_vec = row[STATE_ORDER].to_numpy(dtype=np.float64)
        current_density = row.get('Density', 500.0)
        line_color = DENSITY_COLORS.get(current_density, 'blue')
        
        # We only pass a label for the first group of 4 to keep the legend clean
        line_label = f"Density {int(current_density)}"

        #maybe_plot_dense
        plot_total_stem_cells(
            y0_vec=y0_vec, 
            theta_log=phi_prior, 
            sample_id=int(sample_id), 
            density=float(current_density),
            color=line_color, 
            label=line_label
        )

    # 2. Apply legend and tight_layout to ALL open figures before showing
    for fig_num in plt.get_fignums():
        plt.figure(fig_num)
        plt.legend(loc='best', fontsize='small')
        plt.tight_layout()

    plt.show()