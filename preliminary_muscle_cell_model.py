from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Sequence
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchdiffeq import odeint  # differentiable solver

def softpos(x: torch.Tensor, beta: float = 40.0) -> torch.Tensor:
    # match your training smooth-positivity; tweak beta if desired
    return F.softplus(x, beta=beta)

state_order: Sequence[str] = ["N","N_d","M_d","M","M1","M2","QSC","ASC","M_c","M_n"]

params_opt = dict(
    c_NMd=1.18e-05, c_Nin=1.38e+00, c_Nout=3.10e+00, c_M1Nd=9.17e-5,
    c_Min=2.81e+01, c_MM1=5.81e-05, c_Mout=2.31e+01, c_M1out=4.29e-02,
    c_M1M2=5.48e+02, c_M2inhib=4.58e-12, c_M2out=1.18e-1,
    c_QSCN=5.22e-06, c_QSCMd=2.09e-04, c_ASCM2=8.38e-03, c_ASCpro=4.28e-05,
    c_ASCdiff=7.24e-03, c_ASCout=1.55e-04, c_Mcout=3.73e-06, c_Mcfusion=4.01e+00,
    c_Mdout=9.81e-04, c_QSCself=9.63e-02, c_Mnself=9.39e-05,
    QSCmax=2700.0
)

param_order = [
    "c_NMd","c_Nin","c_Nout","c_M1Nd",
    "c_Min","c_MM1","c_Mout","c_M1out",
    "c_M1M2","c_M2inhib","c_M2out",
    "c_QSCN","c_QSCMd","c_ASCM2","c_ASCpro",
    "c_ASCdiff","c_ASCout","c_Mcout","c_Mcfusion",
    "c_Mdout","c_QSCself","c_Mnself","QSCmax"
]

# CSV Files with the initial conditions
CSV_PATH = Path("/Users/erencimentepe/Desktop/VSCode Projects/Thesis/training_data_initial_conditions.csv")
OUT_SAMPLES_CSV = Path("/Users/erencimentepe/Desktop/VSCode Projects/Thesis/irregular_samples.csv")

# Solver settings — match training
T_MAX = 7.0
METHOD = "dopri8"          # closer to LSODA behavior than dopri5; differentiable
RTOL, ATOL = 1e-7, 1e-10
PLOT_EVERY = 20

# Irregular sampling (unchanged)
rng = np.random.default_rng(0)
IRREG_MIN_PTS, IRREG_MAX_PTS = 10, 25
IRREG_MIN_DT, IRREG_MAX_DT   = 0.1, 0.8
RANDOM_START_FRAC            = 0.6

# torch dtype/device
dtype = torch.float64
device = torch.device("cpu")

phi_prior = torch.tensor(
    np.log(np.array([params_opt[k] for k in param_order], dtype=np.float64)),
    dtype=dtype, device=device
)

# index helpers (unchanged)
idx = {s: i for i, s in enumerate(state_order)}
OBS_STATES: Sequence[str] = tuple(state_order)
obs_idx = [idx[s] for s in OBS_STATES]

def dydt(t: torch.Tensor, y: torch.Tensor, theta_log: torch.Tensor) -> torch.Tensor:
    theta = torch.exp(theta_log)
    (c_NMd, c_Nin, c_Nout, c_M1Nd,
     c_Min, c_MM1, c_Mout, c_M1out,
     c_M1M2, c_M2inhib, c_M2out,
     c_QSCN, c_QSCMd, c_ASCM2, c_ASCpro,
     c_ASCdiff, c_ASCout, c_Mcout, c_Mcfusion,
     c_Mdout, c_QSCself, c_Mnself, QSCmax) = torch.unbind(theta)

    N, N_d, M_d, M, M1, M2, QSC, ASC, M_c, M_n = torch.unbind(y)

    dN_dt  = c_Nin*M_d - c_Nout*N - c_NMd*N*M_d
    dMd_dt = -c_Mdout*M_d*M1 - c_NMd*M_d*N
    dNd_dt = c_NMd*N*M_d - c_M1Nd*N_d*M1
    dM_dt  = c_Min*N - c_MM1*M*(N_d + M_d) - c_Mout*M
    m1_to_m2 = (c_M1M2*M1) / (c_M2inhib + N_d + M_d)
    dM1_dt = c_MM1*M*(N_d + M_d) - c_M1out*M1 - m1_to_m2
    dM2_dt = m1_to_m2 - c_M2out*M2

    # smooth versions of your pos() terms
    dQSCdt = -c_QSCN*QSC*N - c_QSCMd*QSC*M_d + c_ASCM2*ASC*M2 - c_QSCself*QSC*softpos(QSC - QSCmax)
    dASCdt =  c_QSCN*QSC*N + c_QSCMd*QSC*M_d - c_ASCM2*ASC*M2 + c_ASCpro*ASC*M1 - c_ASCdiff*ASC*M2 - c_ASCout*ASC
    dM_cdt = c_ASCdiff*ASC*M2 - c_Mcfusion*M_c - c_Mcout*M_c
    dMn_dt = c_Mcfusion*M_c - c_Mnself*M_n*softpos(M_n - 3000.0)

    return torch.stack([dN_dt, dNd_dt, dMd_dt, dM_dt, dM1_dt, dM2_dt, dQSCdt, dASCdt, dM_cdt, dMn_dt])

# ---------- helpers (same signatures/behavior as your SciPy version) ----------
def irregular_times(t_max: float = T_MAX) -> np.ndarray:
    """
    Return strictly increasing times that ALWAYS include t=0,
    plus irregular times starting after a random t0 > 0.
    """
    n = int(rng.integers(IRREG_MIN_PTS, IRREG_MAX_PTS + 1))
    if n <= 1:
        return np.array([0.0], dtype=float)

    # start some time after 0 (keep your irregular vibe)
    t0 = float(rng.uniform(0.0, RANDOM_START_FRAC * t_max))
    gaps = rng.uniform(IRREG_MIN_DT, IRREG_MAX_DT, size=max(0, n - 2))
    ts = np.concatenate([[t0], t0 + np.cumsum(gaps)]) if gaps.size > 0 else np.array([t0], dtype=float)

    # clip, prepend 0.0, sort & dedupe
    ts = ts[(ts >= 0.0) & (ts <= t_max)]
    ts = np.unique(np.concatenate([[0.0], ts]))
    return ts

def solve_and_sample_torch(y0_vec: np.ndarray, t_obs: np.ndarray) -> np.ndarray:
    y0 = torch.tensor(y0_vec, dtype=dtype, device=device)
    t  = torch.tensor(t_obs, dtype=dtype, device=device)
    y_traj = odeint(lambda tt, yy: dydt(tt, yy, phi_prior),
                    y0, t, method=METHOD, rtol=RTOL, atol=ATOL)  # (T, 10)
    # Do NOT clamp here — keep generator == training forward model
    Y = y_traj[:, obs_idx].detach().cpu().numpy()   # (T, S)
    return Y

def maybe_plot_torch(y0_vec: np.ndarray, sample_id: int) -> None:
    if PLOT_EVERY <= 0 or (sample_id % PLOT_EVERY) != 0:
        return
    t_plot = np.linspace(0, T_MAX, 400)
    y0 = torch.tensor(y0_vec, dtype=dtype, device=device)
    t  = torch.tensor(t_plot, dtype=dtype, device=device)
    y_traj = odeint(lambda tt, yy: dydt(tt, yy, phi_prior),
                    y0, t, method=METHOD, rtol=RTOL, atol=ATOL)
    Y = y_traj.detach().cpu().numpy().T  # shape (states, times) to mimic your SciPy plot
    for i, lab in enumerate(state_order):
        plt.figure(figsize=(6, 4))
        plt.plot(t_plot, Y[i], label=lab)
        plt.xlabel("Time (days)")
        plt.ylabel(lab)
        plt.title(f"{lab} (sample {sample_id})")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()

# ---------- main (identical IO contract) ----------
if __name__ == "__main__":
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Initial-conditions CSV not found at '{CSV_PATH}'. "
            "Generate it first (including your baseline row if desired)."
        )

    df_y0 = pd.read_csv(CSV_PATH)
    missing = [c for c in state_order if c not in df_y0.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    all_rows = []
    for sample_id, row in df_y0.iterrows():
        y0_vec = row[state_order].to_numpy(dtype=float)

        # Optional plots
        maybe_plot_torch(y0_vec, int(sample_id))

        # Irregular sampling (same as before)
        t_obs = irregular_times(T_MAX)
        Y_obs = solve_and_sample_torch(y0_vec, t_obs)  # (T, S)

        for t, vec in zip(t_obs, Y_obs):
            rec = {"sample_id": int(sample_id), "t": float(t)}
            rec.update({s: float(v) for s, v in zip(OBS_STATES, vec)})
            all_rows.append(rec)

    samples_df = pd.DataFrame(all_rows)
    samples_df.to_csv(OUT_SAMPLES_CSV, index=False)
    print(f"[OK] Saved irregular samples -> {OUT_SAMPLES_CSV} (rows={len(samples_df)})")