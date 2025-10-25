# muscle_fit_torch_smooth.py
from __future__ import annotations
from pathlib import Path
from typing import Sequence, List, Tuple
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torchdiffeq import odeint  # native differentiable solver (no SciPy wrapper)

dtype = torch.float64            # double precision helps with stiff-ish systems
device = torch.device("cpu")     # change to "cuda" if you have a GPU

# Output values and parameteres that we are trying to fit our data to
state_order: Sequence[str] = ["N","N_d","M_d","M","M1","M2","QSC","ASC","M_c","M_n"]
param_order: List[str] = [
    "c_NMd","c_Nin","c_Nout","c_M1Nd",
    "c_Min","c_MM1","c_Mout","c_M1out",
    "c_M1M2","c_M2inhib","c_M2out",
    "c_QSCN","c_QSCMd","c_ASCM2","c_ASCpro",
    "c_ASCdiff","c_ASCout","c_Mcout","c_Mcfusion",
    "c_Mdout","c_QSCself","c_Mnself","QSCmax"
]

# Paths for intput and output data
DATA_DIR = Path("/Users/erencimentepe/Desktop/VSCode Projects/Thesis")
SAMPLES_CSV = DATA_DIR / "irregular_samples.csv"
Y0_CSV      = DATA_DIR / "training_data_initial_conditions.csv"
OUT_CSV     = DATA_DIR / "learned_params_torch.csv"
OUT_NPY     = DATA_DIR / "learned_params_torch.npy"
OUT_JSON    = DATA_DIR / "learned_params_torch.json"

# Observation / solver
OBS_STATES: Sequence[str] = tuple(state_order)
T_MAX = 7.0
METHOD = "dopri5"      # native RK solver (better gradients than SciPy wrapper)
RTOL, ATOL = 1e-6, 1e-9

# Optimization
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

INIT_MODE = "jitter"   # "jitter" or "random"
JITTER_SD = 0.05
RAND_WIDTH = 0.35      # ± in log-space around prior if INIT_MODE="random"
N_ITERS = 5000
LR = 1e-3            # slightly gentler for stable early descent
L2_LAMBDA = 1e-4       # prior penalty in log-space
CLIP_NORM = 100.0       # gradient clipping

# Plot quick before/after on the FIRST sample only (set False to skip)
PLOT_FIRST_SAMPLE_COMPARISON = True

# Optimized parameteres to be used from the paper
params_opt = dict(
    c_NMd=1.18e-05, c_Nin=1.38e+00, c_Nout=3.10e+00, c_M1Nd=9.17e-5,
    c_Min=2.81e+01, c_MM1=5.81e-05, c_Mout=2.31e+01, c_M1out=4.29e-02,
    c_M1M2=5.48e+02, c_M2inhib=4.58e-12, c_M2out=1.18e-1,
    c_QSCN=5.22e-06, c_QSCMd=2.09e-04, c_ASCM2=8.38e-03, c_ASCpro=4.28e-05,
    c_ASCdiff=7.24e-03, c_ASCout=1.55e-04, c_Mcout=3.73e-06, c_Mcfusion=4.01e+00,
    c_Mdout=9.81e-04, c_QSCself=9.63e-02, c_Mnself=9.39e-05,
    QSCmax=2700.0
)
# Converting the parameters to NumPy array
phi_prior_np = np.log(np.array([params_opt[k] for k in param_order], dtype=np.float64))
# Converting array to a tensor, so that it can be used in the gradient descent algorithm
phi_prior = torch.tensor(phi_prior_np, dtype=dtype, device=device)

# Using softmax instead of relu (the positive function)
# For differentiability, but biologically it will have the same desired effects
# We can switch back to relu, if everything is running smoothly
def softpos(x: torch.Tensor, beta: float = 50.0) -> torch.Tensor:
    """Smooth ReLU; higher beta -> sharper, but still differentiable."""
    return F.softplus(x, beta=beta)

# Creating a tensor from the output values, with index numbers (ex: "N": 0)
idx_state = {s: i for i, s in enumerate(state_order)}
obs_idx = torch.tensor([idx_state[s] for s in OBS_STATES], dtype=torch.long, device=device)

class MuscleODE(nn.Module):
    def forward(self, t: torch.Tensor, y: torch.Tensor, theta_log: torch.Tensor) -> torch.Tensor:
        """
        y: (10,) state vector 
        theta_log: (22,) log-parameters (exp to enforce positivity)
        returns dy/dt: (10,)
        """
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

        # smooth nonnegativity where you had pos function isntead
        dQSCdt = -c_QSCN*QSC*N - c_QSCMd*QSC*M_d + c_ASCM2*ASC*M2 - c_QSCself*QSC*softpos(QSC - QSCmax)
        dASCdt =  c_QSCN*QSC*N + c_QSCMd*QSC*M_d - c_ASCM2*ASC*M2 + c_ASCpro*ASC*M1 - c_ASCdiff*ASC*M2 - c_ASCout*ASC
        dM_cdt = c_ASCdiff*ASC*M2 - c_Mcfusion*M_c - c_Mcout*M_c
        dMn_dt = c_Mcfusion*M_c - c_Mnself*M_n*softpos(M_n - 3000.0)

        return torch.stack([dN_dt, dNd_dt, dMd_dt, dM_dt, dM1_dt, dM2_dt, dQSCdt, dASCdt, dM_cdt, dMn_dt])

# Loading the irregular samples data from the csv file and converting it to pytorch objects
def load_batches(samples_csv: Path, y0_csv: Path) -> Tuple[List, torch.Tensor, int]:
    samples_df = pd.read_csv(samples_csv)
    y0_table   = pd.read_csv(y0_csv)

    # Per-state weights (1 / max^2), like your NumPy version
    y_max = {s: max(1.0, samples_df[s].max()) for s in OBS_STATES}
    w_np = np.array([1.0 / (y_max[s]**2) for s in OBS_STATES], dtype=np.float64)
    w = torch.tensor(w_np, dtype=dtype, device=device)

    batches = []
    total_points = 0
    for sid, df in samples_df.groupby("sample_id"):
        t = df["t"].to_numpy(np.float64)
        # ensure strictly increasing (drop duplicates if any)
        t = np.unique(t)
        t_obs = torch.tensor(t, dtype=dtype, device=device)
        Y_obs = torch.tensor(df.loc[df["t"].isin(t), list(OBS_STATES)].to_numpy(np.float64),
                             dtype=dtype, device=device)  # (T, S)
        # T = number of (unique) timepoints for this sample.
        # S = number of observed states (OBS_STATES order matters and must match your model’s output order for observed components).
        y0_vec = torch.tensor(y0_table.loc[int(sid), state_order].to_numpy(np.float64),
                              dtype=dtype, device=device)  # (10,)
        batches.append((int(sid), y0_vec, t_obs, Y_obs))
        total_points += Y_obs.numel()
    return batches, w, total_points

# This is the ODE solver, use it at each step of the gradient descent
def simulate_batch(ode: MuscleODE, theta_log: torch.Tensor, y0_vec: torch.Tensor, t_obs: torch.Tensor) -> torch.Tensor:
    """
    Integrate at requested times; return (T, S) for OBS_STATES.
    """
    # ODE solver, solving the differential equations and updating the parameters
    y_traj = odeint(
        func=lambda t, y: ode(t, y, theta_log),
        y0=y0_vec,
        t=t_obs,
        method=METHOD,
        rtol=RTOL,
        atol=ATOL,
    )  # (T, 10)
    y_traj = torch.where(y_traj >= 0, y_traj, 0.01 * y_traj)
    return y_traj[:, obs_idx]  # (T, S)

# Weighted MSE loss function
def total_loss(ode, theta_log, batches, w, npts):
    w_sse  = torch.tensor(0.0, dtype=dtype, device=device)  # The weighted sum of squared errors - Numerator of MSE
    w_mass = torch.tensor(0.0, dtype=dtype, device=device)  # The total weight mass (normalization factor) - Denominator of MSE

    for _, y0_vec, t_obs, Y_obs in batches:
        # Simulating the graphs at each point
        Y_hat = simulate_batch(ode, theta_log, y0_vec, t_obs)
        # Calculating error
        resid = Y_hat - Y_obs
        # Makes sure tht the shapes of the error matches the weight vector so the weights can be applied to the error
        w_exp = w if w.shape == resid.shape else w.expand_as(resid)
        # Squared error calculation
        w_sse  += torch.sum(w_exp * resid**2)
        # Adding it all together
        w_mass += resid.shape[0] * torch.sum(w)

    # Avoid division by 0
    data = w_sse / (w_mass + 1e-12)
    # Regularization term to avoid overfitting
    reg  = L2_LAMBDA * torch.sum((theta_log - phi_prior)**2)
    return data + reg

# Creates the deviance from the optimal parameters so we can re-optimize them based on the generated training data
def init_theta_log(init_mode: str) -> torch.Tensor:
    if init_mode == "jitter":
        noise = torch.randn_like(phi_prior) * JITTER_SD
        return (phi_prior + noise).clone().detach().requires_grad_(True)
    elif init_mode == "random":
        # Uniform random distribution between -1 and 1
        u = (torch.rand_like(phi_prior) * 2.0 - 1.0) * RAND_WIDTH  # per-parameter ± RAND_WIDTH
        # Ensures that the parameters are trainable and detached from gradients
        return (phi_prior + u).clone().detach().requires_grad_(True)

# This is where we run gradient descent and train based on the number of iterations
def main():
    # Sanity check to see if the training data exists
    if not SAMPLES_CSV.exists() or not Y0_CSV.exists():
        raise FileNotFoundError("Missing inputs. Run the generator first.")
    
    # Initialize the data and the ODE model
    batches, w, npts = load_batches(SAMPLES_CSV, Y0_CSV)
    ode = MuscleODE().to(device=device, dtype=dtype)

    # Initialize trainable log-parameters
    theta_log = init_theta_log(INIT_MODE)  # returns a requires_grad=True tensor in log-space
    # Adam is GD with momentum and adaptive learning
    opt = torch.optim.Adam([theta_log], lr=LR)

    # Initializing the best loss from the training process
    best_loss = float('inf')
    best_phi  = theta_log.detach().clone()  # keep a copy in case first step worsens

    # Training loop
    for it in range(1, N_ITERS + 1):
        # Clear old gradients on parameters
        opt.zero_grad(set_to_none=True)
        # Forward pass, compute the total loss by simulating the ODE and computing the weighted MSE with the current parameters 
        loss = total_loss(ode, theta_log, batches, w, npts)
        # Backpropogation, computes partial derivative of the loss wrt trainable parameters, theta_log
        loss.backward()

        # Clipping the gradients to prevent blowing up gradients
        preclip = theta_log.grad.norm().item()
        nn.utils.clip_grad_norm_([theta_log], max_norm=CLIP_NORM)
        postclip = theta_log.grad.norm().item()

        # Optimizer updates the parameters
        opt.step()

        # Track best params
        with torch.no_grad():
            li = loss.item()
            # Keep track of best parameters and update them
            if li < best_loss:
                best_loss = li
                best_phi  = theta_log.detach().clone()

        # Output the error, what the gradient was before and after clipping
        if it % 10 == 0 or it == 1:
            print(f"iter {it:3d}  loss={loss.item():.6e}  |grad| pre={preclip:.3e} post={postclip:.3e}")

        # Reducing the learning rate at different iterations numbers
        if it in {300, 900}:
            for pg in opt.param_groups:
                pg["lr"] *= 0.5

    # Display results
    print("\nBest loss:", best_loss)
    # Convert from log space back to regular values
    theta_hat = torch.exp(best_phi).cpu().numpy()

    # Saving the best parameters/weights that were learned
    pd.DataFrame({"name": param_order, "value": theta_hat}).to_csv(OUT_CSV, index=False)
    np.save(OUT_NPY, theta_hat)
    with open(OUT_JSON, "w") as f:
        json.dump({k: float(v) for k, v in zip(param_order, theta_hat)}, f, indent=2)
    print(f"Saved:\n  {OUT_CSV}\n  {OUT_NPY}\n  {OUT_JSON}")

    # Plot the comparison graphs for how the optimized parameters fits the data vs the trained parameters fit the data
    if PLOT_FIRST_SAMPLE_COMPARISON and len(batches) > 0:
        sid0, y0_vec, t_obs, Y_obs = batches[0]
        with torch.no_grad():
            Y_prior = simulate_batch(ode, phi_prior, y0_vec, t_obs).cpu().numpy()
            Y_fit   = simulate_batch(ode, best_phi,  y0_vec, t_obs).cpu().numpy()

        t_np = t_obs.cpu().numpy()
        for j, s in enumerate(OBS_STATES):
            plt.figure(figsize=(5.4, 3.6))
            plt.scatter(t_np, Y_obs[:, j].cpu().numpy(), s=15, label="data")
            plt.plot(t_np, Y_prior[:, j], linestyle="--", label="prior")
            plt.plot(t_np, Y_fit[:, j], label="fit")
            plt.xlabel("time (days)")
            plt.ylabel(s)
            plt.title(f"Sample {sid0} — {s}")
            plt.legend()
            plt.tight_layout()
            plt.show()

# Specify the order in which the parameters should appear
# theta_ref_np = np.array([params_opt[k] for k in param_order], dtype=np.float64)
# theta_ref    = torch.tensor(theta_ref_np, dtype=dtype, device=device)

# Duplicate the phi_prior, optimized parameters from the paper in log space
# phi_ref = phi_prior.clone()

if __name__ == "__main__":
    main()
