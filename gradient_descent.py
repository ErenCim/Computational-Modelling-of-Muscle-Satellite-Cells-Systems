from __future__ import annotations
from pathlib import Path
from typing import Sequence, List, Tuple
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn

from model import (
    STATE_ORDER, PARAM_ORDER, PARAMS_OPT,
    make_phi_prior, obs_index, integrate, load_batches,
    set_device_dtype, METHOD, RTOL, ATOL, TORCH_OPTS
)

# Basic config, the tensors will exist in the cpu and the gradients will be calculated to a float64 level precision
set_device_dtype("cpu", torch.float64)   # change to ("cuda", torch.float64) if you want GPU

# All of the paths that are used to get the training data and save results
DATA_DIR   = Path("/Users/erencimentepe/Desktop/VSCode Projects/Thesis")
SAMPLES_CSV = DATA_DIR / "irregular_samples.csv"
Y0_CSV      = DATA_DIR / "initial_conditions.csv"
OUT_CSV     = DATA_DIR / "learned_params_torch.csv"
OUT_NPY     = DATA_DIR / "learned_params_torch.npy"
OUT_JSON    = DATA_DIR / "learned_params_torch.json"

# ---------- solver (training-time) ----------
OBS_STATES: Sequence[str] = tuple(STATE_ORDER)  # you can switch to a subset later
T_MAX = 7.0  # (kept for reference; solver tolerances/method are imported)

# Included for reproducibility
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

# Gradient descent optimization parameters
INIT_MODE  = "jitter"                       # "jitter" or "random"
JITTER_SD  = 0.01                            # e^JITTER_SD addition, since the optimization parameters are in log space, so roughly a 1% difference
RAND_WIDTH = 0.35                           # If the random init mode is selected it will add random noise from +- 0.35 range
N_ITERS    = 500                           # Number of iterations
LR         = 1.5e-3                         # Learning rate
CLIP_NORM  = 300.0                          # Clipping that prevents exploding gradients having a huge impact
PLOT_FIRST_SAMPLE_COMPARISON = True         # Whether or not to produce the final comparison plots

# Can stop iterating if the change in error is less than 1e-3
MIN_IMPROVE = 1e-4 

# Optimized parameters in log space
phi_prior = make_phi_prior(PARAMS_OPT)

# Select observed states, currently the default is all of them. But in case in the future we want to observe specific ones
obs_idx = obs_index(OBS_STATES)  # consistent device + order

# Solves the ODE system and returns the predicted trajectories ONLY for the states you observe
def simulate_batch(theta_log: torch.Tensor, y0_vec: torch.Tensor, t_obs: torch.Tensor) -> torch.Tensor:
    # Calls the function for the ODE solver
    y_traj = integrate(y0_vec, t_obs, theta_log)
    # Returns only the states we are observing. More often than not we go with the default which is all of them
    return y_traj[:, obs_idx]

# Weighted MSE loss function
def total_loss(theta_log, batches, w):
    # Weighted MSE across all samples.
    w_sse  = torch.tensor(0.0, dtype=torch.get_default_dtype())
    # Sum of all weights, it is the normalization factor so we are calculating the mean
    w_mass = torch.tensor(0.0, dtype=torch.get_default_dtype())

    # Batches group things by sample id
    for _, y0_vec, t_obs, Y_obs in batches:
        Y_hat = simulate_batch(theta_log, y0_vec, t_obs)
        # Error term, the difference between the observed points when the batch is simulated in this iterations and the ones from the training data
        resid = Y_hat - Y_obs
        # Expand per-state weights so that it can be applied to all of the terms
        # w (S, ) is a state vector, meaning the weight for each state (eg. weight for M, N, M_d, etc.)
        w_exp = w if w.shape == resid.shape else w.expand_as(resid)
        # Square and add to the total error after multiplying it by its weight
        w_sse  += torch.sum(w_exp * resid**2)
        # Normalization step to calculate mean. Total amount of weights applied accross all timepoints and states.
        # resid.shape[0] is the number of time points for that batch
        # torch.sum(w) calculates the total weight accross all states
        w_mass += resid.shape[0] * torch.sum(w)
    
    # Ensure we are not dividing by 0
    data = w_sse / (w_mass + 1e-12)
    return data

# Model initialization basedon the type chosen
def init_theta_log(init_mode: str) -> torch.Tensor:
    if init_mode == "jitter":
        # Create a tensor in the same shape as phi_prior with random values from a normal distribution with mean 0 and std 1
        # Then multiply it by the jitter to scale the noise down
        noise = torch.randn_like(phi_prior) * JITTER_SD
        # Add the noise to the optimized parameters and make it trainable (can calculate gradients)
        return (phi_prior + noise).clone().detach().requires_grad_(True)
    elif init_mode == "random":
        # Mapping the random values between -1 and 1 than multiplying with the scaling factor for bigger range
        u = (torch.rand_like(phi_prior) * 2.0 - 1.0) * RAND_WIDTH
        return (phi_prior + u).clone().detach().requires_grad_(True)
    else:
        raise ValueError("INIT_MODE must be 'jitter' or 'random'.")

# Training
def main():
    # Sanity check
    if not SAMPLES_CSV.exists() or not Y0_CSV.exists():
        raise FileNotFoundError("Missing inputs. Run the generator first.")

    # Load dataframes and the batches
    samples_df = pd.read_csv(SAMPLES_CSV)
    y0_df      = pd.read_csv(Y0_CSV)
    batches, w = load_batches(samples_df, y0_df, OBS_STATES)

    theta_log = init_theta_log(INIT_MODE)
    # Initialize optimizer, Adam
    opt = torch.optim.Adam([theta_log], lr=LR)

    # Best loss and the parameters that gave the best loss
    best_loss = float('inf')
    best_phi  = theta_log.detach().clone()

    # Checking what the loss is before any gradients with the optimized parameters
    with torch.no_grad():
        loss_at_prior = total_loss(phi_prior, batches, w)
        print("loss_at_prior (training solver):", float(loss_at_prior))

    prev_loss = None

    for it in range(1, N_ITERS + 1):
        # Clear previous gradients
        opt.zero_grad(set_to_none=True)
        # Compute loss for the current parameters
        loss = total_loss(theta_log, batches, w)
        # Compute gradients
        loss.backward()

        # Computes magnitude of gradient before clipping - used for debugging
        preclip = theta_log.grad.norm().item()
        # Clipping the gradient if it is too big
        nn.utils.clip_grad_norm_([theta_log], max_norm=CLIP_NORM)
        # Computes magnitude of gradient after clipping - used for debugging
        postclip = theta_log.grad.norm().item()
        # Applies Adam optimizer update, where the parameters change
        opt.step()

        # Extracts the loss value from the tensor
        li = loss.item()
        # Store the best loss
        if li < best_loss:
            best_loss = li
            best_phi  = theta_log.detach().clone()

        # Early stop on small improvement
        if prev_loss is not None and abs(prev_loss - li) < MIN_IMPROVE:
            print(f"[early stop] |Δloss|={abs(prev_loss - li):.3e} < {MIN_IMPROVE:.3e}")
            break
        prev_loss = li

        # Output the gradient and the loss after every 10 iteration
        if it % 10 == 0 or it == 1:
            print(f"iter {it:3d}  loss={li:.6e}  |grad| pre={preclip:.3e} post={postclip:.3e}")

        # After a certain number of iterations lower the learning rate to hone in more on the minimum (open to experimentation)
        if it in {300, 1500}:
            for pg in opt.param_groups:
                pg["lr"] *= 0.5

    print("\nBest loss:", best_loss)

    # The best parameters are saved and converted back from the log space
    final_phi_log = best_phi
    theta_hat = torch.exp(final_phi_log).cpu().numpy()

    # Saving the best parameters that were found
    pd.DataFrame({"name": PARAM_ORDER, "value": theta_hat}).to_csv(OUT_CSV, index=False)
    np.save(OUT_NPY, theta_hat)
    with open(OUT_JSON, "w") as f:
        json.dump({k: float(v) for k, v in zip(PARAM_ORDER, theta_hat)}, f, indent=2)
    print(f"Saved:\n  {OUT_CSV}\n  {OUT_NPY}\n  {OUT_JSON}")

    # Plotting comparison of fitted plots and the optimized parameter ones of the first sample
    if PLOT_FIRST_SAMPLE_COMPARISON and len(batches) > 0:
        sid0, y0_vec, t_obs, Y_obs = batches[0]
        with torch.no_grad():
            # Running the ODE solver with the optimized parameters and the recovered parameters
            Y_prior = simulate_batch(phi_prior,y0_vec, t_obs).cpu().numpy()
            Y_fit   = simulate_batch(final_phi_log, y0_vec, t_obs).cpu().numpy()
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

if __name__ == "__main__":
    main()
