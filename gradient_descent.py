from __future__ import annotations
from pathlib import Path
from typing import Sequence, List, Tuple
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import os
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch import nn

try:
    torch.set_num_threads(1)
except Exception:
    pass

try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

from model import (
    STATE_ORDER, PARAM_ORDER, PARAMS_OPT,
    make_phi_prior, obs_index, integrate, load_batches,
    set_device_dtype, METHOD, RTOL, ATOL, TORCH_OPTS
)

# Basic config, the tensors will exist in the cpu and the gradients will be calculated to a float64 level precision
set_device_dtype("cpu", torch.float64)   # change to ("cuda", torch.float64) if you want GPU

# All of the paths that are used to get the training data and save results
DATA_DIR   = Path("/Users/WangGroup_UofT/Desktop/Coding Projects/Thesis")
# data_collection_final_combined
COMBINED_SAMPLES_CSV = DATA_DIR / "data_collection_final_combined.csv"
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
INIT_MODE  = "uniform_log"                       # "jitter" or "random" or "uniform_log"
JITTER_SD  = 0.7                           # e^JITTER_SD addition, since the optimization parameters are in log space, so roughly a 1% difference
RAND_WIDTH = 2.5                           # If the random init mode is selected it will add random noise from +- 0.35 range
N_ITERS    = 3000                            # Number of iterations
LR         = 1e-2                         # Learning rate 1.5e-3
CLIP_NORM  = 300.0                          # Clipping that prevents exploding gradients having a huge impact
PLOT_FIRST_SAMPLE_COMPARISON = True         # Whether or not to produce the final comparison plots

# Can stop iterating if the change in error is less than 1e-3
MIN_IMPROVE = 1e-4 # Was 1e-4 but it is taking too long 

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
    elif init_mode == "uniform_log":
        param_grids = {
            "c_kAQ":   [0.001, 0.01, 0.1, 1.0, 10.0, 80.0],
            "c_kAP":   [0.16, 0.5, 1.0, 5.0, 10.0, 45.0],
            "c_kQA":   [0.01, 0.1, 1.0, 10.0, 100.0, 185.0],
            "c_kMyo":  [42.0, 100.0, 250.0, 500.0, 1150.0],
            "c_kPQ":   [0.06, 0.6, 6.0, 60.0, 145.0],
            "c_kPT":   [1.75, 3.5, 5.0, 7.0, 9.2],
            "c_kPSC":  [1.9, 3.5, 5.0, 7.5, 9.8],
            "c_kTM":   [0.35, 0.6, 0.8, 1.0, 1.3]
        }
        
        # Default choices for any parameter NOT in the dictionary above
        default_choices = torch.tensor([0.1, 1.0, 10.0], device=phi_prior.device)
        
        # Initialize theta
        theta = torch.zeros_like(phi_prior).to(phi_prior.device)
        
        for i, name in enumerate(PARAM_ORDER):
            if name in param_grids:
                # Convert the specific list for this parameter to a tensor
                choices = torch.tensor(param_grids[name], device=phi_prior.device)
                # Pick ONE random value from this parameter's specific list
                rand_idx = torch.randint(0, len(choices), (1,), device=phi_prior.device)
                theta[i] = torch.log(choices[rand_idx])
            else:
                # Fallback for parameters not explicitly listed
                rand_idx = torch.randint(0, len(default_choices), (1,), device=phi_prior.device)
                theta[i] = torch.log(default_choices[rand_idx])
        
        print(f"Initial log-params (linear scale): {torch.exp(theta).detach().cpu().numpy()}")
        return theta.clone().detach().requires_grad_(True)
    else:
        raise ValueError("INIT_MODE must be 'jitter' or 'random' or 'uniform_log'.")

def train_once(seed: int, init_mode: str = INIT_MODE, jitter_sd: float = JITTER_SD, rand_width: float = RAND_WIDTH, n_iters: int = N_ITERS,
                lr: float = LR, clip_norm: float = CLIP_NORM, min_improve: float = MIN_IMPROVE, run_dir: Path | None = None) -> dict:
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    samples_df = pd.read_csv(COMBINED_SAMPLES_CSV)
    y0_df      = pd.read_csv(Y0_CSV)
    batches, w = load_batches(samples_df, y0_df, OBS_STATES)

    # Local init
    def _init_theta_log() -> torch.Tensor:
        if init_mode == "jitter":
            noise = torch.randn_like(phi_prior) * jitter_sd
            return (phi_prior + noise).clone().detach().requires_grad_(True)
        elif init_mode == "random":
            u = (torch.rand_like(phi_prior) * 2.0 - 1.0) * rand_width
            return (phi_prior + u).clone().detach().requires_grad_(True)
        elif init_mode == "uniform_log":
            param_grids = {
                "c_kAQ":   [0.001, 0.01, 0.1, 1.0, 10.0, 80.0],
                "c_kAP":   [0.16, 0.5, 1.0, 5.0, 10.0, 45.0],
                "c_kQA":   [0.01, 0.1, 1.0, 10.0, 100.0, 185.0],
                "c_kMyo":  [42.0, 100.0, 250.0, 500.0, 1150.0],
                "c_kPQ":   [0.06, 0.6, 6.0, 60.0, 145.0],
                "c_kPT":   [1.75, 3.5, 5.0, 7.0, 9.2],
                "c_kPSC":  [1.9, 3.5, 5.0, 7.5, 9.8],
                "c_kTM":   [0.35, 0.6, 0.8, 1.0, 1.3]
            }
            
            # Default choices for any parameter NOT in the dictionary above
            default_choices = torch.tensor([0.1, 1.0, 10.0], device=phi_prior.device)
            
            # Initialize theta
            theta = torch.zeros_like(phi_prior).to(phi_prior.device)
            
            for i, name in enumerate(PARAM_ORDER):
                if name in param_grids:
                    # Convert the specific list for this parameter to a tensor
                    choices = torch.tensor(param_grids[name], device=phi_prior.device)
                    # Pick ONE random value from this parameter's specific list
                    rand_idx = torch.randint(0, len(choices), (1,), device=phi_prior.device)
                    theta[i] = torch.log(choices[rand_idx])
                else:
                    # Fallback for parameters not explicitly listed
                    rand_idx = torch.randint(0, len(default_choices), (1,), device=phi_prior.device)
                    theta[i] = torch.log(default_choices[rand_idx])
            
            print(f"Initial log-params (linear scale): {torch.exp(theta).detach().cpu().numpy()}")
            return theta.clone().detach().requires_grad_(True)
        else:
            raise ValueError("INIT_MODE must be 'jitter' or 'random' or 'uniform_log.")

    theta_log = _init_theta_log()
    opt = torch.optim.Adam([theta_log], lr=lr)

    best_loss = float('inf')
    best_phi  = theta_log.detach().clone()

    loss_history = []

    with torch.no_grad():
        #loss_at_prior = total_loss(phi_prior, batches, w)
        # initial_loss_of_run 
        loss_at_prior = total_loss(theta_log, batches, w)

    prev_loss = None
    
    for it in range(1, n_iters + 1):
        opt.zero_grad(set_to_none=True)
        loss = total_loss(theta_log, batches, w)
        loss.backward()
        nn.utils.clip_grad_norm_([theta_log], max_norm=clip_norm)
        opt.step()

        li = float(loss.item())
        if li < best_loss:
            best_loss = li
            best_phi  = theta_log.detach().clone()

        if it % 10 == 0 or it == 1:
            loss_history.append((it, li))

        # Early stop
        if prev_loss is not None and abs(prev_loss - li) < min_improve:
            if len(loss_history) == 0 or loss_history[-1][0] != it:
                loss_history.append((it, li))
            # (optional) print once on early stop:
            # print(f"[seed {seed}] early stop at iter {it}, Δloss={abs(prev_loss-li):.3e}", flush=True)
            break
        prev_loss = li

        # LR schedule
        if it in {300, 600}:
            for pg in opt.param_groups:
                pg["lr"] *= 0.5

    theta_hat = torch.exp(best_phi).cpu().numpy()

    # Per-run artifacts (lightweight)
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"name": PARAM_ORDER, "value": theta_hat}).to_csv(run_dir / "params.csv", index=False)
        np.save(run_dir / "params.npy", theta_hat)
        with open(run_dir / "params.json", "w") as f:
            json.dump({k: float(v) for k, v in zip(PARAM_ORDER, theta_hat)}, f, indent=2)
        
        # Save loss history as CSV
        if len(loss_history) > 0:
            loss_df = pd.DataFrame(loss_history, columns=["iteration", "loss"])
            loss_df.to_csv(run_dir / "loss_history.csv", index=False)

            # Save loss curve plot
            plt.figure(figsize=(6, 4))
            plt.plot(loss_df["iteration"], loss_df["loss"])
            
            plt.yscale('log')
            
            plt.xlabel("Iteration")
            plt.ylabel("Loss (MSE)")
            plt.title(f"Loss vs Iteration (seed={seed})")
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.tight_layout()
            plt.savefig(run_dir / "loss_curve.png", dpi=200)
            plt.close()

        # Summary file
        with open(run_dir / "summary.json", "w") as f:
            json.dump({
                "seed": seed,
                "init_mode": init_mode,
                "jitter_sd": jitter_sd,
                "rand_width": rand_width,
                "n_iters": n_iters,
                "base_lr": lr,
                "clip_norm": clip_norm,
                "min_improve": min_improve,
                "loss_at_prior": float(loss_at_prior),
                "best_loss": float(best_loss),
                "logged_points": len(loss_history),
            }, f, indent=2)

    row = {
        "seed": seed,
        "init_mode": init_mode,
        "jitter_sd": jitter_sd,
        "rand_width": rand_width,
        "n_iters": n_iters,
        "base_lr": lr,
        "clip_norm": clip_norm,
        "min_improve": min_improve,
        "loss_at_prior": float(loss_at_prior),
        "best_loss": float(best_loss),
    }
    row.update({f"param_{k}": float(v) for k, v in zip(PARAM_ORDER, theta_hat)})
    return row

def run_sweep(n_runs: int = 6, max_workers: int | None = None) -> Path:
    """
    Launch n_runs parallel train_once jobs and save a combined CSV for PCA.
    Returns the path to the sweep CSV.
    """
    # It is safer to use spawn instead of fork to create multiple processes in mac and Windows
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Create the directory that the results will be stored in
    runs_root = DATA_DIR / "gd_sweep_runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    # Generate a list of seeds to run
    seeds = list(range(127, 127 + n_runs))

    # If no cpu core number was passed, automatically decide how many to use
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, n_runs)
    print(f"Launching sweep with {n_runs} runs using {max_workers} parallel workers.")

    rows = []
    future_to_seed = {}
    # Previous was this
    # with ProcessPoolExecutor(max_workers=max_workers) as ex:
    with ProcessPoolExecutor(max_workers=max_workers, max_tasks_per_child=1) as ex:
        for s in seeds:
            # Create a subfolder for each seed
            run_dir = runs_root / f"seed_{s}"
            # Start the work using the train_once function
            fut = ex.submit(
                train_once,
                seed=s,
                init_mode=INIT_MODE,
                jitter_sd=JITTER_SD,
                rand_width=RAND_WIDTH,
                n_iters=N_ITERS,
                lr=LR,
                clip_norm=CLIP_NORM,
                min_improve=MIN_IMPROVE,
                run_dir=run_dir,
            )
            future_to_seed[fut] = s

        for fut in as_completed(future_to_seed):
            seed_num = future_to_seed[fut]
            try:
                # If this seed hit an underflow error, it will trigger the 'except' block
                result_row = fut.result()
                rows.append(result_row)
                print(f"Finished seed {seed_num}")
            except Exception as e:
                # This catches the Seed 53 error and keeps the script alive
                print(f"Seed {seed_num} CRASHED with error: {e}")
                continue

    sweep_df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    sweep_csv = runs_root / "sweep_results.csv"
    sweep_df.to_csv(sweep_csv, index=False)
    print(f"Saved sweep table: {sweep_csv}")
    print(sweep_df.head())
    return sweep_csv

# Training
def main():
    # Sanity check
    if not COMBINED_SAMPLES_CSV.exists() or not Y0_CSV.exists():
        raise FileNotFoundError("Missing inputs. Run the generator first.")

    # Load dataframes and the batches
    samples_df = pd.read_csv(COMBINED_SAMPLES_CSV)
    y0_df      = pd.read_csv(Y0_CSV)
    batches, w = load_batches(samples_df, y0_df, OBS_STATES)

    # print("-----------------Batches----------------------")
    # print(batches)
    # print("-----------------Weights--------------------")
    # print(w)
    # print("--------------------End-----------------")
    # exit()

    theta_log = init_theta_log(INIT_MODE)
    # Initialize optimizer, Adam
    opt = torch.optim.Adam([theta_log], lr=LR)

    # Best loss and the parameters that gave the best loss
    best_loss = float('inf')
    best_phi  = theta_log.detach().clone()

    # Checking what the loss is before any gradients with the optimized parameters
    with torch.no_grad():
        #loss_at_prior = total_loss(phi_prior, batches, w)
        loss_at_prior = total_loss(theta_log, batches, w)
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
        if it in {300, 600}:
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
    parser = argparse.ArgumentParser()
    # Define terminal argument --sweep, when given will run that many instances of the gradient descent in parallel
    parser.add_argument("--sweep", type=int, default=0)
    # Define terminal argument --workers, when given will determine the number of cpu cores that can be used for training
    parser.add_argument("--workers", type=int, default=None)
    # Read the arguments we are passing to the 
    args = parser.parse_args()

    if args.sweep and args.sweep > 0:
        run_sweep(n_runs=args.sweep, max_workers=args.workers)
    else:
        main()
