# thesis_ode/shared.py
from __future__ import annotations
from typing import Sequence, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
from torchdiffeq import odeint

# Default settings for where to store the tensors and the precision of the gradients
DEFAULT_DTYPE = torch.float64
DEFAULT_DEVICE = torch.device("cpu")
torch.set_default_dtype(DEFAULT_DTYPE)

# Given new DTYPE and DEVICE, we can update them
def set_device_dtype(device: torch.device | str = "cpu", dtype: torch.dtype = torch.float64) -> None:
    global DEFAULT_DEVICE, DEFAULT_DTYPE
    DEFAULT_DEVICE = torch.device(device)
    DEFAULT_DTYPE  = dtype
    torch.set_default_dtype(dtype)

# The states variables, for the relevant cells to the model
STATE_ORDER: Sequence[str] = ["N","N_d","M_d","M","M1","M2","QSC","ASC","M_c","M_n"]
# Parameters of the model, they are the coefficient terms that we are trying to recover with gradient descent
PARAM_ORDER: Sequence[str] = [
    "c_NMd","c_Nin","c_Nout","c_M1Nd",
    "c_Min","c_MM1","c_Mout","c_M1out",
    "c_M1M2","c_M2inhib","c_M2out",
    "c_QSCN","c_QSCMd","c_ASCM2","c_ASCpro",
    "c_ASCdiff","c_ASCout","c_Mcout","c_Mcfusion",
    "c_Mdout","c_QSCself","c_Mnself","QSCmax"
]

# Optimized parameters from literature
PARAMS_OPT = dict(
    c_NMd=1.18e-05, c_Nin=1.38e+00, c_Nout=3.10e+00, c_M1Nd=9.17e-5,
    c_Min=2.81e+01, c_MM1=5.81e-05, c_Mout=2.31e+01, c_M1out=4.29e-02,
    c_M1M2=5.48e+02, c_M2inhib=4.58e-12, c_M2out=1.18e-1,
    c_QSCN=5.22e-06, c_QSCMd=2.09e-04, c_ASCM2=8.38e-03, c_ASCpro=4.28e-05,
    c_ASCdiff=7.24e-03, c_ASCout=1.55e-04, c_Mcout=3.73e-06, c_Mcfusion=4.01e+00,
    c_Mdout=9.81e-04, c_QSCself=9.63e-02, c_Mnself=9.39e-05,
    QSCmax=2700.0
)

# Extract the values form the previously optimized parameter dictionary and return it in log space as a tensor
def make_phi_prior(params: dict[str, float]) -> torch.Tensor:
    arr = np.array([params[k] for k in PARAM_ORDER], dtype=np.float64)
    return torch.tensor(np.log(arr), dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)

# Creating a mapping from state name to index (output eg. {"N":0, "N_d":1, "M_d":2, ... "M_n":9})
def state_index_map(state_order: Sequence[str] = STATE_ORDER) -> dict[str,int]:
    return {s: i for i, s in enumerate(state_order)}

# If only a subset of states is observed (eg. only M, QSC, M2), return a vector of indices to extract them.
# Example input: ["M", "QSC", "M2"], and the output would be a tensor containing [3, 6, 5], correspondind indeces from the state parameter vector
def obs_index(obs_states: Sequence[str], state_order: Sequence[str] = STATE_ORDER) -> torch.Tensor:
    idx_map = state_index_map(state_order)
    return torch.tensor([idx_map[s] for s in obs_states], dtype=torch.long, device=DEFAULT_DEVICE)

# ODE Solver Configs
T_MAX = 7.0
# Relative and absolute tolerances, ATOL needs to be small because when the state parameters are small, they must be more accurate
# RTOL is a bit larger, because if the state is large like 30,000 then the the tolerance for error can be larger
RTOL, ATOL = 1e-6, 1e-9
# ODE solver, bosh3 is good for stiff systems where some states will change fast while others will have smaller changes
METHOD = "bosh3"
# This prevents the solver running infinitely, limiting the number of internal steps it can take to reach the desired time points
TORCH_OPTS = {"max_num_steps": 200_000}

# The paper used ReLu but I swtitched to softplus, which is very similar to relu for differentiability
def softpos(x: torch.Tensor, beta: float = 40.0) -> torch.Tensor:
    return F.softplus(x, beta=beta)

# ODE System - Providing the definition of all of the equations
def dydt(t: torch.Tensor, y: torch.Tensor, theta_log: torch.Tensor) -> torch.Tensor:
    # A very tiny epsilon tensor
    eps = torch.tensor(1e-12, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)
    
    # Getting the parameters that we are trying to optimize back from their log state to real values
    theta = torch.exp(theta_log)

    # torch.unbind will split the tensor and give the numerical values corresponding at this current iteration of the GD
    (c_NMd, c_Nin, c_Nout, c_M1Nd,
     c_Min, c_MM1, c_Mout, c_M1out,
     c_M1M2, c_M2inhib, c_M2out,
     c_QSCN, c_QSCMd, c_ASCM2, c_ASCpro,
     c_ASCdiff, c_ASCout, c_Mcout, c_Mcfusion,
     c_Mdout, c_QSCself, c_Mnself, QSCmax) = torch.unbind(theta)

    N, N_d, M_d, M, M1, M2, QSC, ASC, M_c, M_n = torch.unbind(y)

    # I am experimenting with these, basically ensuring that the apoptotic cells counts are always positive
    # Since it wouldn't biologically make sense for them to me negative values. But, I don't think it makes a big difference
    Nd_pos = torch.clamp(N_d, min=0)
    Md_pos = torch.clamp(M_d, min=0)

    dN_dt  = c_Nin*M_d - c_Nout*N - c_NMd*N*M_d
    dMd_dt = -c_Mdout*M_d*M1 - c_NMd*M_d*N
    dNd_dt = c_NMd*N*M_d - c_M1Nd*N_d*M1
    dM_dt  = c_Min*N - c_MM1*M*(Nd_pos + Md_pos) - c_Mout*M

    # Adding the tiny epsilon to ensure the denominator is never exactly 0
    den = c_M2inhib + Nd_pos + Md_pos + eps
    m1_to_m2 = (c_M1M2*M1) / den

    dM1_dt = c_MM1*M*(Nd_pos + Md_pos) - c_M1out*M1 - m1_to_m2
    dM2_dt = m1_to_m2 - c_M2out*M2

    dQSCdt = -c_QSCN*QSC*N - c_QSCMd*QSC*M_d + c_ASCM2*ASC*M2 - c_QSCself*QSC*softpos(QSC - QSCmax)
    dASCdt =  c_QSCN*QSC*N + c_QSCMd*QSC*M_d - c_ASCM2*ASC*M2 + c_ASCpro*ASC*M1 - c_ASCdiff*ASC*M2 - c_ASCout*ASC
    dM_cdt = c_ASCdiff*ASC*M2 - c_Mcfusion*M_c - c_Mcout*M_c
    dMn_dt = c_Mcfusion*M_c - c_Mnself*M_n*softpos(M_d - 3000.0)

    return torch.stack([dN_dt, dNd_dt, dMd_dt, dM_dt, dM1_dt, dM2_dt, dQSCdt, dASCdt, dM_cdt, dMn_dt])

# Starting from the y0 initial condition solves the system of ODEs and returns the parameters are the given time t (returns y(t))
def integrate(y0: torch.Tensor, t: torch.Tensor, theta_log: torch.Tensor) -> torch.Tensor:
    # t is a tensor of time points, will be irregular
    # theta_log is the parameters that are being optimized in the log space
    # Normally odeint takes two arguments, but our ODE solver has 3, which means we need a wrapper function that allows the odeint to solve the system of ODEs using the dydt function
    # The lambda defines a function that takes current time (tt) and current initial conditions (yy) and solves the system of equations for those time points
    return odeint(func=lambda tt, yy: dydt(tt, yy, theta_log), y0=y0, t=t, method=METHOD, rtol=RTOL, atol=ATOL, options=TORCH_OPTS)

# theta_log is the parameter tensor, y0_np is the initial conditions that we are starting with and t_np is the specific time points we are interested in on the ODEs
def simulate_on_times(y0_np: np.ndarray, t_np: np.ndarray, theta_log: torch.Tensor) -> np.ndarray:
    # Initial conditions and the time points are converted to tensors
    y0 = torch.tensor(y0_np, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)
    tt = torch.tensor(t_np,  dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)
    # Then the ODE solver is called
    Y  = integrate(y0, tt, theta_log)
    # Move the tensor output to cpu and convert it to a np array
    return Y.detach().cpu().numpy()

# Irregular sampling
# When we say sampling, we are building the time points in which the ODE solver solves and returns the values for
def irregular_times(t_max: float, rng: np.random.Generator, min_pts: int = 10, max_pts: int = 25, min_dt: float = 0.1, max_dt: float = 0.8, random_start_frac: float = 0.6) -> np.ndarray:
    # Finding a random number of points to sample
    n = int(rng.integers(min_pts, max_pts + 1))
    # Starting time of the irregular sampling
    t0   = float(rng.uniform(0.0, random_start_frac * t_max))
    # Generates gaps with random size to add to t0, so that new sample points can be determined (minus 2 since first and last time points don't need gaps)
    gaps = rng.uniform(min_dt, max_dt, size=max(0, n - 2)).astype(np.float64)
    # Adding the gaps to the starting time, and building the set of time points that are going to be sampled
    ts   = (np.concatenate([[t0], t0 + np.cumsum(gaps)]) if gaps.size > 0 else np.array([t0], dtype=np.float64))
    # Remove any out of range times
    ts = ts[(ts >= 0.0) & (ts <= t_max)]
    # Enforcing that time = 0 is one of the points in which we are observing the value of the ODE system
    # np.unique ensures if the 0 was repeated, it is not included twice
    ts = np.unique(np.concatenate([[0.0], ts])).astype(np.float64)
    return ts

# Helper function for weighted MSE
def per_state_weights(samples_df, obs_states: Sequence[str]) -> torch.Tensor:
    # Find the maximum value that states reach (eg. for M_d it could be 45,000 since it ranges from 0 - 50,000 randomly)
    y_max = {s: max(1.0, float(samples_df[s].max())) for s in obs_states}
    # Compute weights based on this value, 1 / (max^2) scaling, protects against large dynamic-range states
    w_np = np.array([1.0 / (y_max[s]**2) for s in obs_states], dtype=np.float64)
    # Returns calculated weights for weighted MSE calculation
    return torch.tensor(w_np, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)

# Takes irregular samples (samples_df) and the initial condition data and creates a batch to use during training
def load_batches(samples_df, y0_df, obs_states: Sequence[str] = STATE_ORDER) -> Tuple[List[tuple], torch.Tensor, int]:
    """
    Returns:
      batches: list of (sample_id, y0_vec[T], t_obs[T], Y_obs[T,S])
      w:       per-state weight vector [S]
    """
    # Compute the weights to use in loss calculation
    w = per_state_weights(samples_df, obs_states)
    batches = []
    # Iterate through groups with the same sample id
    for sid, df in samples_df.groupby("sample_id"):
        # Sorts in ascending order of time and collects all the rows in one data frame
        df_t = df.drop_duplicates(subset="t").sort_values("t")
        # Converts the t column to numpy array for the ODE solver then to a tensor
        t = df_t["t"].to_numpy(np.float64)
        t_obs = torch.tensor(t, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)
        # Extracts the columns for the states that we are observing and converts it to a tensor
        Y_obs = torch.tensor(df_t[list(obs_states)].to_numpy(np.float64), dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)  # (T, S)
        # Gets the initial condition for the sample with this sample id and converts it to a tensor
        y0_vec = torch.tensor(y0_df.loc[int(sid), STATE_ORDER].to_numpy(np.float64), dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)  # (10,)
        # Concatenates everything together for the batches
        batches.append((int(sid), y0_vec, t_obs, Y_obs))
    return batches, w
