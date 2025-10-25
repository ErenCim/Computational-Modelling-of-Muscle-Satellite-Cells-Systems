import numpy as np
import pandas as pd
import os

n_samples = 20

state_names = ["N","N_d","M_d","M","M1","M2","QSC","ASC","M_c","M_n"]
state_ranges = {
    "N": (0, 2000),
    "N_d": (0, 500),
    "M_d": (10000, 50000),
    "M": (0, 1000),
    "M1": (0, 1000),
    "M2": (0, 1000),
    "QSC": (2000, 3000),
    "ASC": (0, 500),
    "M_c": (0, 1000),
    "M_n": (0, 5000),
}

baseline_y0 = np.array([0, 0, 30000, 0, 0, 0, 2700, 0, 0, 0])

def sample_initial_conditions(n_samples):
    y0_list = []
    for _ in range(n_samples):
        y0 = np.array([np.random.uniform(*state_ranges[name]) for name in state_names])
        y0_list.append(y0)
    return np.vstack(y0_list)

random_y0s = sample_initial_conditions(n_samples)
all_y0s = np.vstack([baseline_y0, random_y0s])

df_y0 = pd.DataFrame(all_y0s, columns=state_names)

csv_path = "/Users/erencimentepe/Desktop/VSCode Projects/Thesis/training_data_initial_conditions.csv"
df_y0.to_csv(csv_path, index=False)
print(df_y0)
print(f"\nSaved to {csv_path}")
