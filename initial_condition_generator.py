import numpy as np
import pandas as pd

from model import STATE_ORDER as state_names

n_samples = 20

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

# This is the intial values from the paper, it will be the first item in the list
baseline_y0 = np.array([0, 0, 30000, 0, 0, 0, 2700, 0, 0, 0])

def sample_initial_conditions(n_samples):
    y0_list = []
    for _ in range(n_samples):
        # The * unpacks the tuple we get from the state_ranges dictionary to two separate values
        # Then random values between the ranges between those two values can be generated
        y0 = np.array([np.random.uniform(*state_ranges[name]) for name in state_names])
        y0_list.append(y0)
    return np.vstack(y0_list)

random_y0s = sample_initial_conditions(n_samples)
# Combining the initial condition values given by the paper and the randomly generated ones
all_y0s = np.vstack([baseline_y0, random_y0s])

# Convert it to pandas dataframe, makes it easy to convert to a csv file and save it
df_y0 = pd.DataFrame(all_y0s, columns=state_names)

csv_path = "/Users/erencimentepe/Desktop/VSCode Projects/Thesis/initial_conditions.csv"
df_y0.to_csv(csv_path, index=False)
print(df_y0)
print(f"\nSaved to {csv_path}")
