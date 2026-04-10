import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import fill
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the parameters we are analysing
df = pd.read_csv("fitted_parameters_data.csv")
#df = df[df['best_loss'] < 15]
#df_new = df[df['best_loss'] < 1000]
#df_rev = df[df['best_loss'] > 1000]
#print(df)
# print("==============")
# print(df_new.size == df.size)
# print("==============")

# low_losses = df.loc[df['best_loss'] < 1000, 'best_loss']


# Filter for the GD runs that have converged to the absolute minimum (or have come close to converging)
# df = df[df["best_loss"] < 1].copy()
# print(df) <---- Here for debugging

param_cols = [c for c in df.columns if c.startswith("param_")]
X = df[param_cols].values

# Standardize (mean=0, sd=1) so no parameter dominates by scale
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

param_stats = pd.DataFrame({
    "parameter": param_cols,
    "mean": scaler.mean_,
    "variance": scaler.var_,
    "std": scaler.scale_,
})
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

print("\nUnscaled parameter statistics (filtered runs only):")
print(param_stats)

# Full PCA (all components)
n_components = min(X_std.shape)   # = min(n_samples, n_features), compute as many PCs as we can
print("N components")
print(n_components)
print("X_std shape")
print(min(X_std.shape))

pca = PCA(n_components=n_components, random_state=0)
PC = pca.fit_transform(X_std)     # shape: (runs, PCs)

pc_components = pca.components_.shape[0]
print("Number of PCs:", pc_components)

def plot_pc_vs(pc_x_num: int, pc_y_num: int, annotate=True):
    """
    Plot PC{pc_x_num} vs PC{pc_y_num} using the already-defined PC, pca, and df.
    Uses 1-based PC numbers (PC1 => 1, PC10 => 10, etc.).
    """
    # convert to 0-based indices
    ix, iy = pc_x_num - 1, pc_y_num - 1

    # bounds check
    n_avail = PC.shape[1]
    if ix >= n_avail or iy >= n_avail or ix < 0 or iy < 0:
        print(f"Not enough PCs: requested PC{pc_x_num} vs PC{pc_y_num}, but only {n_avail} PCs available.")
        return

    # variance % for labels
    vx = pca.explained_variance_ratio_[ix] * 100
    vy = pca.explained_variance_ratio_[iy] * 100

    # plot
    plt.figure(figsize=(7,6))
    
    #plt.scatter(PC[:, ix], PC[:, iy], alpha=0.8)

    # --- COLORING LOGIC ---
    # We use 'c' for the data, and 'cmap' for the color palette.
    # 'viridis_r' is good because it makes low loss (good) dark/purple 
    # and high loss (bad) yellow.
    sc = plt.scatter(PC[:, ix], PC[:, iy], 
                     c=df["best_loss"], 
                     cmap='viridis_r', 
                     alpha=0.8, 
                     edgecolors='w', 
                     linewidth=0.5)

    # Add a colorbar so we know what the colors mean
    cbar = plt.colorbar(sc)
    cbar.set_label('Loss Value (Model Error)')

    '''
    # optional point labels by seed
    if annotate and "seed" in df.columns:
        for i, seed in enumerate(df["seed"]):
            plt.text(PC[i, ix], PC[i, iy], str(seed), fontsize=8, ha="center", va="center")
    '''

    plt.title(f"PCA: PC{pc_x_num} vs PC{pc_y_num} (standardized)")
    plt.xlabel(f"PC{pc_x_num} ({vx:.2f}% var)")
    plt.ylabel(f"PC{pc_y_num} ({vy:.2f}% var)")
    plt.tight_layout()
    plt.show()

plot_pc_vs(1, 2)
plot_pc_vs(1, 3)
plot_pc_vs(1, 4)
plot_pc_vs(1, 5)
plot_pc_vs(2, 3)
plot_pc_vs(2, 4)
plot_pc_vs(2, 5)
plot_pc_vs(3, 4)
plot_pc_vs(3, 5)
plot_pc_vs(4, 5)




from mpl_toolkits.mplot3d import Axes3D

'''
ix, iy, iz = 0, 9, 10  # PC1, PC10, PC11

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(PC[:, ix], PC[:, iy], PC[:, iz], alpha=0.8)
for i, seed in enumerate(df["seed"]):
    ax.text(PC[i, ix], PC[i, iy], PC[i, iz], str(seed),
            fontsize=7, ha="center", va="center")

ax.set_title("3D PCA Scatter: PC1 vs PC10 vs PC11")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[ix]*100:.2f}% var)")
ax.set_ylabel(f"PC10 ({pca.explained_variance_ratio_[iy]*100:.2f}% var)")
ax.set_zlabel(f"PC11 ({pca.explained_variance_ratio_[iz]*100:.2f}% var)")

# Try different angles:
# view_init(elevation, azimuth)
ax.view_init(elev=45, azim=45)
plt.show()
'''

'''

# Scree plot
expl_var = pca.explained_variance_ratio_
cum_var = np.cumsum(expl_var)

plt.figure(figsize=(7,4))
plt.bar(range(1, len(expl_var)+1), expl_var*100)
# plt.plot(range(1, len(cum_var)+1), cum_var*100, marker="o")
#plt.yscale("log")  # <--- log scale here
plt.xlabel("Principal Component")
plt.ylabel("% Variance") #(In Log Space)
plt.title("Scree Plot")
plt.show()

print("Explained variance ratio per PC (%):",
      [f"{v*100:.2f}" for v in expl_var])

# Loadings Heatmap
# Loadings = coefficients of each feature on each PC.
# In sklearn, components_ has shape (PC, feature); transpose for (feature, PC)
loadings = pd.DataFrame(
    pca.components_.T,
    index=param_cols,
    columns=[f"PC{i+1}" for i in range(n_components)]
)

# Save them for reference
loadings.to_csv("pca_loadings.csv", index=True)

# Plot first K PCs (e.g., 5) so the heatmap is readable
K = min(pc_components, n_components)
plt.figure(figsize=(1.6*K+6, 0.4*len(param_cols)+3))
sns.heatmap(loadings.iloc[:, :K],
            cmap="coolwarm", center=0, annot=False,
            cbar_kws={"label": "Loading (±)"},
            yticklabels=True)
plt.title(f"PCA Loadings Heatmap (first {K} PCs)")
plt.xlabel("Principal Components")
plt.ylabel("Parameters")
plt.tight_layout()
plt.subplots_adjust(left=0.15)   
plt.show()


top_k = 3
rows = []

n_components = pca.components_.shape[0]
K = n_components   # use all PCs; or set K = min(few_components, n_components) for just first few

for j in range(K):
    vec = loadings.iloc[:, j]
    pos = vec.nlargest(top_k).round(4)
    neg = vec.nsmallest(top_k).round(4)

    rows.append({
        "PC": f"PC{j+1}",
        "Explained (%)": f"{pca.explained_variance_ratio_[j]*100:.1f}",
        "Top + contributors": "; ".join([f"{k}={v:+.4f}" for k, v in pos.items()]),
        "Top - contributors": "; ".join([f"{k}={v:+.4f}" for k, v in neg.items()])
    })

summary_df = pd.DataFrame(rows)
out_path = "pca_parameter_summary.csv"
summary_df.to_csv(out_path, index=False)
print(summary_df)

'''