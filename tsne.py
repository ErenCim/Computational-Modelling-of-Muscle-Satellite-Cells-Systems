import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import fill
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load the parameters we are analysing
# fitted_targeted_recip
df = pd.read_csv("fitted_reciprocal_all_param.csv")
df = df[df['best_loss'] < 6.3]
print(df[df['param_c_kMyo'] > 10000])
#df = df[df['param_c_kMyo'] < 10000]
print(df.shape[0])
#df.to_csv("filtered_results_11.csv", index=False)

param_cols = [c for c in df.columns if c.startswith("param_")]
X = df[param_cols].values

# Standardize (mean=0, sd=1) so no parameter dominates by scale
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
tsne = TSNE(n_components=2, perplexity=3,random_state=42)

X_tsne = tsne.fit_transform(X_std)

plt.figure(figsize=(8, 6))
sc = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['best_loss'], cmap='viridis_r', alpha=0.8)
plt.colorbar(sc, label='Best Loss')
plt.title("t-SNE Visualization of Parameter Space")
plt.xlabel("t-SNE dimension 1")
plt.ylabel("t-SNE dimension 2")
plt.show()

print("KL Divergence:")
print(tsne.kl_divergence_)

param_cols = [c for c in df.columns if c.startswith("param_")]
for col in param_cols:
    m = df[col].mean()
    s = df[col].std()
    print(f"{col:15} | Mean: {m:12.2f} | STD: {s:12.2f}")




df = pd.read_csv("fitted_reciprocal_all_param.csv")
df_target = df[df['best_loss'] < 6.3]
print(len(df_target))
#df_target = df[df['param_c_kMyo'] < 10000]

# Choose parameter to visualize
parameter_name = 'param_c_kMyo' # 'param_c_knew'


plt.figure(figsize=(10, 5))
n, bins, patches = plt.hist(df_target[parameter_name], bins=20, color='skyblue', edgecolor='black', alpha=0.8)
lower_bound = bins[0]
upper_bound = bins[1]
print((upper_bound + lower_bound)/2)
print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
print("First column height: ", n[0])

#plt.xlabel(f'Values of cMyo')
plt.xlabel(f'Values of {parameter_name}')
plt.ylabel('Count of Initializations')
plt.title(f'Histogram of Recovered {parameter_name} Across Best Loss')
#plt.title(f'Standard Histogram of {parameter_name}')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


# Scatter plot of loss vs log(ckMyo)
plt.figure(figsize=(10, 6))
log_ckmyo = np.log10(df_target[parameter_name])

plt.scatter(
    log_ckmyo, 
    df_target['best_loss'],
    c='teal', 
    edgecolor='k'
)

# Add a trend line to see the relationship
#z = np.polyfit(log_ckmyo, df_target['best_loss'], 1)
#p = np.poly1d(z)
#plt.plot(log_ckmyo, p(log_ckmyo), "r--", alpha=0.8, label="Linear Trend")

plt.xlabel(f'log10({parameter_name})')
plt.ylabel('Best Loss')
plt.title('Relationship: Best Loss vs. Log Carrying Capacity')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()