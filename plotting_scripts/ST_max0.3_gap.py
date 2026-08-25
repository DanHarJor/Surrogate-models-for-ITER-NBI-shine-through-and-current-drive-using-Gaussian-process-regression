import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import os
import scienceplots
plt.style.use(['science', 'no-latex'])

script_dir = os.path.dirname(os.path.abspath(__file__))
input_csv = os.path.join(script_dir, "..", "shine_through_data_sets", "sobol_seq", "maxST0.3_filtered.csv")

# Load raw CSV (multiple repeats per simulation) and collapse to mean/std
df_raw = pd.read_csv(input_csv)

# Physical parameters
params = ["enbi", "hfactor", "nbar", "np"]

df = (
    df_raw.groupby(params)["output_shine_inj1"]
    .agg(output_mean="mean", output_std="std")
    .reset_index()
)

# Matplotlib publication style
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.figsize": (3.4, 2.6),   # single-column width
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "lines.markersize": 3,
})

# Normalize output_mean for colormap
norm = plt.Normalize(df["output_mean"].min(), df["output_mean"].max())
cmap = plt.cm.plasma

# -----------------------------
# Heatmap-style scatter: output_std
# -----------------------------
x = df["enbi"].values
y = df["nbar"].values
z_std = df["output_std"].values
z_mean = df["output_mean"].values

# -----------------------------
# Heatmap-style scatter: output_mean
# -----------------------------
fig, ax = plt.subplots()
sc = ax.scatter(x, y, c=z_mean, cmap="plasma", s=12, alpha=0.85, linewidths=0)
ax.set_xlabel(r"$E_\text{NBI} \, [keV]$")
ax.set_ylabel(r"$\bar{n_e}$  $\left[10^{19}\text{m}^{-3}\right]$")
# ax.set_title("output_mean across (enbi, nbar)")

cbar = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.05)
cbar.set_label("ST")

tri = Polygon(
    [(575, 1.0), (1000, 1.0), (1000, 1.65)],
    closed=True,
    facecolor="none",
    edgecolor="black",
    linewidth=1.1,
    zorder=3,
)
ax.add_patch(tri)
ax.text(
    900, 1.2, "ST > 0.3",
    ha="center", va="center", fontsize=8, zorder=4,
)

plt.tight_layout()

output_png = os.path.join(script_dir, "plots", "ST_max0.3_gap.png")
plt.savefig(output_png, dpi=300, bbox_inches="tight")

plt.close()
