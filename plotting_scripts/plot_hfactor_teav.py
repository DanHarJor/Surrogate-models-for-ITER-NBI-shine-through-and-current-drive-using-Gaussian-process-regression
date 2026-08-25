import pandas as pd
import matplotlib.pyplot as plt
import os
import scienceplots
plt.style.use(['science', 'no-latex'])

script_dir = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(script_dir, "..", "shine_through_data_sets", "sobol_seq", "maxST0.3_filtered.csv")
OUT_PATH = os.path.join(script_dir, "plots", "scatter_hfactor_vs_teav.png")

df = pd.read_csv(CSV_PATH)
df = df[df["success"] == True]

# Matplotlib publication style, single-column figure
plt.rcParams.update({
    "font.size": 8,
    "font.family": "serif",
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.figsize": (3.4, 2.6),   # single-column width
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

p_nbi = 16.5 * (df["enbi"] / df["enbi"].max()) ** 2.5
color_val = p_nbi / df["nbar"]

fig, ax = plt.subplots()

sc = ax.scatter(
    df["hfactor"],
    df["teav"],
    c=color_val,
    cmap="plasma",
    s=8,
    alpha=0.85,
    linewidths=0,
)

ax.set_xlabel(r"$H_{98}$")
ax.set_ylabel(r"$T_{e,av}$ [keV]")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
cbar.set_label(r"$P_\text{NBI}/\bar{n}$  [MW / $10^{19}\,\mathrm{m}^{-3}$]")
cbar.ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(OUT_PATH)
plt.savefig(OUT_PATH.replace(".png", ".pdf"))
plt.close()

print(f"Saved: {OUT_PATH}")
