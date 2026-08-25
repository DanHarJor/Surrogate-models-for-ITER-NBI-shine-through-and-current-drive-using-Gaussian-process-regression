import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Load the CSV


script_dir = os.path.dirname(os.path.abspath(__file__))
input_csv = os.path.join(script_dir, "..", "shine_through_data_sets", "noise_vs_markers_ST.csv")
do_violin = True
do_annotate = False

import scienceplots
plt.style.use(['science', 'no-latex'])


remove_bbnbi_n_markers = [10, 100]

true_values = [] # {1000: 0.255490, 10000: 0.253318, 100000: 0.252822}

df = pd.read_csv(input_csv)

df = df[~df["bbnbi_n_markers"].isin(remove_bbnbi_n_markers)]

# Color‑blind‑safe Okabe–Ito palette
okabe_ito = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
]

# Validate required columns

out_col = "output_shine_inj1"

required = {out_col, "bbnbi_n_markers", "run_time_min"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Extract raw data
x = df["bbnbi_n_markers"]
y = df[out_col]

# Compute total CPU hours per marker count
cpumin = df.groupby("bbnbi_n_markers")["run_time_min"].mean()

# Scale factor for y-axis
scale_factor = 1.0

# --- Scientific paper formatting ---
plt.rcParams.update({
    "font.size": 9,              # base font size
    "axes.labelsize": 10,        # axis labels
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.2,
    "figure.dpi": 300
})

# Single-column width: ~3.4 inches, now with 2 subplots
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(7.0, 3.0))

# Scatter plot of raw points (scaled)
ax1.scatter(
    x, y * scale_factor,
    s=18,
    alpha=0.8,
    edgecolor=okabe_ito[0],
    facecolor=okabe_ito[2],   # same blue as mean curve
    linewidth=0.4
)

# Compute mean shine output per marker count
mean_shine = df.groupby("bbnbi_n_markers")[out_col].mean() * scale_factor
# Compute SEM for each marker count
sem_shine = df.groupby("bbnbi_n_markers")[out_col].sem() * scale_factor

std_shine = df.groupby("bbnbi_n_markers")[out_col].std() * scale_factor

print('std_shine:\n',std_shine)
print('mean shine:\n',mean_shine)

# Plot mean values instead of raw scatter
# Plot mean with SEM error bars
ax1.errorbar(
    mean_shine.index,
    mean_shine.values,
    yerr= np.repeat(0, len(mean_shine)), #sem_shine.values*1.96,#sem_shine.values,
    fmt="s-",
    markersize=5,
    color=okabe_ito[4],   # sky blue
    linewidth=1.2,
    capsize=3
)

if len(true_values) > 0:
    ax1.plot(
        true_values.keys(),
        true_values.values() * scale_factor,
        marker="s",
        markersize=5,
        color=okabe_ito[4],   # yellow
        linewidth=1.2
    )

if do_annotate == True:
    # Add annotation boxes for SEM and % error
    for m, mean_val, sem_val in zip(mean_shine.index, mean_shine.values, sem_shine.values):
        percent_err = sem_val * 100 / mean_val

        text = (
            f"95% CI: {sem_val:.3g}\n"
            f"%er: {percent_err:.2f}%"
        )

        ax1.annotate(
            text,
            xy=(m, mean_val),
            xytext=(-20, 45),  # offset in points
            textcoords="offset points",
            fontsize=6,
            color=okabe_ito[0],
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                ec=okabe_ito[0],
                lw=0.5,
                alpha=0.8
            )
        )


ax1.set_xscale('log')
ax1.set_xlabel("#N markers")
ax1.set_ylabel("Shine Through Fraction")
# ax1.set_title("Noise vs. Markers", pad=6)

ax1.grid(True, linestyle="--", alpha=0.35, linewidth=0.4)

from scipy.stats import gaussian_kde
import numpy as np

marker_vals = np.sort(df["bbnbi_n_markers"].unique())

if do_violin == True:
    for m in marker_vals:
        subset = df.loc[df["bbnbi_n_markers"] == m, out_col].values * scale_factor

        if len(subset) < 2:
            continue

        kde = gaussian_kde(subset)
        y_grid = np.linspace(subset.min(), subset.max(), 200)
        density = kde(y_grid)

        # --- Wider log-space width so violins are visible ---
        log_m = np.log10(m)
        max_width_decades = 0.3   # <-- this is the key change
        density_scaled = density / density.max() * max_width_decades

        # Convert back to linear x positions
        left = 10**(log_m - density_scaled)
        right = 10**(log_m + density_scaled)

        ax1.fill_betweenx(
            y_grid,
            left,
            right,
            facecolor=okabe_ito[5],   # deep blue
            alpha=0.25,
            linewidth=0.3,
            zorder=1
        )


# --- Right subplot: Bootstrap samples ---
np.random.seed(90)  # for reproducibility
bootstrap_means = []
bootstrap_sems = []
marker_labels = []

other_csv = '/home/ITER/jordand/enchanted_plugins/data/DT_D_noise_vs_markers_5/enchanted_dataset.csv'
df = pd.read_csv(other_csv)
for m in marker_vals:
    subset = df.loc[df["bbnbi_n_markers"] == m, out_col].values * scale_factor
    
    if len(subset) < 5:
        continue
    
    # Draw 5 random samples for this marker count
    sample_means = []
    for _ in range(5):
        sample = np.random.choice(subset, size=len(subset), replace=True)
        sample_means.append(np.mean(sample))
    
    # Plot individual sample means as points with same styling as left plot
    jitter = np.random.normal(0, 0.05, size=len(sample_means))
    ax3.scatter(
        m + jitter,
        sample_means,
        s=18,
        alpha=0.8,
        edgecolor=okabe_ito[0],
        facecolor=okabe_ito[2],   # same blue as left plot
        linewidth=0.4,
        zorder=2
    )
    
    bootstrap_means.append(np.mean(sample_means))
    bootstrap_sems.append(np.std(sample_means) / np.sqrt(5))  # SEM of the bootstrap means
    marker_labels.append(m)

# Plot bootstrap means with error bars (black circles with error bars)
ax3.errorbar(
    marker_labels,
    bootstrap_means,
    yerr=[b * 1.96 for b in bootstrap_sems],  # 95% CI
    fmt="o-",
    markersize=5,
    color=okabe_ito[0],  # black
    linewidth=1.2,
    capsize=3,
    zorder=3
)

# Overlay the mean of all data for each marker count (yellow squares, no error bars)
ax3.plot(
    mean_shine.index,
    mean_shine.values,
    marker="s",
    markersize=5,
    color=okabe_ito[4],  # yellow
    linewidth=1.2,
    linestyle="-",
    zorder=4
)

ax3.set_xscale('log')
ax3.set_xlabel("#N markers")
ax3.set_ylabel("Shine Through Fraction")
ax3.grid(True, linestyle="--", alpha=0.35, linewidth=0.4)
# ax3.legend(fontsize=8)

# Twin axis for CPU hours
ax2 = ax1.twinx()
ax2.plot(
    cpumin.index,
    cpumin.values,
    color=okabe_ito[1],   # orange
    marker="x",
    markersize=5,
    linewidth=1.2
)
ax2.set_ylabel("METIS-CHEASE-BBNBI, 1 core [min]", color=okabe_ito[1])
ax2.tick_params(axis='y', labelcolor=okabe_ito[1])

# ax2.set_ylim(0,13)

plt.tight_layout(pad=0.5)

# Show and save
plt.show()

output_png = os.path.join(script_dir, "plots", "noise_vs_markers_ST.png")
plt.savefig(output_png, dpi=300, bbox_inches="tight")