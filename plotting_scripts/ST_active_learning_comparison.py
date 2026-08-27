import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import scienceplots

plt.style.use(['science', 'no-latex'])

plt.rcParams.update({
    # Grid defaults
    "axes.grid": True,            # Always show grid
    "grid.linestyle": "--",       # Dashed lines
    "grid.linewidth": 0.5,        # Thin lines
    "grid.alpha": 0.7,            # Transparency
    "grid.color": "gray",         # Neutral color

    # Optional: enable minor grid lines too
    "axes.grid.which": "both"     # Apply to both major and minor ticks
})


# Defaults optimized for scientific papers
plt.rcParams.update({
    "font.size": 10,        # Base font size (body text)
    "axes.titlesize": 12,   # Axis title (slightly larger for emphasis)
    "axes.labelsize": 11,   # Axis labels (x/y)
    "xtick.labelsize": 10,  # Tick labels
    "ytick.labelsize": 10,  # Tick labels
    "legend.fontsize": 9,   # Legend text (slightly smaller, but still readable)
    "legend.title_fontsize": 9,  # Legend title text
    "figure.titlesize": 13  # Overall figure title
})


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def find_csv_files(data_folder):
    """Find all batch_info.csv files one directory deep."""
    csv_files = []
    for entry in os.scandir(data_folder):
        if entry.is_dir():
            candidate = os.path.join(entry.path, "batch_info.csv")
            if os.path.exists(candidate):
                csv_files.append(candidate)
    return csv_files


def extract_sampling_strategy(file_path):
    """Extract strategy name from folder name."""
    return os.path.basename(os.path.dirname(file_path))


def normalize_strategy(name):
    """Remove trailing digits to group strategy variants."""
    return re.sub(r'\d+$', '', name)


def infer_step_size(values):
    """Infer natural step size from a list of num_samples values."""
    values = np.sort(np.unique(values))
    diffs = np.diff(values)
    if len(diffs) == 0:
        return 1
    return int(pd.Series(diffs).mode().iloc[0])


# ------------------------------------------------------------
# Metric grouping
# ------------------------------------------------------------

def group_metrics(columns):
    """Group metric columns into logical families."""
    groups = {}
    for col in columns:
        if col in ("num_samples", "num_samples_snapped", "strategy", "strategy_norm"):
            continue

        c = col.lower()
        base = None

        if "rmse" in c and 'nn' not in c and 'noise' not in c and 'quantile' not in c:
            base = "rmse"
        elif "nnrmse" in c and 'noise' not in c and 'quantile' not in c:
            base = "nnrmse"
        elif "mean" in c:
            base = "mean"
        elif "std" in c:
            base = "std"
        elif "var_integral" in c:
            base = "var_integral"
        elif col.endswith("_sobolf"):
            base = col
        elif col.endswith("_sobolt"):
            base = col
        else:
            continue

        groups.setdefault(base, []).append(col)

    return groups


# ------------------------------------------------------------
# Main plotting function
# ------------------------------------------------------------

def plot_sampling_strategy_comparison(
    strategy_csv_dict,
    output_dir="plots",
    true_values=None,
    plot_mode="aggregated",   # "aggregated" or "individual"
    num_samples=None
):
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------
    # Load all CSVs
    # --------------------------------------------------------
    dfs = []
    for strategy, path in strategy_csv_dict.items():
        df = pd.read_csv(path)
        df["strategy"] = strategy

        # Replace tiny values with NaN
        df = df.map(lambda v: np.nan if isinstance(v, (int, float)) and abs(v) < 1e-8 else v)

        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Normalized strategy names
    combined["strategy_norm"] = combined["strategy"].apply(normalize_strategy)

    # --------------------------------------------------------
    # SNAP num_samples to nearest natural increment per strategy family
    # --------------------------------------------------------
    combined["num_samples_snapped"] = np.nan

    if num_samples is None:
        raise ValueError('You must specify num_samples, it must be the same for each strategy for the aggregation to work')

    for strategy in combined["strategy"].unique():
        subset = combined[combined["strategy"] == strategy]
        values = subset["num_samples"].values

        if 'random' in strategy:
            n = 10000
            num_samples = np.array([5 * (i+1) for i in range(n)])
        else:
            n = 10000
            num_samples = np.array([5 + i for i in range(n)])

        combined.loc[subset.index, "num_samples_snapped"] = num_samples[0:len(values)].astype(int)

    # --------------------------------------------------------
    # Metric groups
    # --------------------------------------------------------
    metric_groups = group_metrics(combined.columns)

    # --------------------------------------------------------
    # Aggregation (mean + std)
    # --------------------------------------------------------
    agg_mean = (
        combined.groupby(["strategy_norm", "num_samples_snapped"])
        .mean(numeric_only=True)
        .reset_index()
    )

    agg_std = (
        combined.groupby(["strategy_norm", "num_samples_snapped"])
        .std(numeric_only=True)
        .reset_index()
    )

    agg_min = (
        combined.groupby(["strategy_norm", "num_samples_snapped"])
        .min(numeric_only=True)
        .reset_index()
    )

    # --------------------------------------------------------
    # Color map (one color per normalized strategy)
    # --------------------------------------------------------
    # Alphabetical, deterministic strategy order
    unique_norm = sorted(combined["strategy_norm"].unique())

    # Color‑blind‑safe Okabe–Ito palette
    okabe_ito = [
        "#000000", "#E69F00", "#56B4E9", "#009E73",
        "#CC79A7", "#0072B2", "#D55E00", "#F0E442",
    ]

    # Deterministic color assignment
    color_map = {
        strategy: okabe_ito[i % len(okabe_ito)]
        for i, strategy in enumerate(unique_norm)
    }

    # --------------------------------------------------------
    # Plotting
    # --------------------------------------------------------
    for base_metric, variants in metric_groups.items():
        metric = variants[0]  # representative column

        # Create figure with GridSpec: main plot + legend row
        fig = plt.figure(figsize=(3.35, 3.0))
        gs = GridSpec(2, 1, height_ratios=[4, 1], figure=fig)

        ax = fig.add_subplot(gs[0])
        if base_metric == 'rmse':
            ax.set_xlim(0, 50)
            ax.set_ylim(0.001, 0.01)
        ax_legend = fig.add_subplot(gs[1])
        ax_legend.axis("off")

        # True value (if provided)
        tv = true_values.get(base_metric, 0) if true_values else 0

        # ----------------------------------------------------
        # Find the individual model with the lowest RMSE after 50 points
        # ----------------------------------------------------
        if base_metric == 'rmse':
            best_strategy = None
            best_rmse_after_50 = np.inf
            for strategy in combined["strategy"].unique():
                df_s = combined[combined["strategy"] == strategy]
                if metric not in df_s.columns:
                    continue

                x = df_s["num_samples"].to_numpy()
                y = np.abs(df_s[metric].to_numpy() - tv)

                mask = (x >= 50) & ~np.isnan(y)
                if not np.any(mask):
                    continue

                candidate_min = np.min(y[mask])
                if candidate_min < best_rmse_after_50:
                    best_rmse_after_50 = candidate_min
                    best_strategy = strategy

            if best_strategy is not None:
                best_model_dir = os.path.dirname(strategy_csv_dict[best_strategy])
                best_model_path = os.path.join(output_dir, "ST_best_model.txt")
                with open(best_model_path, "w") as f:
                    f.write(best_model_dir + "\n")
                print(f"Best model after 50 points: {best_strategy} (rmse={best_rmse_after_50:.6g}) -> {best_model_path}")

        ax.minorticks_on()
        # Minor grid override (must be per-axis)
        ax.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.4)

        # ----------------------------------------------------
        # INDIVIDUAL MODE
        # ----------------------------------------------------
        best_rmse = np.inf
        if plot_mode == "individual":
            for strategy in combined["strategy"].unique():
                df_s = combined[combined["strategy"] == strategy]
                if metric not in df_s.columns:
                    continue

                x = df_s["num_samples"].to_numpy()
                y = np.abs(df_s[metric].to_numpy() - tv)

                mask = ~np.isnan(y)
                x, y = x[mask], y[mask]
                if base_metric == 'rmse':
                    if np.min(y) < best_rmse:
                        best_rmse = np.min(y)
                ax.plot(
                    x, y,
                    marker='o',
                    markersize=1,         # Smaller points
                    alpha=0.4,             # Slight transparency for both lines and markers
                    label=strategy,
                    color=color_map[normalize_strategy(strategy)],
                    linewidth=1.2,
                )

        # ----------------------------------------------------
        # AGGREGATED MODE (mean ± std)
        # ----------------------------------------------------
        elif plot_mode == "aggregated":
            for strategy in unique_norm:
                df_mean = agg_mean[agg_mean["strategy_norm"] == strategy]
                df_std = agg_std[agg_std["strategy_norm"] == strategy]
                df_min = agg_min[agg_min["strategy_norm"] == strategy]

                if metric not in df_mean.columns:
                    continue

                x = df_mean["num_samples_snapped"].to_numpy()
                y_mean = np.abs(df_mean[metric].to_numpy() - tv)
                y_std = df_std[metric].to_numpy()
                y_min = np.abs(df_min[metric].to_numpy() - tv)

                mask = ~np.isnan(y_mean)
                x, y_mean, y_std, y_min = x[mask], y_mean[mask], y_std[mask], y_min[mask]

                if 'random' in strategy:
                    # Mean line
                    ax.plot(
                        x, y_mean,
                        marker='o',
                        markersize=3,         # Smaller points
                        alpha=1,             # Slight transparency for both lines and markers
                        label=strategy,
                        color=color_map[normalize_strategy(strategy)],
                        linewidth=1.5,
                    )
                else:
                    ax.plot(
                        x, y_mean,
                        marker='o',
                        markersize=1,         # Smaller points
                        alpha=1,             # Slight transparency for both lines and markers
                        label=strategy,
                        color=color_map[normalize_strategy(strategy)],
                        linewidth=1.5,
                    )

                # Shaded region
                if base_metric == 'rmse':
                    # Upper bound: mean + std. Lower bound: minimum observed value.
                    lower_bound = y_min
                    upper_bound = y_mean + y_std
                else:
                    lower_bound = y_mean - y_std
                    upper_bound = y_mean + y_std

                ax.fill_between(
                    x,
                    lower_bound,
                    upper_bound,
                    color=color_map[strategy],
                    alpha=0.2,
                )

        # ----------------------------------------------------
        # Axis formatting
        # ----------------------------------------------------
        ax.set_xlabel("N. Train Points")
        if base_metric == 'rmse':
            ax.set_ylabel("RMSE, ST")
        else:
            ax.set_ylabel(base_metric)
        ax.tick_params(axis="both")

        # ----------------------------------------------------
        # Legend in bottom row
        # ----------------------------------------------------
        handles, labels = ax.get_legend_handles_labels()
        ax_legend.legend(
            handles, labels,
            loc="center",
            ncol=max(1, len(labels) // 2),
            frameon=False,
        )

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"ST_{base_metric}_{plot_mode}.png"), dpi=300)
        plt.close(fig)

    print(f"Plots saved to {output_dir}")


# ------------------------------------------------------------
# main()
# ------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(script_dir, "..", "shine_through_data_sets", "active_learning_comparison")
    plots_folder = os.path.join(script_dir, "plots")

    csv_files = find_csv_files(data_folder)
    if not csv_files:
        print("No batch_info.csv files found.")
        return

    strategy_csv_dict = {
        extract_sampling_strategy(f): f
        for f in csv_files
    }

    # Remove old keys
    for k in list(strategy_csv_dict.keys()):
        if "old" in k or "200_2" in k:
            strategy_csv_dict.pop(k)

    num_samples = np.array([5 + i for i in range(100000)])

    plot_sampling_strategy_comparison(
        strategy_csv_dict,
        output_dir=plots_folder,
        plot_mode="aggregated",   # "aggregated" or "individual"
        num_samples=num_samples
    )

    plot_sampling_strategy_comparison(
        strategy_csv_dict,
        output_dir=plots_folder,
        plot_mode="individual",   # "aggregated" or "individual"
        num_samples=num_samples
    )


if __name__ == "__main__":
    main()
