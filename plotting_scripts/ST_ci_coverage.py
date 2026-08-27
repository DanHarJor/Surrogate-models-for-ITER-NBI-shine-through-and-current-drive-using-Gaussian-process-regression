"""CI-coverage diagnostics for the best shine-through (ST) active-learning
surrogate (selected by ST_active_learning_comparison.py, persisted in
plotting_scripts/best_model_dir_ST/).

Checks how well the model's predictive uncertainty is calibrated against
the held-out Sobol-sequence test set, under two definitions of predictive
std:
 - "test_sem": epistemic std (predict_noiseless) plus the test set's own
   per-point standard error of the mean (from its repeats).
 - "scaled_epistemic_sem": the same, but with the epistemic std rescaled by
   a fitted factor lambda so that exactly ci_pct% of test points fall
   inside the ci_pct% CI (see fit_epistemic_scale_lambda). Writes
   ci_summary.txt reporting the fitted lambda.
"""
import os

import numpy as np
import pandas as pd
from enchanted_surrogates.samplers.gpy_analytic_sobol_sampler import (
    GpyAnalyticSobolSampler,
    okabe_ito,
)

import scienceplots
import matplotlib.pyplot as plt

plt.style.use(["science", "no-latex"])
plt.rcParams["axes.grid"] = False


def restore_noise_normalization(gpy):
    """See ST_make_best_model_plots.py: recompute the noise GP's target
    normalization, which isn't persisted to disk."""
    _, _, noise_vars, _, _, counts = gpy._get_unitXY_with_noise()
    std, std_err = gpy.var_to_std(noise_vars, counts)
    mask = np.isfinite(std) & np.isfinite(std_err) & (std > 0)
    gpy._noise_mean = float(np.mean(std[mask]))
    gpy._noise_std = float(np.std(std[mask])) or 1.0


def fit_epistemic_scale_lambda(epistemic_std, se_mean, abs_error, z, target_coverage):
    """Smallest multiplicative scale factor lambda such that
    z*sqrt((lambda*epistemic_std)^2 + se_mean^2) covers at least
    `target_coverage` fraction of the test-set residuals `abs_error`.

    Per test point i, the minimum lambda needed for that point to fall
    inside the CI solves z*sqrt((lambda*epistemic_i)^2 + se_mean_i^2) =
    abs_error_i, i.e. lambda_i = sqrt(max(0, (abs_error_i/z)^2 -
    se_mean_i^2)) / epistemic_i (0 if the point is already covered at
    lambda=0). Setting lambda to the target_coverage-th percentile of these
    per-point thresholds is the smallest lambda achieving at least that
    fraction of points inside the CI — the same logic as conformal
    calibration.
    """
    needed_var = np.maximum((abs_error / z) ** 2 - se_mean**2, 0.0)
    lambda_needed = np.sqrt(needed_var) / np.maximum(epistemic_std, 1e-300)
    return float(np.percentile(lambda_needed, target_coverage * 100))


def write_ci_summary_report(gpy, lam, frac_inside, out_path, z, ci_pct, n):
    with open(out_path, "w") as f:
        f.write("CI coverage summary (lambda * predict_noiseless + test-set SEM)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Output: {gpy.output_name}\n")
        f.write(f"Test set size (n): {n}\n")
        f.write(f"Confidence level: {ci_pct}% (z={z})\n\n")
        f.write(f"Fitted epistemic scale factor (lambda): {lam:.6g}\n\n")
        f.write(f"Test points inside {ci_pct}% CI (using lambda*predict_noiseless + test-set SEM): {frac_inside:.1f}%\n")

    print("saving CI summary report to:", out_path)


def _plot_ci_histogram(y_true, y_pred, y_pred_std, output_name, output_scale, out_path, title, z, ci_pct):
    """Histogram of (z*sigma) - |prediction error|. Values > 0 mean the
    prediction error falls inside the CI; values < 0 mean it was violated."""
    abs_error = np.abs(y_pred - y_true) * output_scale
    ci_halfwidth = z * y_pred_std * output_scale
    margin = ci_halfwidth - abs_error

    frac_inside = np.mean(margin > 0) * 100

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.hist(margin, bins=30, color=okabe_ito[5], edgecolor="black", linewidth=0.5)
    ax.axvline(0, color=okabe_ito[6], linestyle="--", linewidth=1.5, label="Error = CI bound")
    ax.set_xlabel(f"{output_name}: {z}$\\sigma$ $-$ |Pred $-$ Test|")
    ax.set_ylabel("Count")
    ax.set_title(f"{title}\n({frac_inside:.1f}% inside {ci_pct}% CI)")
    ax.legend()
    ax.minorticks_on()
    ax.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.4)
    fig.tight_layout()

    print("saving CI coverage histogram to:", out_path)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")

    return fig, frac_inside


def plot_ci_coverage_histogram(gpy, out_dir, noise_mode, z=1.96, ci_pct=95):
    """CI-coverage histogram on the held-out test_data_csv.

    `noise_mode`:
      - "test_sem": adds the *test* set's own per-point standard error of
        the mean (from its repeats) to the epistemic std. y_test_unique is
        a finite-sample mean, not the true function value, so its own
        sampling uncertainty (SEM = sigma/sqrt(n_repeats)) is unavoidably
        present in every residual computed against it.
      - "scaled_epistemic_sem": lambda*predict_noiseless + test-set SEM,
        where lambda is fit so that exactly ci_pct% of test points fall
        inside the ci_pct% CI. Writes ci_summary.txt.
    """
    os.makedirs(out_dir, exist_ok=True)

    test_df = pd.read_csv(gpy.test_data_csv)
    out_col = gpy.get_output_col(df=test_df)

    X_test = test_df[gpy.parameters].values
    y_test = test_df[out_col].values

    X_test_unique, y_test_unique, noise_vars, _, se_mean, _ = gpy._collapse_data(X_test, y_test)
    mask = noise_vars != 0
    X_test_unique, y_test_unique, se_mean = X_test_unique[mask], y_test_unique[mask], se_mean[mask]

    if noise_mode == "test_sem":
        y_pred, epistemic_std = gpy.surrogate_predict(X_test_unique)
        y_pred_std = np.sqrt(epistemic_std**2 + se_mean**2)
        tag, title = "test_sem", "CI coverage check (predict_noiseless + test-set SEM)"
    elif noise_mode == "scaled_epistemic_sem":
        y_pred, epistemic_std = gpy.surrogate_predict(X_test_unique)
        abs_error = np.abs(y_pred - y_test_unique)
        lam = fit_epistemic_scale_lambda(epistemic_std, se_mean, abs_error, z, ci_pct / 100)
        y_pred_std = np.sqrt((lam * epistemic_std) ** 2 + se_mean**2)
        tag = "scaled_epistemic_sem"
        title = f"CI coverage check ($\\lambda$={lam:.3g} $\\times$ predict_noiseless + test-set SEM)"
    else:
        raise ValueError(f"Unsupported noise_mode: {noise_mode}")

    n = X_test_unique.shape[0]
    out_path = os.path.join(out_dir, f"ci_coverage_hist_{ci_pct}_{tag}-{n}.png")

    fig, frac_inside = _plot_ci_histogram(
        y_test_unique, y_pred, y_pred_std, gpy.output_name, gpy.output_scale, out_path, title, z=z, ci_pct=ci_pct
    )

    if noise_mode == "scaled_epistemic_sem":
        report_path = os.path.join(out_dir, "ci_summary.txt")
        write_ci_summary_report(gpy, lam, frac_inside, report_path, z, ci_pct, n)

    return fig


script_dir = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(script_dir, "..", "shine_through_data_sets")
best_model_dir = os.path.join(script_dir, "best_model_dir_ST")
out_dir = os.path.join(script_dir, "plots", "ST_best_model")

sampler_config = {
    "type": "gpy_analytic_sobol_sampler",
    "output_col": "output_shine_inj1",
    "output_name": "ST",
    "test_data_csv": os.path.join(data_root, "sobol_seq", "maxST0.3_filtered.csv"),
    "pool_csv_path": os.path.join(data_root, "random", "maxST0.3_filtered.csv"),
    "write_batch_info_every_x_samples": 5,
    "initial_batch_size": 5,
    "batch_size": 1,
    "num_repeats": 5,
    "acquisition_mode": "vigf",
    "parameters": ["enbi", "nbar", "np", "hfactor"],
    "parameters_labels": {
        "enbi": r"$E_\text{NBI}$",
        "nbar": r"$\bar n$",
        "np": r"$n_{\text{pf}}$",
        "hfactor": r"$H_{98}$",
    },
    "bounds": [[500, 1000], [0.6, 5], [1, 1.5], [0.8, 1.5]],
    "do_normalize_y": True,
    "budget": 500,
}

if __name__ == "__main__":
    gpy = GpyAnalyticSobolSampler(**sampler_config)
    gpy.base_run_dir = best_model_dir
    gpy.set_standardisation_params()
    gpy.load_model(directory=best_model_dir)
    gpy.load_noise_model(directory=best_model_dir)
    restore_noise_normalization(gpy)

    plot_ci_coverage_histogram(gpy, out_dir=out_dir, noise_mode="test_sem", z=1.96, ci_pct=95)
    plot_ci_coverage_histogram(gpy, out_dir=out_dir, noise_mode="scaled_epistemic_sem", z=1.96, ci_pct=95)

    print(f"CI-coverage diagnostics saved to {out_dir}")
