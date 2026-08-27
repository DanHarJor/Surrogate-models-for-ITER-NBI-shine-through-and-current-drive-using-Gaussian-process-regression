"""Diagnostic plots for the best current-drive (CD) active-learning surrogate
(selected by CD_active_learning_comparison.py, see
plotting_scripts/plots/CD_best_model.txt).

Loads the persisted GPy model + noise model and produces:
 - 2D slice plots of the predicted output and its predicted noise
 - residuals / true-vs-predicted plots against the held-out Sobol-sequence
   test set, collapsed to the mean of each point's 5 repeats before taking
   the residual.
"""
import os

import numpy as np
from enchanted_surrogates.samplers.gpy_analytic_sobol_sampler import GpyAnalyticSobolSampler

import scienceplots
import matplotlib.pyplot as plt

plt.style.use(["science", "no-latex"])
plt.rcParams["axes.grid"] = False


def restore_noise_normalization(gpy):
    """`load_noise_model` restores the fitted noise GP, but its target
    normalization (mean/std of the training points' measured noise std,
    computed in `fit_noise`) isn't persisted to disk. Recompute those two
    scalars deterministically from the training data so `predict_noise`
    can de-normalize the loaded noise GP's output, without re-optimizing
    the noise GP itself (which is already fitted and loaded from disk).
    """
    _, _, noise_vars, _, _, counts = gpy._get_unitXY_with_noise()
    std, std_err = gpy.var_to_std(noise_vars, counts)
    mask = np.isfinite(std) & np.isfinite(std_err) & (std > 0)
    gpy._noise_mean = float(np.mean(std[mask]))
    gpy._noise_std = float(np.std(std[mask])) or 1.0

script_dir = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(script_dir, "..", "current_drive_data_sets")

best_model_dir = os.path.join(script_dir, "best_model_dir_CD")
out_dir = os.path.join(script_dir, "plots", "CD_best_model")
os.makedirs(out_dir, exist_ok=True)

sampler_config = {
    "type": "gpy_analytic_sobol_sampler",
    "output_col": "current_drive_A",
    "output_name": "CD [MA]",
    "long_output_name": "Current Drive [MA]",
    "output_scale": 1e-6,
    "test_data_csv": os.path.join(data_root, "sobol_seq", "enchanted_dataset.csv"),
    "pool_csv_path": os.path.join(data_root, "random", "enchanted_dataset.csv"),
    "write_batch_info_every_x_samples": 50,
    "initial_batch_size": 5,
    "batch_size": 1,
    "num_repeats": 5,
    "acquisition_mode": "var",
    "parameters": ["enbi", "nbar", "np", "hfactor"],
    "parameters_labels": {
        "enbi": r"$E_\text{NBI}$",
        "nbar": r"$\bar n$",
        "np": r"$n_{\text{pf}}$",
        "hfactor": r"$H_{98}$",
    },
    "bounds": [[500, 870], [3.58, 11.94], [1, 1.5], [0.8, 1.5]],
    "do_normalize_y": True,
    "budget": 1000,
}

gpy = GpyAnalyticSobolSampler(**sampler_config)
gpy.base_run_dir = best_model_dir
gpy.set_standardisation_params()
gpy.load_model(directory=best_model_dir)
gpy.load_noise_model(directory=best_model_dir)
restore_noise_normalization(gpy)

# Output-vs-input slices (and their predicted-noise counterpart)
gpy.plot_slices(out_dir=out_dir)

# Residuals / true-vs-predicted against the Sobol-sequence test set. Test
# points are collapsed to the mean over their 5 repeats (via
# GpyAnalyticSobolSampler._collapse_data) before residuals are computed.
gpy.residuals_plot(out_dir=out_dir)

print(f"Best-model plots saved to {out_dir}")
