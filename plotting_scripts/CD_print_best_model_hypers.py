"""Print GPy kernel hyperparameters for the best current-drive (CD)
active-learning surrogate (selected by CD_active_learning_comparison.py,
persisted in plotting_scripts/best_model_dir_CD/).

For both the output model (gpy_model.pkl) and the noise model
(gpy_noise_model.pkl), prints the ARD lengthscale for each input dimension
and the RBF kernel's variance (amplitude) hyperparameter.
"""
import os
import pickle

import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
best_model_dir = os.path.join(script_dir, "best_model_dir_CD")
out_dir = os.path.join(script_dir, "plots", "CD_best_model")
out_path = os.path.join(out_dir, "kernel_hyperparameters.txt")

parameters = ["enbi", "nbar", "np", "hfactor"]


def format_kernel_hypers(model_path, label):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    kern = model.kern
    lengthscales = np.atleast_1d(kern.lengthscale.values)
    variance = float(np.atleast_1d(kern.variance.values)[0])

    lines = [f"{label} ({os.path.basename(model_path)})"]
    names = parameters if len(parameters) == len(lengthscales) else range(len(lengthscales))
    for name, ls in zip(names, lengthscales):
        lines.append(f"  lengthscale[{name}] = {ls:.6g}")
    lines.append(f"  amplitude (kernel variance) = {variance:.6g}")
    return lines


if __name__ == "__main__":
    os.makedirs(out_dir, exist_ok=True)

    lines = []
    lines += format_kernel_hypers(os.path.join(best_model_dir, "gpy_model.pkl"), "Output model")
    lines.append("")
    lines += format_kernel_hypers(os.path.join(best_model_dir, "gpy_noise_model.pkl"), "Noise model")

    text = "\n".join(lines)
    print(text)

    with open(out_path, "w") as f:
        f.write(text + "\n")
    print(f"\nSaved to {out_path}")
