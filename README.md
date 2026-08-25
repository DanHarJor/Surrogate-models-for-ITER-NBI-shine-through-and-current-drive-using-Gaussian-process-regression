# Surrogate-models-for-ITER-NBI-shine-through-and-current-drive-using-Gaussian-process-regression
This is the repo accompanying the paper:

Link to paper

The repo includes

 - The datasets generated
 - The GPR models trained
 - Plotting scripts for the plots in the paper

## Cloning

This repo uses [enchanted-surrogates](https://github.com/DIGIfusion/enchanted-surrogates) as a git submodule (pinned to its `ascot_ai_branch` branch), so clone with submodules:

```
git clone --recurse-submodules <this-repo-url>
```

If you already cloned without `--recurse-submodules`, initialize it after the fact:

```
git submodule update --init --recursive
```

To pull in the latest `ascot_ai_branch` changes later:

```
git submodule update --remote enchanted-surrogates
```

## Environment setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e ./enchanted-surrogates
```
