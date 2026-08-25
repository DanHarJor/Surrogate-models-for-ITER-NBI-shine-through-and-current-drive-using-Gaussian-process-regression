import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import scienceplots

plt.style.use(['science', 'no-latex'])

script_dir = os.path.dirname(os.path.abspath(__file__))
input_csv = os.path.join(script_dir, "..", "shine_through_data_sets", "current_drive_1d_demo", "1d_slice_current_drive.csv")
output_png = os.path.join(script_dir, "plots", "Figure_1D_Analysis.png")
base_run_dir = os.path.join(script_dir, "dummy_brd_1D_demo")

# ---------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------
def normalise(score):
    score = np.asarray(score).reshape(-1)
    smin, smax = score.min(), score.max()
    if smax > smin:
        return (score - smin) / (smax - smin)
    return np.zeros_like(score)

import matplotlib.patches as patches
def plot_1d_slice_statistics(input_file, target_col, output_col):
    # --- Load and Process Data ---
    df = pd.read_csv(input_file)
    stats = df.groupby(target_col)[output_col].agg(['mean', 'std', 'count']).reset_index()
    stats['sem'] = stats['std'] / np.sqrt(stats['count'])
    stats['se_sd'] = stats['std'] / np.sqrt(2 * (stats['count'] - 1))

    enbi_test, post_mean, post_std, pred_std, pred_se_std, vigf, eigf, bald, var = get_gpr_pred()
    post_std = np.abs(post_std)

    # --- Figure Layout ---
    # Width matches a full A4 text width (\textwidth ~ 6.5 in with standard margins)
    fig = plt.figure(figsize=(6.5, 3.0))
    gs_main = fig.add_gridspec(2, 1, height_ratios=[3, 1.2], hspace=0.35)

    gs_plots = gs_main[0].subgridspec(1, 2, wspace=0.45)
    ax0 = fig.add_subplot(gs_plots[0])
    ax1 = fig.add_subplot(gs_plots[1])

    gs_legs = gs_main[1].subgridspec(1, 2)
    ax_leg1 = fig.add_subplot(gs_legs[0]); ax_leg2 = fig.add_subplot(gs_legs[1])
    for a in [ax_leg1, ax_leg2]: a.axis('off')

    f_size, t_size = 8, 7

    # Defining the New Color Palette
    color_gpr_mean = '#4C72B0'  # Deep Blue
    color_stats_mean = '#DD8452' # Orange
    color_vigf = '#8172B2'      # Purple
    color_eigf = '#CC78BC'      # Pink/Magenta

    def draw_main_panel(ax):
        ax.scatter(df[target_col], df[output_col]/1e6, alpha=0.2, color='black', s=8, zorder=8)
        ax.errorbar(stats[target_col], stats['mean']/1e6, yerr=stats['sem']/1e6, fmt='_',
                    color=color_stats_mean, capsize=2, elinewidth=0.8, zorder=9)
        ax.plot(enbi_test, post_mean/1e6, color=color_gpr_mean, linewidth=1.2, zorder=1)
        ax.fill_between(enbi_test, post_mean/1e6 - post_std/1e6, post_mean/1e6 + post_std/1e6,
                        color=color_gpr_mean, alpha=0.1, zorder=1)

        ax.set_xlabel(r'$E_{NBI}$ [keV]', fontsize=f_size)
        ax.tick_params(labelsize=t_size)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
        ax.grid(True, which='major', linestyle=':', alpha=0.7, linewidth=0.7)
        ax.grid(True, which='minor', linestyle=':', alpha=0.15, linewidth=0.5)

    # 1. Zoomed Panel
    draw_main_panel(ax0)
    ax0.set_xlim(500, 550)
    ax0.set_ylim(52000/1e6, 65000/1e6)
    ax0.set_title('Zoomed Data', fontsize=f_size, fontweight='bold')
    ax0.set_ylabel('Current Drive [MA]', fontsize=f_size)

    # 2. Full Panel
    draw_main_panel(ax1)
    ax1.set_title('GPR model and Acq.', fontsize=f_size, fontweight='bold')
    ax1.indicate_inset_zoom(ax0, edgecolor="black", alpha=0.5, lw=1.0)
    ax1.set_xlim(490,880)
    # Twin axis for acquisition
    ax1_twin = ax1.twinx()
    ax1_twin.plot(enbi_test, vigf, color=color_vigf, linestyle='--', linewidth=1, label='VIGF')
    ax1_twin.plot(enbi_test, eigf, color=color_eigf, linestyle='-.', linewidth=1, label='EIGF')

    vigf_max_idx = np.argmax(vigf)
    ax1_twin.scatter(enbi_test[vigf_max_idx], vigf[vigf_max_idx], marker='*', color='gold',
                      s=60, edgecolor='black', linewidth=0.5, zorder=10)
    ax1_twin.set_ylabel('Acq. Value', fontsize=f_size, labelpad=5)
    ax1_twin.tick_params(labelsize=t_size)
    ax1_twin.grid(False, which='both')  # keep only the left-axis grid

    # RE-LAYERING: Move data to front
    ax1.set_zorder(ax1_twin.get_zorder() + 1)
    ax1.patch.set_visible(False)

    from matplotlib.legend_handler import HandlerErrorbar
    from matplotlib.container import ErrorbarContainer
    from matplotlib.lines import Line2D

    from matplotlib.legend_handler import HandlerErrorbar
    from matplotlib.container import ErrorbarContainer

    # 1. Create proxies that explicitly include the caps
    # We use fmt='_' to match your plot style (horizontal line at the mean)
    h_err_stats = ax1.errorbar([], [], yerr=[1], fmt='_', color=color_stats_mean,
                            capsize=3, elinewidth=1, capthick=1)

    # 2. Define the Legend Style and Handler
    # xerr_size=0.5 controls the width, yerr_size controls the height of the errorbar in the legend
    # marker_pad=0 ensures the mean line and the vertical bar align perfectly
    error_handler = HandlerErrorbar(xerr_size=0.5, yerr_size=0.5, marker_pad=0)

    leg_opt = {
        'frameon': False,
        'fontsize': 7,
        'loc': 'center',
        'handler_map': {ErrorbarContainer: error_handler}
    }

    # 3. Apply to Legends
    h_data = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=4),
        h_err_stats,
        Line2D([0], [0], color=color_gpr_mean, lw=1.2)
    ]

    ax_leg1.legend(h_data, ['Raw Data', 'Mean±seMean', 'GPR Pred±Unc '], **leg_opt)

    # 4. Legends with Updated Colors
    # h_data = [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=4),
    #           Line2D([0], [0], color=color_stats_mean, lw=1.2),
    #           Line2D([0], [0], color=color_gpr_mean, lw=1.2)]

    h_acq = [Line2D([0], [0], color=color_vigf, linestyle='--'),
             Line2D([0], [0], color=color_eigf, linestyle='-.')]

    leg_opt = {'frameon': False, 'fontsize': 7, 'loc': 'center'}
    # ax_leg1.legend(h_data, ['Raw', 'Mean±SEM', 'GPR Mean'], title=r"$\mathbf{Data\ Stats}$", title_fontsize=8, **leg_opt)
    ax_leg2.legend(h_acq, ['VIGF','EIGF'], ncol=2, **leg_opt)

    plt.savefig(output_png, bbox_inches='tight', dpi=600)
    plt.show()


def get_gpr_pred():
    from enchanted_surrogates.samplers.gpy_analytic_sobol_sampler import GpyAnalyticSobolSampler

    sampler_config = {
        "type": "gpy_analytic_sobol_sampler",
        "output_col": "current_drive_A",
        "initial_batch_size": 30,
        "batch_size": 30,
        "pool_csv_path": input_csv,
        "num_repeats": 10,
        "seed": 42,
        "acquisition_mode": "VIGF",
        "parameters": ["enbi", "nbar", "np", "hfactor"],
        "bounds": [
            [500, 870],
            [3.58, 11.94],
            [1, 1.5],
            [0.8, 1.5]
        ],
        "do_normalize_y": True,
        "budget": 30,
        "num_optimise_restarts": 20,
    }

    gpr = GpyAnalyticSobolSampler(**sampler_config)
    gpr.batch_number=1
    gpr.base_run_dir = base_run_dir
    gpr.fit()
    X_test = np.array([[i, 7.972076395987758, 1.1571065369897822, 1.165853608533115] for i in np.linspace(500, 870, 1000)])
    post_mean, post_std = gpr.surrogate_predict(X_test)
    pred_std, pred_se_std = gpr.predict_noise(X_test)
    enbi_test = X_test[:,0]

    X_pool = gpr.to_unit(X_test)
    vigf = normalise(gpr._compute_acquisition_unchunked(X_pool=X_pool,mode='vigf'))
    eigf = normalise(gpr._compute_acquisition_unchunked(X_pool=X_pool,mode='eigf'))
    bald = normalise(gpr._compute_acquisition_unchunked(X_pool=X_pool,mode='bald'))
    var = normalise(gpr._compute_acquisition_unchunked(X_pool=X_pool,mode='var'))

    return enbi_test, post_mean, post_std, pred_std, pred_se_std, vigf, eigf, bald, var


# --- Run Configuration ---
plot_1d_slice_statistics(
    input_file=input_csv,
    target_col='enbi',
    output_col='current_drive_A'
)
