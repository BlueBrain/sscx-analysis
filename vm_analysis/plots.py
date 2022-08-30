"""
V_m analysis related plots
author: András Ecker, last update: 08.2022
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set(style="ticks", context="notebook")


def plot_vm_dist_spect(v, mean, std, rate, f, pxx, coeffs, freq_window, fig_name):
    """Plots V_m's distribution and power spectrum
    (`mean`, `std`, and `spiking` could be easily computed)"""
    fig = plt.figure(figsize=(10, 6.5))
    ax = fig.add_subplot(1, 2, 1)
    col = "red" if rate > 0. else "blue"
    ax.hist(v[v < -55], bins=30, color=col, label="%.2f+/-%.2f" % (mean, std))  # -55 is a rather arbitrary threshold
    # ax.set_xlim([-80, -50])
    ax.set_xlabel("V_m (mV)")
    ax.legend(frameon=False)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(f, pxx, color="black")
    if rate == 0.:
        idx = np.where((freq_window[0] < f) & (f < freq_window[1]))[0]
        fit = np.polyval(coeffs, np.log10(f[idx]))
        ax2.plot(f[idx], 10**fit, color="red", label="alpha=%.2f" % np.abs(coeffs[0]))
        ax2.legend(frameon=False)
    plt.xscale("log")
    plt.yscale("log")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("PSD (V^2/Hz)")
    sns.despine()
    fig.tight_layout()
    fig.savefig(fig_name, bbox_inches="tight", dpi=100)
    plt.close(fig)

    
def plot_heatmap_grid(df, var, row_var, col_var, annot_var, fig_name):
    """Plots annotated heatmaps on a grid (row and col as extra vars. on top of mean and std.)"""
    vmin, vmax = df[var].min(), df[var].max()
    rows, cols = df[row_var].unique(), df[col_var].unique()
    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(len(rows), len(cols)+1, width_ratios=[10 for _ in range(len(cols))] + [1])
    for i, row_val in enumerate(rows):
        for j, col_val in enumerate(cols):
            ax = fig.add_subplot(gs[i, j])
            idx = df.loc[(df[col_var] == col_val) & (df[row_var] == row_val)].index
            df_plot = df.loc[idx].pivot(index="std", columns="mean", values=var)
            df_annot = df.loc[idx].pivot(index="std", columns="mean", values="rate")
            show_annots = df_annot.to_numpy() > 0.
            if i == 0 and j == 0:
                sns.heatmap(df_plot, cmap="viridis", vmin=vmin, vmax=vmax, annot=df_annot, fmt=".1f", ax=ax,
                            cbar_ax=fig.add_subplot(gs[:, -1]), cbar_kws={"label": var})
            else:
                sns.heatmap(df_plot, cmap="viridis", vmin=vmin, vmax=vmax, annot=df_annot, fmt=".1f", cbar=False, ax=ax)
            for text, show_annot in zip(ax.texts, (element for row in show_annots for element in row)):
                text.set_visible(show_annot)
            if i == 0:
                col_val = col_val if isinstance(col_val, str) else "%.2f" % col_val
                ax.set_title("%s = %s" % (col_var, col_val))
            if i != len(rows) - 1:
                ax.set_xlabel("")
            if j == 0:
                row_val = row_val if isinstance(row_val, str) else "%.2f" % row_val
                ax.set_ylabel("%s = %s\nstd" % (row_var, row_val))
            else:
                ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(fig_name, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_corrs(df, var_cols, value_cols, hue, size, fig_name):
    """Plot correlations for input variables and output features"""
    if hue is not None:
        grid = sns.PairGrid(data=df, hue=hue, x_vars=var_cols, y_vars=value_cols)
    else:
        grid = sns.PairGrid(data=df, x_vars=var_cols, y_vars=value_cols)
    grid.fig.set_size_inches(10, 6.5)
    if size is not None:
        grid.map(sns.scatterplot, size=df[size])
    else:
        grid.map(sns.kdeplot, bw_method="silverman")
    if hue is not None:
        if size is not None:
            grid.add_legend(title="", adjust_subtitles=True)
        else:
            grid.add_legend(frameon=False)
    grid.tight_layout()
    grid.savefig(fig_name, dpi=100, bbox_inches="tight")