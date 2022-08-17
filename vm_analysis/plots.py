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


def plot_vm_dist_spect(v, mean, std, spiking, f, pxx, coeffs, freq_window, fig_name):
    """Plots V_m's distribution and power spectrum
    (`mean`, `std`, and `spiking` could be easily computed)"""
    fig = plt.figure(figsize=(10, 6.5))
    ax = fig.add_subplot(1, 2, 1)
    col = "red" if spiking else "blue"
    ax.hist(v[v < -55], bins=30, color=col, label="%.2f+/-%.2f" % (mean, std))  # -55 is a rather arbitrary threshold
    # ax.set_xlim([-80, -50])
    ax.set_xlabel("V_m (mV)")
    ax.legend(frameon=False)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(f, pxx, color="black")
    if not spiking:
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
    
    
def plot_heatmap_grid(df, fig_name):
    """Plots heatmaps on grid (row and col as extra vars. on top of mean and std.)
    (-> used for shot noise, which has ampCV and tau atm.)"""
    vmin, vmax = df["V_mean"].min(), df["V_mean"].max()
    amp_cvs = df["amp_cv"].unique()
    taus = df["tau"].unique()
    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(len(taus), len(amp_cvs)+1, width_ratios=[10 for _ in range(len(amp_cvs))] + [1])
    for i, tau in enumerate(taus):
        for j, amp_cv in enumerate(amp_cvs):
            ax = fig.add_subplot(gs[i, j])
            df_tmp = df.loc[(df["amp_cv"] == amp_cv) & (df["tau"] == tau)].pivot(
                     index="std", columns="mean", values="V_mean")
            if i == 0 and j == 0:
                sns.heatmap(df_tmp, cmap="viridis", vmin=vmin, vmax=vmax, ax=ax,
                            cbar_ax=fig.add_subplot(gs[:, -1]), cbar_kws={"label": "mean V_m (mV)"})
            else:
                sns.heatmap(df_tmp, cmap="viridis", vmin=vmin, vmax=vmax, cbar=False, ax=ax)
            if i == 0:
                ax.set_title("ampCV = %.2f" % amp_cv)
            if i != len(taus) - 1:
                ax.set_xlabel("")
            if j == 0:
                ax.set_ylabel("tau = %.1f\nstd" % tau)
            else:
                ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(fig_name, bbox_inches="tight", dpi=300)
    plt.close(fig)

