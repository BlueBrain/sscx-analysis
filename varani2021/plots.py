"""
SSCx analysis plots (related to the analysis of Varani et al. 2021)
author: András Ecker, last update: 02.2022
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set(style="ticks", context="notebook")
RED = "#e32b14"


def plot_voltages(gids, t, voltages, fig_name):
    """Plots voltage traces"""
    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(len(gids), 1)
    for i in range(len(gids)):
        ax = fig.add_subplot(gs[i])
        ax.plot(t, voltages[i, :], "k-")
        ax.set_xlim([t[0], t[-1]])
        ax.set_ylabel("a%s" % gids[i])
    ax.set_xlabel("Time (ms)")
    sns.despine()
    fig.tight_layout()
    fig.savefig(fig_name, bbox_inches="tight", dpi=100)
    plt.close(fig)


def plot_all_voltages(v_spiking, v_subtreshold, rate, t_start, t_end, fig_name):
    """Plots voltage collages of spiking and non-spiking cells"""
    v_subtreshold_mean = np.mean(v_subtreshold, axis=0)
    t_v = np.linspace(t_start, t_end, len(v_subtreshold_mean))
    t_rate = np.linspace(t_start, t_end, len(rate))
    xticks = np.linspace(t_start, t_end, 6).astype(int)
    non_spiking_pct = (v_subtreshold.shape[0] / (v_spiking.shape[0] + v_subtreshold.shape[0])) * 100

    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 4, 15, 1], width_ratios=[69, 1])
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t_rate, rate, color=RED)
    ax.fill_between(t_rate, np.zeros_like(t_rate), rate, color=RED, alpha=0.1)
    ax.set_xticks(xticks)
    ax.set_xlim(t_start, t_end)
    ax.set_ylabel("Rate (Hz)")
    sns.despine(ax=ax)
    ax2 = fig.add_subplot(gs[1, 0])
    i2 = ax2.imshow(v_spiking, cmap="inferno", aspect="auto", origin="lower")
    plt.colorbar(i2, cax=fig.add_subplot(gs[1, 1]))
    ax2.set_ylabel("Spiking gids")
    ax2.set_xticks(np.linspace(0, v_spiking.shape[1], 6).astype(int))
    ax2.set_xticklabels(xticks)
    ax3 = fig.add_subplot(gs[2, 0])
    i3 = ax3.imshow(v_subtreshold, cmap="inferno", aspect="auto", origin="lower")
    cbar = plt.colorbar(i3, cax=fig.add_subplot(gs[2, 1]))
    cbar.set_label("Voltage (mV)")
    ax3.set_ylabel("Non-spiking gids (%.1f%%)" % non_spiking_pct)
    ax3.set_xticks(np.linspace(0, v_subtreshold.shape[1], 6).astype(int))
    ax3.set_xticklabels(xticks)
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.plot(t_v, v_subtreshold_mean, "k-")
    ax4.set_xticks(xticks)
    ax4.set_xlim(t_start, t_end)
    ax4.set_xlabel("Time (ms)")
    ax4.set_ylabel("V (mV)")
    ax4.set_yticks([np.floor(np.mean(v_subtreshold_mean))])
    sns.despine(ax=ax4)
    fig.align_ylabels()
    gs.tight_layout(fig, h_pad=0.2, w_pad=0.2)
    fig.savefig(fig_name, bbox_inches="tight", dpi=100)
    plt.close(fig)
    
    
def plot_epsps_amplitudes(epsps, fig_name):
    """Plots EPSP amplitudes of non-spiking cells"""
    range = [np.min(epsps), np.max(epsps)]
    fig = plt.figure(figsize=(10, 6.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(epsps, bins=30, range=range, color="gray")
    ax.set_xlim([0, range[1]])
    ax.set_xlabel("EPSP amplitude (mV)")
    sns.despine(offset=5, trim=True)
    fig.savefig(fig_name, bbox_inches="tight", dpi=100)
    plt.close(fig)

