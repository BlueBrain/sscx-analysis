"""
SSCx analysis plots (related to the analysis of Varani et al. 2021)
author: András Ecker, last update: 02.2022
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import seaborn as sns
from utils import calc_rate

sns.set(style="ticks", context="notebook")
# these 4 colors are copied from `prepare_raster_asth.py`
RED = "#e32b14"
BLUE = "#3271b8"
GREEN = "#67b32e"
ORANGE = "#c9a021"
legend_handles = [mlines.Line2D([], [], color=RED, marker="s", linestyle='None', markersize=10, label="PC"),
                  mlines.Line2D([], [], color=BLUE, marker="s", linestyle='None', markersize=10, label="PV"),
                  mlines.Line2D([], [], color=GREEN, marker="s", linestyle='None', markersize=10, label="Sst"),
                  mlines.Line2D([], [], color=ORANGE, marker="s", linestyle='None', markersize=10, label="5HT3aR")]
PROJ_COLORS = {"VPM": "#4a4657", "POm": "#442e8a"}


def setup_raster(spike_times, spiking_gids, ys, colors, groups, types, type_colors, t_start, t_end):
    """Converts array of gids to array of y coordinates and colors and calculates populational firing rates
    (neurons are grouped in `prepare_raster_asth.py`)
    :params ys, colors, groups: look ups (dicts) with gids as keys, y-coordinates, colors, and cell types as values"""

    unique_gids, idx = np.unique(spiking_gids, return_inverse=True)
    print("%i spikes from %i neurons" % (len(spiking_gids), len(unique_gids)))
    unique_ys = np.zeros_like(unique_gids, dtype=np.int64)
    unique_cols = np.empty(unique_gids.shape, dtype=object)
    Ns = {type_: 0 for type_ in types}  # counts for rate normalization
    for i, gid in enumerate(unique_gids):
        unique_ys[i] = ys[gid]
        unique_cols[i] = colors[gid]
        Ns[groups[gid]] += 1
    spiking_ys = unique_ys[idx]
    cols = unique_cols[idx]

    rates_dict = {}
    for type_, type_color in zip(types, type_colors):
        rates_dict[type_] = calc_rate(spike_times[cols == type_color], Ns[type_], t_start, t_end)

    return spiking_ys, cols, rates_dict


def plot_raster(spike_times, spiking_gids, proj_rate_dict, asthetics, t_start, t_end, fig_name):
    """Plots raster"""
    spiking_ys, cols, rates_dict = setup_raster(spike_times, spiking_gids,
                                                asthetics["ys"], asthetics["colors"], asthetics["groups"],
                                                asthetics["types"], asthetics["type_colors"], t_start, t_end)
    if proj_rate_dict is None:
        fig = plt.figure(figsize=(20, 9))
        gs = gridspec.GridSpec(3, 1, height_ratios=[15, 1, 1])
    else:
        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(4, 1, height_ratios=[14, 1, 1, 1])
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor((0.95, 0.95, 0.95))
    ax.scatter(spike_times, spiking_ys, c=cols, alpha=0.9, marker='.', s=2., edgecolor="none")
    ax.set_xlim([t_start, t_end])
    ax.set_ylim([np.max(asthetics["yticks"]), np.min(asthetics["yticks"])])
    ax.set_yticks(asthetics["yticks"])
    ax.set_yticklabels(asthetics["yticklabels"])
    ax.set_ylabel("Cortical depth (um)")
    ax.legend(handles=legend_handles, ncol=4, frameon=False, bbox_to_anchor=(0.77, 1.))
    ax2 = fig.add_subplot(gs[1])
    sns.despine(ax=ax2)
    t_rate = np.linspace(t_start, t_end, len(rates_dict[asthetics["types"][0]]))
    ax2.plot(t_rate, rates_dict["PC"], color=asthetics["type_colors"][0])  # label="PC")
    ax2.fill_between(t_rate, np.zeros_like(t_rate), rates_dict["PC"], color=asthetics["type_colors"][0], alpha=0.1)
    ax2.set_xlim([t_start, t_end])
    ax2.set_ylabel("PC Rate\n(Hz)")
    # ax2.set_ylim(bottom=0)
    # ax2.legend(frameon=False, loc=1)
    ax3 = fig.add_subplot(gs[2])
    sns.despine(ax=ax3)
    for type_, col in zip(asthetics["types"][1:], asthetics["type_colors"][1:]):
        ax3.plot(t_rate, rates_dict[type_], color=col)  # label=type_
        ax3.fill_between(t_rate, np.zeros_like(t_rate), rates_dict[type_], color=col, alpha=0.1)
    ax3.set_xlim([t_start, t_end])
    ax3.set_ylabel("IN Rate\n(Hz)")
    # ax3.set_ylim(bottom=0)
    # ax3.legend(frameon=False, ncol=3, loc=1)
    if proj_rate_dict is None:
        ax3.set_xlabel("Time (ms)")
        fig.align_ylabels([ax2, ax3])
    else:
        ax4 = fig.add_subplot(gs[3])
        sns.despine(ax=ax4)
        for i, (proj_name, proj_rate) in enumerate(proj_rate_dict.items()):
            ax4.plot(t_rate, proj_rate, color=PROJ_COLORS[proj_name], label=proj_name)
            ax4.fill_between(t_rate, np.zeros_like(t_rate), proj_rate, color=PROJ_COLORS[proj_name], alpha=0.1)
        ax4.legend(ncol=i+1, frameon=False, loc=1)
        ax4.set_xlim([t_start, t_end])
        ax4.set_ylabel("TC Rate\n(Hz)")
        ax4.set_xlabel("Time (ms)")
        fig.align_ylabels([ax2, ax3, ax4])
    gs.tight_layout(fig, h_pad=0.2)
    fig.savefig(fig_name, bbox_inches="tight", dpi=100)
    plt.close(fig)


def plot_dvdt(t, v, fig_name):
    """Plots voltage trace and its derivative"""
    fig = plt.figure(figsize=(10, 6.5))
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(t, v, "k-")
    ax.set_xlim([t[0], t[-1]])
    ax.set_ylabel("V (mV)")
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(t, np.gradient(v, t[1] - t[0]), "r-")
    ax2.set_xlabel("Time (ms)")
    ax2.set_xlim([t[0], t[-1]])
    ax2.set_ylabel("dV/dt")
    sns.despine()
    fig.tight_layout()
    fig.savefig(fig_name, bbox_inches="tight", dpi=100)
    plt.close(fig)


def plot_all_voltages(rate, v_spiking, v_subtreshold, t_start, t_end, fig_name):
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
    ax.set_xlim([t_start, t_end])
    ax.set_ylabel("Rate (Hz)")
    sns.despine(ax=ax)
    ax2 = fig.add_subplot(gs[1, 0])
    i2 = ax2.imshow(v_spiking, cmap="inferno", aspect="auto", origin="lower")
    plt.colorbar(i2, cax=fig.add_subplot(gs[1, 1]))
    ax2.set_ylabel("Spiking gids")
    ax2.set_xticks(np.linspace(-0.5, v_spiking.shape[1]-0.5, 6))
    ax2.set_xticklabels(xticks)
    ax3 = fig.add_subplot(gs[2, 0])
    i3 = ax3.imshow(v_subtreshold, cmap="inferno", aspect="auto", origin="lower")
    cbar = plt.colorbar(i3, cax=fig.add_subplot(gs[2, 1]))
    cbar.set_label("Voltage (mV)")
    ax3.set_ylabel("Non-spiking gids (%.1f%%)" % non_spiking_pct)
    ax3.set_xticks(np.linspace(-0.5, v_subtreshold.shape[1]-0.5, 6))
    ax3.set_xticklabels(xticks)
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.plot(t_v, v_subtreshold_mean, "k-")
    ax4.set_xticks(xticks)
    ax4.set_xlim([t_start, t_end])
    ax4.set_xlabel("Time (ms)")
    ax4.set_ylabel("V (mV)")
    ax4.set_yticks([np.floor(np.mean(v_subtreshold_mean))])
    sns.despine(ax=ax4)
    fig.align_ylabels()
    gs.tight_layout(fig, h_pad=0.2, w_pad=0.2)
    fig.savefig(fig_name, bbox_inches="tight", dpi=100)
    plt.close(fig)


def plot_selected_voltages(v_subtreshold, t_start, t_end, fig_name):
    """Same as the bottom part of `plot_all_voltages` above..."""
    v_subtreshold_mean = np.mean(v_subtreshold, axis=0)
    t_v = np.linspace(t_start, t_end, len(v_subtreshold_mean))
    xticks = np.linspace(t_start, t_end, 6).astype(int)

    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(2, 2, height_ratios=[5, 1], width_ratios=[69, 1])
    ax = fig.add_subplot(gs[0, 0])
    i = ax.imshow(v_subtreshold, cmap="inferno", aspect="auto", origin="lower")
    plt.colorbar(i, cax=fig.add_subplot(gs[0, 1]))
    ax.set_xticks(np.linspace(-0.5, v_subtreshold.shape[1] - 0.5, 6))
    ax.set_xticklabels(xticks)
    ax.set_ylabel("Responsive non-spiking gids")
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t_v, v_subtreshold_mean, "k-")
    ax2.set_xticks(xticks)
    ax2.set_xlim([t_start, t_end])
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("V (mV)")
    sns.despine(ax=ax2, trim=True)
    fig.align_ylabels()
    gs.tight_layout(fig, h_pad=0.2, w_pad=0.2)
    fig.savefig(fig_name, bbox_inches="tight", dpi=100)
    plt.close(fig)


def plot_mean_voltages(t, mean_ctrl_voltage, mean_opto_voltages, fig_name):
    """Plot mean voltages at different opto. depol. percents"""
    fig = plt.figure(figsize=(10, 6.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t, mean_ctrl_voltage, "k-", label="control")
    opto_depol_pcts = np.sort(list(mean_opto_voltages.keys()))
    cm = sns.color_palette("Greens", as_cmap=True)
    col_idx = np.linspace(0.3, 0.8, len(opto_depol_pcts))
    for i, opto_depol_pct in enumerate(opto_depol_pcts):
        ax.plot(t, mean_opto_voltages[opto_depol_pct], color=cm(col_idx[i]), label="%i%%" % opto_depol_pct)
    ax.legend(frameon=False)
    ax.set_xlim([t[0], t[-1]])
    ax.set_ylim([np.min(mean_ctrl_voltage), np.max(mean_ctrl_voltage)])
    sns.despine(trim=True, offset=2)
    fig.savefig(fig_name, bbox_inches="tight", dpi=100)
    plt.close(fig)



