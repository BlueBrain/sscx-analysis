"""
Plot SSCx hex_O1 rasters
author: András Ecker, last update: 06.2022
"""

import os
import time
import pickle
import numpy as np
from bluepy import Simulation
import utils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import seaborn as sns

sns.set(style="ticks", context="notebook")
# these 4 colors are copied from `prepare_raster_asth.py`
RED = "#e32b14"
BLUE = "#3271b8"
GREEN = "#67b32e"
ORANGE = "#c9a021"
# "fake" legend for 4 cell classes
legend_handles = [mlines.Line2D([], [], color=RED, marker="s", linestyle='None', markersize=10, label="PC"),
                  mlines.Line2D([], [], color=BLUE, marker="s", linestyle='None', markersize=10, label="PV"),
                  mlines.Line2D([], [], color=GREEN, marker="s", linestyle='None', markersize=10, label="Sst"),
                  mlines.Line2D([], [], color=ORANGE, marker="s", linestyle='None', markersize=10, label="5HT3aR")]
PROJ_COLORS = {"VPM": "#4a4657", "POm": "#442e8a"}
PATTERN_COLORS = {"A": "#253e92", "B": "#57b4d0", "C": "#c4a943", "D": "#7e1e18", "E": "#3f79b2",
                  "F": "#8dad8a", "G": "#a1632e", "H": "#66939d", "I": "#97885c", "J": "#665868"}
FIGS_DIR = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/figures/sscx-analysis"


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
        rates_dict[type_] = utils.calc_rate(spike_times[cols == type_color], Ns[type_], t_start, t_end)

    return spiking_ys, cols, rates_dict


def get_tc_rates(sim, t_start, t_end):
    """Read VPM and POm spikes from SpikeFile"""
    spike_times, spiking_gids = utils.get_tc_spikes(sim, t_start, t_end)
    vpm_gids, pom_gids = utils.load_tc_gids(os.path.split(sim.config.Run_Default.CurrentDir)[0])
    proj_spikes, proj_rates = {}, {}
    for proj_name, proj_gids in zip(["VPM", "POm"], [vpm_gids, pom_gids]):
        if proj_gids is not None:
            mask = np.isin(spiking_gids, proj_gids)
            if mask.sum() > 0:
                proj_spikes[proj_name] = {"spike_times": spike_times[mask], "spiking_gids": spiking_gids[mask]}
                proj_rates[proj_name] = utils.calc_rate(spike_times[mask], len(np.unique(spiking_gids[mask])),
                                                            t_start, t_end)
    if len(proj_rates):
        return proj_spikes, proj_rates
    else:
        return None, None


def get_pattern_rates(pattern_gids, spike_times, spiking_gids, t_start, t_end):
    """Gets VPM spikes by pattern"""
    pattern_spikes, pattern_sc_rates, pattern_rates = {}, {}, {}
    for name, gids in pattern_gids.items():
        mask = np.isin(spiking_gids, gids)
        if mask.sum() > 0:
            pattern_spikes[name] = {"spike_times": spike_times[mask], "spiking_gids": spiking_gids[mask]}
            _, sc_rate = utils.calc_sc_rate(spiking_gids[mask], t_start, t_end)
            pattern_sc_rates[name] = sc_rate
            pattern_rates[name] =  utils.calc_rate(spike_times[mask], len(np.unique(spiking_gids[mask])), t_start, t_end)
    return pattern_spikes, pattern_sc_rates, pattern_rates



def plot_raster(spike_times, spiking_gids, proj_rate_dict, asthetics, t_start, t_end, fig_name):
    """Plots raster"""
    start_time = time.time()
    spiking_ys, cols, rates_dict = setup_raster(spike_times, spiking_gids,
                                                asthetics["ys"], asthetics["colors"], asthetics["groups"],
                                                asthetics["types"], asthetics["type_colors"], t_start, t_end)
    t_rate = np.linspace(t_start, t_end, len(rates_dict[asthetics["types"][0]]))

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
    print("Plotted and saved in: %s" % time.strftime("%M:%S", time.gmtime(time.time() - start_time)))
    plt.close(fig)


def plot_patterns(pattern_gids, all_gids, pos, patterns_dir):
    """Plots projection patterns in flatmap space"""
    utils.ensure_dir(patterns_dir)
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(pos[:, 0], pos[:, 1], color="gray", marker='.', s=50, alpha=0.5)
    ax.set_xlabel("x (flat space)")
    ax.set_ylabel("y (flat space)")
    sns.despine()
    for name, gids in pattern_gids.items():
        pattern_pos = pos[np.isin(all_gids, gids), :]
        if name in ["A", "B", "C", "D"]:
            ax.scatter(pattern_pos[:, 0], pattern_pos[:, 1], color=PATTERN_COLORS[name], marker='.', s=100, label=name)
        fig2 = plt.figure(figsize=(10, 9))
        ax2 = fig2.add_subplot(1, 1, 1)
        ax2.scatter(pos[:, 0], pos[:, 1], color="gray", marker='.', s=50, alpha=0.5)
        ax2.scatter(pattern_pos[:, 0], pattern_pos[:, 1], color=PATTERN_COLORS[name], marker='.', s=100)
        ax2.set_xlabel("x (flat space)")
        ax2.set_ylabel("y (flat space)")
        sns.despine()
        fig_name = os.path.join(patterns_dir, "pattern_%s.png" % name)
        fig2.savefig(fig_name, bbox_inches="tight", dpi=100)
        plt.close(fig2)
    ax.legend(frameon=False)
    fig_name = os.path.join(patterns_dir, "patterns_ABCD.png")
    fig.savefig(fig_name, bbox_inches="tight", dpi=100)
    plt.close(fig)


def _normalize_gids(spiking_gids):
    """Normalize TC gids for better plotting (works the same way as `setup_raster()` does)"""
    unique_gids, idx = np.unique(spiking_gids, return_inverse=True)
    shifted_gids = np.arange(0, len(unique_gids))
    return shifted_gids[idx]


def plot_pattern_rates(pattern_spikes, pattern_sc_rates, pattern_rates, t_start, t_end, fig_name):
    """Plot VPM spikes and rates grouped by individual patterns"""
    fig = plt.figure(figsize=(10, 12))
    gs = gridspec.GridSpec(len(pattern_spikes), 2, width_ratios=[9, 1])
    pattern_names = np.sort(list(pattern_spikes.keys()))
    t_rate = np.linspace(t_start, t_end, len(pattern_rates[pattern_names[0]]))
    max_sc_rate = np.max([np.max(rate) for _, rate in pattern_sc_rates.items()])
    max_rate = np.max([np.max(rate) for _, rate in pattern_rates.items()])

    for i, name in enumerate(pattern_names):
        color = PATTERN_COLORS[name]
        ax = fig.add_subplot(gs[i, 0])
        spiking_gids = _normalize_gids(pattern_spikes[name]["spiking_gids"])
        ax.scatter(pattern_spikes[name]["spike_times"], spiking_gids, color=color, marker='.', s=10., edgecolor="none")
        ax.set_xlim([t_start, t_end])
        ax.set_yticks([0, np.max(spiking_gids)])
        ax.set_ylabel(name)
        ax2 = ax.twinx()
        ax2.plot(t_rate, pattern_rates[name], color=color)
        ax2.fill_between(t_rate, np.zeros_like(t_rate),  pattern_rates[name], color=color, alpha=0.1)
        ax2.set_ylim([0, max_rate])
        # ax2.set_ylabel("Rate (Hz)")
        ax3 = fig.add_subplot(gs[i, 1])
        ax3.hist(pattern_sc_rates[name], bins=10, range=(0, max_sc_rate), color=color)
        ax3.set_xlim([0, max_sc_rate])
        # ax3.set_ylabel("Count")
        sns.despine(ax=ax3)
        if i == len(pattern_names) - 1:
            ax.set_xlabel("Time (ms)")
            ax3.set_xlabel("Single cell rate (Hz)")
        else:
            ax.set_xticks([])
            ax3.set_xticks([])
    gs.tight_layout(fig, h_pad=0.2)
    fig.savefig(fig_name, bbox_inches="tight", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    project_name = "66bcc1ef-4d2f-4941-be68-3aa33a33c6a9"
    t_start = 1900
    plt_patterns = True
    plt_pattern_spikes = True

    sim_paths = utils.load_sim_paths(project_name)
    level_names = sim_paths.index.names
    utils.ensure_dir(os.path.join(FIGS_DIR, project_name))
    with open("raster_asth.pkl", "rb") as f:
        raster_asthetics = pickle.load(f)
    if plt_patterns and "stim_seed" not in level_names:
        pattern_gids, tc_gids, tc_pos, _ = utils.load_patterns(project_name)
        plot_patterns(pattern_gids, tc_gids, tc_pos, os.path.join(FIGS_DIR, project_name, "patterns"))

    for idx, sim_path in sim_paths.iteritems():
        sim = Simulation(sim_path)
        t_end = 7000 #sim.t_end
        spike_times, spiking_gids = utils.get_spikes(sim, t_start, t_end)
        proj_spikes, proj_rates = get_tc_rates(sim, t_start, t_end)
        fig_name = os.path.join(FIGS_DIR, project_name, "%sraster.png" % utils.midx2str(idx, level_names))
        plot_raster(spike_times, spiking_gids, proj_rates, raster_asthetics, t_start, t_end, fig_name)

        if plt_pattern_spikes and "stim_seed" not in level_names:
            if not plt_patterns:
                pattern_gids, tc_gids, tc_pos, _ = utils.load_patterns(project_name)
            pattern_spikes, pattern_sc_rates, pattern_rates = get_pattern_rates(pattern_gids,
                            proj_spikes["VPM"]["spike_times"], proj_spikes["VPM"]["spiking_gids"], t_start, t_end)
            fig_name = os.path.join(FIGS_DIR, project_name, "%spatterns.png" % utils.midx2str(idx, level_names))
            plot_pattern_rates(pattern_spikes, pattern_sc_rates, pattern_rates, t_start, t_end, fig_name)
        if "stim_seed" in level_names:
            if plt_patterns:
                stim_seed = idx[level_names == "stim_seed"] if len(level_names) > 1 else idx
                pattern_gids, tc_gids, tc_pos, _ = utils.load_patterns(project_name, stim_seed)
                fig_name = os.path.join(FIGS_DIR, project_name, "patterns_seed%i" % stim_seed)
                plot_patterns(pattern_gids, tc_gids, tc_pos, fig_name)
            if plt_pattern_spikes:
                pattern_spikes, pattern_sc_rates, pattern_rates = get_pattern_rates(pattern_gids,
                                proj_spikes["VPM"]["spike_times"], proj_spikes["VPM"]["spiking_gids"], t_start, t_end)
                fig_name = os.path.join(FIGS_DIR, project_name, "%spatterns.png" % utils.midx2str(idx, level_names))
                plot_pattern_rates(pattern_spikes, pattern_sc_rates, pattern_rates, t_start, t_end, fig_name)






