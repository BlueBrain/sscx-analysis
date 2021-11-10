"""
Plot SSCx hex_O1 rasters
author: András Ecker, last update: 11.2021
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
# copied from `prepare_raster_asth.py`
RED = "#e32b14"
BLUE = "#3271b8"
GREEN = "#67b32e"
ORANGE = "#c9a021"
legend_handles = [mlines.Line2D([], [], color=RED, marker="s", linestyle='None', markersize=10, label="PC"),
                  mlines.Line2D([], [], color=BLUE, marker="s", linestyle='None', markersize=10, label="PV"),
                  mlines.Line2D([], [], color=GREEN, marker="s", linestyle='None', markersize=10, label="Sst"),
                  mlines.Line2D([], [], color=ORANGE, marker="s", linestyle='None', markersize=10, label="5HT3aR")]
PROJ_COLORS = {"VPM": "#4a4657", "POm": "#442e8a"}
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
    """Read VPM and POm spikes from spike replay file"""
    spike_times, spiking_gids = utils.get_tc_spikes(sim, t_start, t_end)
    vpm_gids, pom_gids = utils.load_tc_gids(os.path.split(sim.config.Run_Default.CurrentDir)[0])
    proj_rate_dict = {}
    for proj_name, proj_gids in zip(["VPM", "POm"], [vpm_gids, pom_gids]):
        if proj_gids is not None:
            mask = np.isin(spiking_gids, proj_gids)
            if mask.sum() > 0:
                proj_spike_times, proj_spiking_gids = spike_times[mask], spiking_gids[mask]
                proj_rate_dict[proj_name] = utils.calc_rate(proj_spike_times, len(np.unique(proj_spiking_gids)),
                                                            t_start, t_end)
    return proj_rate_dict if len(proj_rate_dict) else None


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
    cmap = plt.cm.get_cmap("tab10", 10)
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(pos[:, 0], pos[:, 1], color="gray", marker='.', s=50, alpha=0.5)
    ax.set_xlabel("x (flat space)")
    ax.set_ylabel("y (flat space)")
    sns.despine()
    for i, (name, gids) in enumerate(pattern_gids.items()):
        pattern_pos = pos[np.isin(all_gids, gids), :]
        if name in ["A", "B", "C", "D"]:
            ax.scatter(pattern_pos[:, 0], pattern_pos[:, 1], color=cmap(i), marker='.', s=100, label=name)
        fig2 = plt.figure(figsize=(10, 9))
        ax2 = fig2.add_subplot(1, 1, 1)
        ax2.scatter(pos[:, 0], pos[:, 1], color="gray", marker='.', s=50, alpha=0.5)
        ax2.scatter(pattern_pos[:, 0], pattern_pos[:, 1], color=cmap(i), marker='.', s=100)
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


if __name__ == "__main__":

    project_name = "cdf61143-0299-4a41-928d-b2cf0577d543"
    t_start = 1500
    plt_patterns = True

    sim_paths = utils.load_sim_path(project_name)
    level_names = sim_paths.index.names
    utils.ensure_dir(os.path.join(FIGS_DIR, project_name))
    with open("raster_asth.pkl", "rb") as f:
        raster_asthetics = pickle.load(f)

    for idx, sim_path in sim_paths.iteritems():
        sim = Simulation(sim_path)
        t_end = sim.t_end
        spike_times, spiking_gids = utils.get_spikes(sim, t_start, t_end)
        proj_rate_dict = get_tc_rates(sim, t_start, t_end)
        fig_name = os.path.join(FIGS_DIR, project_name, "%sraster.png" % utils.midx2str(idx, level_names))
        plot_raster(spike_times, spiking_gids, proj_rate_dict, raster_asthetics, t_start, t_end, fig_name)

    if plt_patterns:
        if "stim_seed" not in level_names:
            pattern_gids, gids, pos, _ = utils.load_patterns(project_name)
            plot_patterns(pattern_gids, gids, pos, os.path.join(FIGS_DIR, project_name, "patterns"))
        else:
            for seed in sim_paths.index.levels[level_names == "stim_seed"].to_numpy():
                pattern_gids, gids, pos, _ = utils.load_patterns(project_name, seed)
                plot_patterns(pattern_gids, gids, pos, os.path.join(FIGS_DIR, project_name, "patterns_seed%i" % seed))






