"""
Plot evolution of synaptic weights (in plasticity simulations) over time
author: András Ecker, last update: 12.2021
"""

import os
import numpy as np
import utils
from bluepy import Simulation
from bluepy.enums import Cell
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set(style="ticks", context="notebook")
FIGS_DIR = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/figures/sscx-analysis"
RED = "#e32b14"
BLUE = "#3271b8"


def plot_gmax_dists(gmax, fig_name):
    """Plots gmax distribution over time"""
    n_tbins = gmax.shape[0]
    gmax_range = [np.min(gmax), np.percentile(gmax, 95)]
    gmax_means = np.mean(gmax, axis=1)
    min_mean, max_mean = np.min(gmax_means), np.max(gmax_means)
    cmap = plt.get_cmap("coolwarm")
    cmap_idx = (gmax_means - min_mean) / (max_mean - min_mean)  # get them to [0, 1] for cmap

    fig = plt.figure(figsize=(10, 6.5))
    gs = gridspec.GridSpec(n_tbins, 2, width_ratios=[69, 1])
    for i in range(n_tbins):
        ax = fig.add_subplot(gs[i, 0])
        color = cmap(cmap_idx[i])
        ax.hist(gmax[i, :], bins=30, range=gmax_range, color=color, edgecolor=color)
        ax.set_xlim(gmax_range)
        ax.set_yticks([])
        if i == n_tbins - 1:
            ax.set_xlabel("gmax_AMPA (nS)")
            sns.despine(ax=ax, left=True, trim=True, offset=2)
        else:
            ax.set_xticks([])
            sns.despine(ax=ax, left=True, bottom=True)
        ax.patch.set_alpha(0)
    gs.update(hspace=-0.2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(vmin=min_mean, vmax=max_mean))
    plt.colorbar(sm, cax=fig.add_subplot(gs[:, 1]), ticks=[min_mean, max_mean], format="%.3f")  #, label="Mean gmax (nS)")
    fig.savefig(fig_name, bbox_inches="tight", dpi=100)
    plt.close(fig)


def plot_gmax_change_hist(gmax, fig_name):
    """Plots changes in gmax distribution over time"""
    gmax_change = np.diff(gmax, axis=0) * 1000
    gmax_change[gmax_change == 0.] = np.nan
    gmax_change_95p = np.nanpercentile(np.abs(gmax_change), 95)
    n_tbins = gmax_change.shape[0]

    fig = plt.figure(figsize=(10, 6.5))
    gs = gridspec.GridSpec(n_tbins, 1)
    for i in range(n_tbins):
        ax = fig.add_subplot(gs[i, 0])
        pos_hist, bin_edges = utils.numba_hist(gmax_change[gmax_change > 0], 30, (0, gmax_change_95p))
        neg_hist, _ = utils.numba_hist(gmax_change[gmax_change < 0], 30, (-1 * gmax_change_95p, 0))
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        ax.bar(bin_centers, pos_hist, color=RED, edgecolor="black", lw=0.5)
        ax.bar(-1 * bin_centers[::-1], -1 * neg_hist, color=BLUE, edgecolor="black", lw=0.5)
        ax.set_xlim([-gmax_change_95p, gmax_change_95p])
        ax.set_yticks([])
        if i == n_tbins - 1:
            ax.set_xlabel(r"$\Delta$ gmax_AMPA (pS)")
            sns.despine(ax=ax, left=True, trim=True)
        else:
            ax.set_xticks([])
            sns.despine(ax=ax, left=True, bottom=True)
        ax.patch.set_alpha(0)
    gs.update(hspace=-0.1)
    fig.savefig(fig_name, bbox_inches="tight", dpi=100)
    plt.close(fig)


def plot_gmax_change_pie(c, data, fig_name, target="hex_O1ExcitatoryPlastic"):
    """Plots layer-wise pie charts of percentage of gmax changing"""
    plt.rcParams["patch.edgecolor"] = "black"
    fig = plt.figure(figsize=(10, 6.5))
    for i, layer in enumerate([23, 4, 5, 6]):
        bluepy_layer = layer if layer != 23 else [2, 3]
        gids = c.cells.ids({"$target": target, Cell.LAYER: bluepy_layer})
        gmax = data[gids].to_numpy()
        gmax_change = gmax[-1] - gmax[0]
        n_syns = len(gmax_change)
        if layer == 6:  # the non-reported L6 synapses don't change... but will add to the total number
            n_syns += len(utils.load_nonrep_syn_df())
        potentiated = len(np.where(gmax_change > 0)[0])
        depressed = len(np.where(gmax_change < 0)[0])
        sizes = np.array([potentiated, n_syns - (potentiated + depressed), depressed])
        ratios = 100 * sizes / np.sum(sizes)
        ax = fig.add_subplot(2, 2, i+1)
        ax.pie(sizes, labels=["%.2f%%" % ratio for ratio in ratios], colors=[RED, "lightgray", BLUE])
        ax.set_title("L%i (n = %i)" % (layer, n_syns))
    fig.tight_layout()
    fig.savefig(fig_name, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_rho_hist(t, data, fig_name):
    """Plot histogram of rho (at fixed time t)"""
    t /= 1000.  # ms to second
    fig = plt.figure(figsize=(10, 6.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(data, bins=30, range=(0, 1), color="gray", edgecolor="black")
    ax.set_xlim([0, 1])
    ax.set_xlabel("Rho at t = %s (s)" % t)
    sns.despine(offset=True, trim=True)
    fig.savefig(fig_name, bbox_inches="tight", dpi=100)
    plt.close(fig)


def plot_rho_stack(bins, t, data, fig_name):
    """Plots stacked time series of (binned) rho"""
    t /= 1000.  # ms to second
    cmap = plt.get_cmap("coolwarm", 10)
    colors = [cmap(i) for i in range(10)]
    fig = plt.figure(figsize=(10, 6.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.stackplot(t, data, colors=colors)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.BoundaryNorm(bins, cmap.N))
    plt.colorbar(sm, ax=ax, ticks=bins, label="rho")
    ax.set_xlim([t[0], t[-1]])
    ax.set_xlabel("Time (s)")
    ax.set_ylim([0, np.sum(data[:, 0])])
    ax.set_ylabel("#Synapses")
    fig.tight_layout()
    fig.savefig(fig_name, bbox_inches="tight", dpi=100)
    plt.close(fig)


def get_synapse_report_hist(sim_path, report_name):
    """Local wrapper of `utils.get_hist_synapse_report()` that saves and loads data"""
    npzf_name = os.path.join(os.path.split(sim_path)[0], "hist_%s.npz" % report_name)
    if not os.path.isfile(npzf_name):
        h5f_name = os.path.join(os.path.split(sim_path)[0], "%s.h5" % report_name)
        bins, t, data = utils.get_synapse_report_hist(h5f_name)
        data = utils.update_hist_data(report_name, data, bins)
        np.savez(npzf_name, bins=bins, t=t, data=data)
    else:
        npzf = np.load(npzf_name)
        bins, t, data = npzf["bins"], npzf["t"], npzf["data"]
    return bins, t, data


if __name__ == "__main__":
    # project_name = "5b1420ca-dd31-4def-96d6-46fe99d20dcc"
    project_name = "LayerWiseEShotNoise_disconnected"

    sim_paths = utils.load_sim_paths(project_name)
    level_names = sim_paths.index.names
    utils.ensure_dir(os.path.join(FIGS_DIR, project_name))

    for idx, sim_path in sim_paths.iteritems():
        sim = Simulation(sim_path)
        report_name = "gmax_AMPA"
        h5f_name = os.path.join(os.path.split(sim_path)[0], "%s.h5" % report_name)
        data = utils.load_synapse_report(h5f_name, return_idx=True)
        fig_name = os.path.join(FIGS_DIR, project_name, "%sgmax_AMPA_delta_pies.png" % utils.midx2str(idx, level_names))
        plot_gmax_change_pie(sim.circuit, data, fig_name)
        fig_name = os.path.join(FIGS_DIR, project_name, "%sgmax_AMPA_delta_hists.png" % utils.midx2str(idx, level_names))
        plot_gmax_change_hist(data.to_numpy(), fig_name)
        fig_name = os.path.join(FIGS_DIR, project_name, "%sgmax_AMPA_hists.png" % utils.midx2str(idx, level_names))
        plot_gmax_dists(data.to_numpy(), fig_name)

        report_name = "rho"
        h5f_name = os.path.join(os.path.split(sim_path)[0], "%s.h5" % report_name)
        last_t, data = utils.load_synapse_report(h5f_name, t_start=-1)
        fig_name = os.path.join(FIGS_DIR, project_name, "%srho_hist.png" % utils.midx2str(idx, level_names))
        plot_rho_hist(last_t[0], data.reshape(-1), fig_name)
        bins, t, data = get_synapse_report_hist(sim_path, report_name)
        fig_name = os.path.join(FIGS_DIR, project_name, "%srho_stack.png" % utils.midx2str(idx, level_names))
        plot_rho_stack(bins, t, data, fig_name)





