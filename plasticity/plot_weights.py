"""
Plot evolution of synaptic weights (in plasticity simulations) over time
author: András Ecker, last update: 12.2021
"""

import os
import numpy as np
import utils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set(style="ticks", context="notebook")
FIGS_DIR = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/figures/sscx-analysis"
RED = "#e32b14"
BLUE = "#3271b8"


def plot_gmax_dist(gmax, fig_name):
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


def plot_gmax_change(gmax, fig_name):
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


def plot_rho_2dhist(bins, t, data, fig_name):
    """Plot 2d histogram of rho's time evolution"""
    fig = plt.figure(figsize=(10, 6.5))
    ax = fig.add_subplot(1, 1, 1)
    i = ax.imshow(data, cmap="inferno",  norm=matplotlib.colors.LogNorm(vmax=np.max(data)),
                  aspect="auto", origin="lower")
    plt.colorbar(i, label="#Synapses")
    xtick_idx = np.linspace(0, len(t)-1, 5).astype(int)
    ax.set_xticks(xtick_idx)
    ax.set_xticklabels(t[xtick_idx]/1000.)
    ax.set_xlabel("Time (s)")
    ytick_idx = np.linspace(0, len(bins)-1, 5).astype(int)
    ax.set_yticks(ytick_idx)
    ax.set_yticklabels(bins[ytick_idx])
    ax.set_ylabel("rho")
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


def get_binned_synapse_report(sim_path, report_name):
    """Local wrapper of `utils.get_binned_synapse_report()` that saves and loads data"""
    npzf_name = os.path.join(os.path.split(sim_path)[0], "binned_%s.npz" % report_name)
    if not os.path.isfile(npzf_name):
        h5f_name = os.path.join(os.path.split(sim_path)[0], "%s.h5" % report_name)
        bins, t, data = utils.get_binned_synapse_report(h5f_name)
        data = utils.update_binned_data(report_name, data, bins)
        np.savez(npzf_name, bins=bins, t=t, data=data)
    else:
        npzf = np.load(npzf_name)
        bins, t, data = npzf["bins"], npzf["t"], npzf["data"]
    return bins, t, data


if __name__ == "__main__":
    project_name = "5b1420ca-dd31-4def-96d6-46fe99d20dcc"

    sim_paths = utils.load_sim_path(project_name)
    level_names = sim_paths.index.names
    utils.ensure_dir(os.path.join(FIGS_DIR, project_name))

    for idx, sim_path in sim_paths.iteritems():
        report_name, t_step = "gmax_AMPA", 60000
        h5f_name = os.path.join(os.path.split(sim_path)[0], "%s.h5" % report_name)
        _, data = utils.load_synapse_report(h5f_name, t_step)
        fig_name = os.path.join(FIGS_DIR, project_name, "%sgmax_AMPA.png" % utils.midx2str(idx, level_names))
        plot_gmax_dist(data, fig_name)
        fig_name = os.path.join(FIGS_DIR, project_name, "%sdelta_gmax_AMPA.png" % utils.midx2str(idx, level_names))
        plot_gmax_change(data, fig_name)

        report_name = "rho"
        bins, t, data = get_binned_synapse_report(sim_path, report_name)
        fig_name = os.path.join(FIGS_DIR, project_name, "%srho_hist.png" % utils.midx2str(idx, level_names))
        plot_rho_2dhist(bins, t, data, fig_name)
        bins, data = utils.coarse_binning(bins, data, 10)
        fig_name = os.path.join(FIGS_DIR, project_name, "%srho_stack.png" % utils.midx2str(idx, level_names))
        plot_rho_stack(bins, t, data, fig_name)




