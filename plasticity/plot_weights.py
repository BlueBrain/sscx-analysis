"""
Plot evolution of synaptic weights (in plasticity simulations) over time
author: András Ecker, last update: 12.2021
"""

import os
from copy import deepcopy
import numpy as np
import utils
from bluepy import Circuit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns

sns.set(style="ticks", context="notebook")
FIGS_DIR = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/figures/sscx-analysis"
RED, BLUE = "#e32b14", "#3271b8"


def plot_gmax_dists(gmax, fig_name):
    """Plots gmax distribution over time"""
    n_tbins = gmax.shape[0]
    gmax_range = [np.min(gmax), np.percentile(gmax, 95)]
    gmax_means = np.mean(gmax, axis=1)
    min_mean, max_mean = np.min(gmax_means), np.max(gmax_means)
    cmap = plt.get_cmap("viridis")
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


def plot_gmax_change_pie(gmax_diffs, fig_name):
    """Plots layer-wise pie charts of percentage of gmax changing"""
    plt.rcParams["patch.edgecolor"] = "black"
    fig = plt.figure(figsize=(10, 6.5))
    for i, layer in enumerate([23, 4, 5, 6]):
        gmax_diff = np.concatenate((gmax_diffs[2], gmax_diffs[2])) if layer == 23 else gmax_diffs[layer]
        n_syns = len(gmax_diff)
        potentiated = len(np.where(gmax_diff > 0)[0])
        depressed = len(np.where(gmax_diff < 0)[0])
        sizes = np.array([potentiated, n_syns - (potentiated + depressed), depressed])
        ratios = 100 * sizes / np.sum(sizes)
        ax = fig.add_subplot(2, 2, i+1)
        ax.pie(sizes, labels=["%.2f%%" % ratio for ratio in ratios], colors=[RED, "lightgray", BLUE])
        ax.set_title("L%i (n = %i)" % (layer, n_syns))
    fig.tight_layout()
    fig.savefig(fig_name, bbox_inches="tight", transparent=True)
    plt.close(fig)
    plt.rcParams["patch.edgecolor"] = "white"


def plot_rho_hist(t, data, fig_name):
    """Plot histogram of rho (at fixed time t)"""
    t /= 1000.  # ms to second
    categories = list(data.keys())
    cmap = plt.get_cmap("tab20", len(categories))
    colors = [cmap(i) for i in range(len(categories))]
    plot_data = [data[cat].reshape(-1) for cat in categories]
    fig = plt.figure(figsize=(10, 6.5))
    ax = fig.add_subplot(1, 1, 1)
    ns, bins, _ = ax.hist(plot_data, bins=30, range=(0, 1), stacked=True, color=colors, label=categories)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, frameon=False)
    ax.set_xlim([0, 1])
    ax.set_xlabel("Rho at t = %s (s)" % t)
    sns.despine(ax=ax, offset=True, trim=True)
    ax2 = inset_axes(ax, width="80%", height="70%", loc=9)
    # one should be able to reuse `ns` from the prev. hist, but idk. how...
    ax2.hist(plot_data, bins=bins[1:-1], stacked=True, color=colors, label=categories)
    ax2.set_xlim([bins[1], bins[-2]])
    ax2.set_yticks([])
    sns.despine(ax=ax2, left=True, offset=True, trim=True)
    fig.savefig(fig_name, bbox_inches="tight", dpi=100)
    plt.close(fig)


def plot_rho_stack(bins, t, data, fig_name):
    """Plots stacked time series of (binned) rho"""
    t /= 1000.  # ms to second
    n = data.shape[0]
    assert len(bins)-1 == n
    cmap = plt.get_cmap("coolwarm", n)
    colors = [cmap(i) for i in range(n)]
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


def plot_transition_matrix(transition_matrix, bins, fig_name):
    """Plots transition matrix (on log scale)"""
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad(cmap(0.0))
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(1, 1, 1)
    i = ax.imshow(transition_matrix, cmap=cmap, aspect="auto",
                  extent=(0, transition_matrix.shape[1], transition_matrix.shape[0], 0),
                  norm=matplotlib.colors.LogNorm(vmax=np.max(transition_matrix)))
    plt.colorbar(i, label="Prob. (of transition)")
    ticks = np.arange(len(bins))
    ticklabels = ["%.1f" % bin for bin in bins]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.set_xlabel("To (rho)")
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels)
    ax.set_ylabel("From (rho)")
    fig.savefig(fig_name, bbox_inches="tight", dpi=100)
    plt.close(fig)


def get_transition_matrix(data, bins):
    """Calculates transition matrix from 2 time steps of data (after binning)"""
    assert data.shape[0] == 2, "Transition matrix can only be calculated from 2 (ideally consecutive) time steps"
    bin_edges = np.linspace(0, 1, bins+1) if isinstance(bins, int) else bins
    bin_edges[-1] += 1e-5
    idx = np.digitize(data, bin_edges) - 1
    # work only with values that change
    diff_idx = idx[:, np.where(np.diff(idx, axis=0) != 0)[1]]
    unique_transitions, counts = np.unique(diff_idx, return_counts=True, axis=1)
    transition_matrix = np.zeros((len(bin_edges)-1, len(bin_edges)-1))
    for i in range(unique_transitions.shape[1]):
        transition_matrix[unique_transitions[0, i], unique_transitions[1, i]] = counts[i]
    # normalize with (total) count (in the 1st state) to get probabilities
    unique_states, counts = np.unique(idx[0, :], return_counts=True)
    transition_matrix = np.divide(transition_matrix[unique_states], counts.reshape(-1, 1))
    # fill in missing diagonal (non-changing) values by getting rows sum to 1
    diags = 1 - np.sum(transition_matrix, axis=1)
    for i, diag in enumerate(diags):
        transition_matrix[i, i] = diag
    bin_edges[-1] -= 1e-5
    return transition_matrix, bin_edges


def get_total_change_by(sim_path, report_name, split_by="layer", return_data=False):
    """Loads full report, splits it and gets total change (last-first time step)"""
    c = Circuit(sim_path)
    h5f_name = os.path.join(os.path.split(sim_path)[0], "%s.h5" % report_name)
    data = utils.load_synapse_report(h5f_name, return_idx=True)
    split_data = utils.split_synapse_report(c, data, split_by)
    split_data = utils.update_split_data(c, report_name, split_data, split_by)
    diffs = {key: val[-1]-val[0] for key, val in split_data.items()}
    return data, diffs if return_data else diffs


def get_last_synapse_report_by(sim_path, report_name, split_by="mtype"):
    """Loads last time step of report, splits it and updates it with non-saved data"""
    c = Circuit(sim_path)
    h5f_name = os.path.join(os.path.split(sim_path)[0], "%s.h5" % report_name)
    last_data = utils.load_synapse_report(h5f_name, t_start=-1, return_idx=True)
    t = last_data.index.to_numpy()[0]
    split_data = utils.split_synapse_report(c, last_data, split_by)
    split_data = utils.update_split_data(c, report_name, split_data, split_by)
    return t, split_data


def main(project_name):
    sim_paths = utils.load_sim_paths(project_name)
    level_names = sim_paths.index.names
    utils.ensure_dir(os.path.join(FIGS_DIR, project_name))

    for idx, sim_path in sim_paths.iteritems():
        report_name = "gmax_AMPA"
        data, diffs = get_total_change_by(sim_path, report_name, return_data=True)
        fig_name = os.path.join(FIGS_DIR, project_name, "%sgmax_AMPA_delta_pies.png" % utils.midx2str(idx, level_names))
        plot_gmax_change_pie(diffs, fig_name)
        fig_name = os.path.join(FIGS_DIR, project_name, "%sgmax_AMPA_delta_hists.png" % utils.midx2str(idx, level_names))
        plot_gmax_change_hist(data.to_numpy(), fig_name)
        fig_name = os.path.join(FIGS_DIR, project_name, "%sgmax_AMPA_hists.png" % utils.midx2str(idx, level_names))
        plot_gmax_dists(data.to_numpy(), fig_name)

        report_name = "rho"
        h5f_name = os.path.join(os.path.split(sim_path)[0], "%s.h5" % report_name)
        bins, t, hist_data = utils.get_synapse_report_hist(h5f_name)
        hist_data = utils.update_hist_data(report_name, hist_data, bins)
        fig_name = os.path.join(FIGS_DIR, project_name, "%srho_stack.png" % utils.midx2str(idx, level_names))
        plot_rho_stack(bins, deepcopy(t), hist_data, fig_name)
        t_idx = int(len(t) / 2)
        _, middle_data = utils.load_synapse_report(h5f_name, t_start=t[t_idx], t_end=t[t_idx + 1])
        transition_matrix, _ = get_transition_matrix(middle_data, bins)
        fig_name = os.path.join(FIGS_DIR, project_name, "%srho_transition.png" % utils.midx2str(idx, level_names))
        plot_transition_matrix(deepcopy(transition_matrix), bins, fig_name)
        last_t, last_data = get_last_synapse_report_by(sim_path, report_name)
        fig_name = os.path.join(FIGS_DIR, project_name, "%srho_hist.png" % utils.midx2str(idx, level_names))
        plot_rho_hist(last_t, last_data, fig_name)


if __name__ == "__main__":
    project_name = "LayerWiseEShotNoise_PyramidPatterns"  # "5b1420ca-dd31-4def-96d6-46fe99d20dcc"
    main(project_name)
