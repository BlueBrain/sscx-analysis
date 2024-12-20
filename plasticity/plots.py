"""
Plastic SSCx analysis related plots
author: András Ecker, last update: 02.2023
"""

import numpy as np
from utils import ensure_dir, numba_hist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
from statannotations.Annotator import Annotator

sns.set(style="ticks", context="notebook")
RED, BLUE = "#e32b14", "#3271b8"


def plot_lw_rates(df, mean_str, sd_str, rp_2015_rates, dks_2007_rates,  fig_name):
    """Plot layer-wise firing rates (split by E-I types) and (in vivo) reference values"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(4, 2, 1)
    sns.barplot(x=mean_str, y="rate", hue=sd_str, data=df[df["cell_type"] == "L23E"],
                palette="OrRd", ax=ax)
    ax.axhline(rp_2015_rates["L23E"], color="red", ls="--", label="RP_2015")
    ax.axhline(dks_2007_rates["L23E"], color="gray", ls="--", label="dKS_2007")
    ax.set_title("Excitatory")
    ax.set_xlabel(""); ax.set_ylabel("L23 rate (Hz)")
    ax.legend(frameon=False)
    # for container in ax.containers:
    #     ax.bar_label(container)
    ax = fig.add_subplot(4, 2, 3)
    sns.barplot(x=mean_str, y="rate", hue=sd_str, data=df[df["cell_type"] == "L4E"],
                palette="OrRd", ax=ax)
    ax.axhline(rp_2015_rates["L4E"], color="red", ls="--")
    ax.axhline(dks_2007_rates["L4E"], color="gray", ls="--")
    ax.set_xlabel(""); ax.set_ylabel("L4 rate (Hz)")
    ax.legend([], [], frameon=False)
    ax = fig.add_subplot(4, 2, 5)
    sns.barplot(x=mean_str, y="rate", hue=sd_str, data=df[df["cell_type"] == "L5E"],
                palette="OrRd", ax=ax)
    ax.axhline(rp_2015_rates["L5E"], color="red", ls="--")
    ax.axhline(dks_2007_rates["L5E"], color="gray", ls="--")
    ax.legend([], [], frameon=False)
    ax.set_xlabel(""); ax.set_ylabel("L5 rate (Hz)")
    ax = fig.add_subplot(4, 2, 7)
    sns.barplot(x=mean_str, y="rate", hue=sd_str, data=df[df["cell_type"] == "L6E"],
                palette="OrRd", ax=ax)
    ax.axhline(dks_2007_rates["L6E"], color="gray", ls="--")
    ax.legend([], [], frameon=False)
    ax.set_ylabel("L6 rate (Hz)")
    ax = fig.add_subplot(4, 2, 2)
    sns.barplot(x=mean_str, y="rate", hue=sd_str, data=df[df["cell_type"] == "L23I"],
                palette="PuBu", ax=ax)
    ax.axhline(rp_2015_rates["L23I"], color="red", ls="--")
    ax.set_title("Inhibitory")
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.legend(title=sd_str, frameon=False)
    ax = fig.add_subplot(4, 2, 4)
    sns.barplot(x=mean_str, y="rate", hue=sd_str, data=df[df["cell_type"] == "L4I"],
                palette="PuBu", ax=ax)
    ax.axhline(rp_2015_rates["L4I"], color="red", ls="--")
    ax.legend([], [], frameon=False)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax = fig.add_subplot(4, 2, 6)
    sns.barplot(x=mean_str, y="rate", hue=sd_str, data=df[df["cell_type"] == "L5I"],
                palette="PuBu", ax=ax)
    ax.axhline(rp_2015_rates["L5I"], color="red", ls="--")
    ax.legend([], [], frameon=False)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax = fig.add_subplot(4, 2, 8)
    sns.barplot(x=mean_str, y="rate", hue=sd_str, data=df[df["cell_type"] == "L6I"],
                palette="PuBu", ax=ax)
    ax.legend([], [], frameon=False)
    ax.set_ylabel("")
    sns.despine(bottom=True, trim=True)
    fig.tight_layout()
    fig.savefig(fig_name, bbox_inches="tight", dpi=100)
    plt.close(fig)


def plot_lw_rates_pct(df, mean_str, sd_str, fig_name):
    """Plot percentage of layer-wise in silico firing rates (split by E-I types)"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(3, 2, 1)
    sns.barplot(x=mean_str, y="rate_pct", hue=sd_str, data=df[df["cell_type"] == "L23E"],
                palette="OrRd", ax=ax)
    ax.set_title("Excitatory")
    ax.set_xlabel(""); ax.set_ylabel("L23 rate (% of in vivo)"); ax.set_ylim([0, 100])
    ax.legend(frameon=False)
    ax = fig.add_subplot(3, 2, 3)
    sns.barplot(x=mean_str, y="rate_pct", hue=sd_str, data=df[df["cell_type"] == "L4E"],
                palette="OrRd", ax=ax)
    ax.set_xlabel(""); ax.set_ylabel("L4 rate (% of in vivo)"); ax.set_ylim([0, 100])
    ax.legend([], [], frameon=False)
    ax = fig.add_subplot(3, 2, 5)
    sns.barplot(x=mean_str, y="rate_pct", hue=sd_str, data=df[df["cell_type"] == "L5E"],
                palette="OrRd", ax=ax)
    ax.set_ylabel("L5 rate (% of in vivo)"); ax.set_ylim([0, 100])
    ax.legend([], [], frameon=False)
    ax = fig.add_subplot(3, 2, 2)
    sns.barplot(x=mean_str, y="rate_pct", hue=sd_str, data=df[df["cell_type"] == "L23I"],
                palette="PuBu", ax=ax)
    ax.set_title("Inhibitory")
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_ylim([0, 100])
    ax.legend(title=sd_str, frameon=False)
    ax = fig.add_subplot(3, 2, 4)
    sns.barplot(x=mean_str, y="rate_pct", hue=sd_str, data=df[df["cell_type"] == "L4I"],
                palette="PuBu", ax=ax)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_ylim([0, 100])
    ax.legend([], [], frameon=False)
    ax = fig.add_subplot(3, 2, 6)
    sns.barplot(x=mean_str, y="rate_pct", hue=sd_str, data=df[df["cell_type"] == "L5I"],
                palette="PuBu", ax=ax)
    ax.set_ylabel(""); ax.set_ylim([0, 100])
    ax.legend([], [], frameon=False)
    sns.despine(bottom=True, trim=True)
    fig.tight_layout()
    fig.savefig(fig_name, bbox_inches="tight", dpi=100)
    plt.close(fig)


def _get_nsyn_range(n_syns_dict):
    """Concatenates convergence results and return overall min, max and 95% percentile"""
    pattern_names = list(n_syns_dict.keys())
    n_syns = n_syns_dict[pattern_names[0]]
    for pattern_name in pattern_names[1:]:
        n_syns = np.concatenate((n_syns, n_syns_dict[pattern_name]))
    return np.min(n_syns), np.max(n_syns), np.percentile(n_syns, 95)


def plot_tc_convergence(n_syns_dict, patterns_dir):
    """Plots TC convergence histograms"""
    ensure_dir(patterns_dir)
    min_syns, max_syns, p95_syns = _get_nsyn_range(n_syns_dict)
    cmap = plt.cm.get_cmap("tab10", len(n_syns_dict))
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(1, 1, 1)
    for i, (pattern_name, n_syns) in enumerate(n_syns_dict.items()):
        fig2 = plt.figure(figsize=(10, 9))
        ax2 = fig2.add_subplot(1, 1, 1)
        ax.hist(n_syns, bins=25, range=(min_syns, p95_syns), color=cmap(i), histtype="step", label=pattern_name)
        ax2.hist(n_syns, bins=25, range=(min_syns, p95_syns), color=cmap(i))
        ax2.set_xlabel("Nr. of TC synapses per cortical EXC cell")
        ax2.set_xlim([min_syns, p95_syns])
        ax2.set_ylabel("Count")
        sns.despine(ax=ax2, offset=5, trim=True)
        fig_name = os.path.join(patterns_dir, "convergence_%s.png" % pattern_name)
        fig2.savefig(fig_name, bbox_inches="tight", dpi=100)
        plt.close(fig2)
    ax.set_xlabel("Nr. of TC synapses per cortical EXC cell")
    ax.set_xlim([min_syns, p95_syns])
    ax.set_ylabel("Count")
    sns.despine(ax=ax, offset=5, trim=True)
    ax.legend(frameon=False)
    fig_name = os.path.join(patterns_dir, "convergence.png")
    fig.savefig(fig_name, bbox_inches="tight", dpi=100)
    plt.close(fig)


def plot_gmax_dists(gmax, fig_name):
    """Plots gmax distribution (on log scale) over time"""
    n_tbins = gmax.shape[0]
    gmax_range = [0.1, 5]  # [np.percentile(gmax, 1), np.max(gmax)]
    log_bins = np.logspace(np.log10(gmax_range[0]), np.log10(gmax_range[1]), 30)
    gmax_means = np.mean(gmax, axis=1)
    min_mean, max_mean = np.min(gmax_means), np.max(gmax_means)
    cmap = LinearSegmentedColormap.from_list("cmap", plt.cm.Greys(np.linspace(0.2, 0.8, 256)))
    cmap_idx = (gmax_means - min_mean) / (max_mean - min_mean)  # get them to [0, 1] for cmap

    fig = plt.figure(figsize=(10, 6.5))
    gs = gridspec.GridSpec(n_tbins, 2, width_ratios=[69, 1])
    for i in range(n_tbins):
        ax = fig.add_subplot(gs[i, 0])
        color = cmap(cmap_idx[i])
        ax.hist(gmax[i, :], bins=log_bins, color=color, edgecolor=color)
        ax.set_xscale("log")
        ax.set_xlim(gmax_range)
        ax.set_yticks([])
        if i == n_tbins - 1:
            ax.set_xlabel("gmax_AMPA (nS)")
            sns.despine(ax=ax, left=True, offset=2)
        else:
            ax.set_xticks([])
            ax.minorticks_off()
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
        pos_hist, bin_edges = numba_hist(gmax_change[gmax_change > 0], 30, (0, gmax_change_95p))
        neg_hist, _ = numba_hist(gmax_change[gmax_change < 0], 30, (-1 * gmax_change_95p, 0))
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        width = bin_centers[1] - bin_centers[0]
        ax.bar(bin_centers, pos_hist, width=width, color=RED, edgecolor="black", lw=0.5)
        ax.bar(-1 * bin_centers[::-1], -1 * neg_hist, width=width, color=BLUE, edgecolor="black", lw=0.5)
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
        gmax_diff = np.concatenate((gmax_diffs[2], gmax_diffs[3])) if layer == 23 else gmax_diffs[layer]
        n_syns = len(gmax_diff)
        potentiated = len(np.where(gmax_diff > 0)[0])
        depressed = len(np.where(gmax_diff < 0)[0])
        sizes = np.array([potentiated, n_syns - (potentiated + depressed), depressed])
        ratios = 100 * sizes / np.sum(sizes)
        ax = fig.add_subplot(2, 2, i+1)
        ax.pie(sizes, labels=["%.2f%%" % ratio for ratio in ratios], colors=[RED, "lightgray", BLUE])
        ax.set_title("L%i (n = %i)" % (layer, n_syns))
    fig.tight_layout()
    fig.savefig(fig_name, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams["patch.edgecolor"] = "white"


def plot_rho_hist(t, data, fig_name):
    """Plot histogram of rho (at fixed time t)"""
    t /= 1000.  # ms to second
    hue_order = np.sort(data["post_mtype"].unique())
    cmap = plt.get_cmap("tab20", len(hue_order))
    colors = [cmap(i) for i in range(len(hue_order))]
    fig = plt.figure(figsize=(13, 6.5))
    ax = fig.add_subplot(1, 2, 1)
    sns.histplot(data=data[data["loc"] == "apical"], y="rho", hue="post_mtype", multiple="stack",
                 bins=30, binrange=(0, 1), hue_order=hue_order, palette=colors, legend=False, ax=ax)
    ax.set_title("apical")
    ax.set_ylim([0, 1])
    ax.set_ylabel("Rho at t = %s (s)" % t)
    ax2 = fig.add_subplot(1, 2, 2)
    sns.histplot(data=data[data["loc"] == "basal"], y="rho", hue="post_mtype", multiple="stack",
                 bins=30, binrange=(0, 1), hue_order=hue_order, palette=colors, ax=ax2)
    ax2.set_title("basal")
    ax2.set_ylim([0, 1])
    ax2.set_ylabel("")
    plt.legend(handles=ax2.legend_.legendHandles, labels=[t.get_text() for t in ax2.legend_.texts],
               title=ax2.legend_.get_title().get_text(), bbox_to_anchor=(1.05, 1), loc=2, frameon=False)
    sns.despine(offset=2, trim=True)
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_rho_stack(bins, t, data, fig_name, split=True):
    """Plots stacked time series of (binned) rho"""
    t /= 1000.  # ms to second
    n = data.shape[0]
    assert len(bins) - 1 == n
    cmap = plt.get_cmap("coolwarm", n)
    colors = [cmap(i) for i in range(n)]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.BoundaryNorm(bins, cmap.N))
    fig = plt.figure(figsize=(10, 6.5))
    if not split:  # simply use matplotlib's `stackplot`
        ax = fig.add_subplot(1, 1, 1)
        ax.stackplot(t, data, colors=colors)
        plt.colorbar(sm, ax=ax, ticks=bins, label="rho")
        ax.set_xlim([t[0], t[-1]])
        ax.set_xlabel("Time (s)")
        ax.set_ylim([0, np.sum(data[:, 0])])
        ax.set_ylabel("#Synapses")
    else:  # building a stackplot with `fill_between` on 4 subplots (2 logscaled)
        assert np.mod(n, 2) == 0, "Logscaling splits depressed and potentiated synapses" \
                                  "and is only possible with even number of bins"
        n_pot, n_dep = data[-1, 0], data[0, 0]
        pot_ratio = n_pot / (n_pot + n_dep)
        # non-trivial `nt` bins and number of synapses within them
        pot_idx, dep_idx = np.arange(int(n / 2), n - 1), np.arange(int(n / 2), 1, -1) - 1
        n_nt_pot, n_nt_dep = np.sum([data[i, -1] for i in pot_idx]), np.sum([data[i, -1] for i in dep_idx])
        nt_pot_ratio = n_nt_pot / (n_nt_pot + n_nt_dep)
        height_ratios = [pot_ratio, nt_pot_ratio, 1 - nt_pot_ratio, 1 - pot_ratio]
        gs = gridspec.GridSpec(4, 2, width_ratios=[40, 1], wspace=0.1, height_ratios=height_ratios, hspace=0.2)

        # potentiated synapses
        ax = fig.add_subplot(gs[0, 0])
        all_pot = n_pot * np.ones_like(t)
        ax.fill_between(t, all_pot - data[-1, :], all_pot, color=colors[-1])
        ax.set_xlim([t[0], t[-1]])
        ax.set_xticks([])
        ax.set_ylim([1, n_pot])
        ax.set_yscale("log")
        sns.despine(ax=ax, bottom=True, trim=True, offset=2)
        ax = fig.add_subplot(gs[1, 0])
        low = np.zeros_like(t)
        for i in pot_idx:
            high = low + data[i, :]
            ax.fill_between(t, low, high, color=colors[i])
            low = high
        ax.set_xlim([t[0], t[-1]])
        ax.set_xticks([])
        ax.set_ylim([0, n_nt_pot])
        sns.despine(ax=ax, bottom=True, trim=True, offset=2)
        # depressed synapses
        ax = fig.add_subplot(gs[2, 0])
        low = np.zeros_like(t)
        for i in dep_idx:
            high = low + data[i, :]
            ax.fill_between(t, low, high, color=colors[i])
            low = high
        ax.set_xlim([t[0], t[-1]])
        ax.set_xticks([])
        ax.set_ylim([n_nt_dep, 0])
        sns.despine(ax=ax, bottom=True, trim=True, offset=1)
        ax = fig.add_subplot(gs[3, 0])
        all_dep = n_dep * np.ones_like(t)
        ax.fill_between(t, all_dep - data[0, :], all_dep, color=colors[0])
        ax.set_xlim([t[0], t[-1]])
        ax.set_xlabel("Time (s)")
        ax.set_ylim([n_dep, 1])
        ax.set_yscale("log")
        sns.despine(ax=ax, trim=True, offset=2)

        cax = fig.add_subplot(gs[:, 1])
        fig.colorbar(sm, cax=cax, ticks=bins, label="rho")
        fig.add_subplot(1, 1, 1, frameon=False)
        plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
        plt.ylabel("#Synapses")
    fig.tight_layout()
    fig.savefig(fig_name, bbox_inches="tight", dpi=100)
    plt.close(fig)


def plot_mean_rho_matrix(t, mtypes, rho_matrix, fig_name):
    """Plots transition matrix (on log scale)"""
    t /= 1000.  # ms to second
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("white")
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(1, 1, 1)
    i = ax.imshow(rho_matrix, cmap=cmap, aspect="auto", vmin=0.15, vmax=0.85)
    plt.colorbar(i, label="Mean rho at t = %s (s)" % t)
    ticks = np.arange(len(mtypes))
    ax.set_xticks(ticks)
    ax.set_xticklabels(mtypes, rotation=45)
    ax.set_xlabel("To (mtype)")
    ax.set_yticks(ticks)
    ax.set_yticklabels(mtypes)
    ax.set_ylabel("From (mtype)")
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


def plot_rate_vs_change(df, fig_name):
    """Plots total changes (of mean per connection)
    vs. (well it's just a histogram, not scatter plot....) mean pairwise firing rates"""
    max_rate = df["pw_rate"].quantile(0.999)
    dep_hist, bin_edges = numba_hist(df.loc[df["delta"] < 0, "pw_rate"].to_numpy(), 30, (0, max_rate))
    pot_hist, _ = numba_hist(df.loc[df["delta"] > 0, "pw_rate"].to_numpy(), 30, (0, max_rate))
    unch_hist, _ = numba_hist(df.loc[df["delta"] == 0, "pw_rate"].to_numpy(), 30, (0, max_rate))
    norms = pot_hist + dep_hist + unch_hist
    dep_hist, pot_hist, unch_hist = 100 * dep_hist / norms, 100 * pot_hist / norms, 100 * unch_hist / norms
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    width = bin_centers[1] - bin_centers[0]
    fig = plt.figure(figsize=(10, 6.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(bin_centers, dep_hist, color=BLUE, width=width, edgecolor="white", lw=0.5)
    ax.bar(bin_centers, unch_hist, bottom=dep_hist, color="lightgray", width=width, edgecolor="white", lw=0.5)
    ax.bar(bin_centers, pot_hist, bottom=dep_hist + unch_hist, color=RED, width=width, edgecolor="white", lw=0.5)
    ax.set_xlabel("Mean pairwise rate (Hz)")
    ax.set_xlim([0, max_rate])
    ax.set_ylim([0, 100])
    ax.set_ylabel("% of connections")
    sns.despine(offset=2)
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_agg_edge_dists(ts, dists, fig_name):
    """Plots L2 norm of differences between edges (of aggregated connectivity matrix) in consecutive time bins"""
    ts = ts / 1000  # ms -> s conversion
    xticks = np.arange(0, ts[-1]+0.1, 30)
    fig = plt.figure(figsize=(10, 6.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ts, dists, color="black", linewidth=1.5)
    for t in xticks:
        ax.axvline(t, color="gray", alpha=0.5, linewidth=0.5)
    ax.set_xticks(xticks)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Norm. dist. in agg. conn. mat.")
    sns.despine(trim=True)
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


# duplicated from `analyse_syn_clusters.py`...
def _sort_keys(key_list):
    """Sort keys of assembly idx. If -1 is part of the list (standing for non-assembly) then that comes last"""
    if -1 not in key_list:
        return np.sort(key_list)
    else:
        keys = np.array(key_list)
        return np.concatenate((np.sort(keys[keys >= 0]), np.array([-1])))
    

def plot_2x2_cond_probs(probs, pot_matrices, dep_matrices, fig_name):
    """For every assembly plots pie chart with total changes in sample neurons and 2 matrices
    with the cond. prob. of potentiation and depression (in a 2x2 grid)"""
    plt.rcParams["patch.edgecolor"] = "black"
    neg_colors = plt.cm.Greys_r(np.linspace(0, 1, 128))
    pot_colors = plt.cm.Reds(np.linspace(0, 1, 128))
    dep_colors = plt.cm.Blues(np.linspace(0, 1, 128))
    pot_cmap = LinearSegmentedColormap.from_list("pot_cmap", np.vstack((neg_colors, pot_colors)))
    dep_cmap = LinearSegmentedColormap.from_list("dep_cmap", np.vstack((neg_colors, dep_colors)))
    pot_cmap.set_bad(color="tab:pink")
    dep_cmap.set_bad(color="tab:pink")
    pot_extr = np.max([np.nanmax(np.abs(pot_matrix)) for _, pot_matrix in pot_matrices.items()])
    dep_extr = np.max([np.nanmax(np.abs(dep_matrix)) for _, dep_matrix in dep_matrices.items()])

    assembly_idx = _sort_keys(list(probs.keys()))
    n = len(assembly_idx)
    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(3, n+1, width_ratios=[10 for i in range(n)] + [1])
    for i, assembly_id in enumerate(assembly_idx):
        ax = fig.add_subplot(gs[0, i])
        ax.pie(probs[assembly_id], labels=["%.2f%%" % (prob * 100) for prob in probs[assembly_id]],
               colors=[RED, "lightgray", BLUE], normalize=True)
        ax.set_title("assembly %i" % assembly_id)
        ax2 = fig.add_subplot(gs[1, i])
        i_pot = ax2.imshow(pot_matrices[assembly_id], cmap=pot_cmap, aspect="auto", vmin=-pot_extr, vmax=pot_extr)
        ax3 = fig.add_subplot(gs[2, i])
        i_dep = ax3.imshow(dep_matrices[assembly_id], cmap=dep_cmap, aspect="auto", vmin=-dep_extr, vmax=dep_extr)
        if i == 0:
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(["assembly", "non-assembly"])
            ax3.set_yticks([0, 1])
            ax3.set_yticklabels(["assembly", "non-assembly"])
        else:
            ax2.set_yticks([])
            ax3.set_yticks([])
        ax2.set_xticks([])
        ax3.set_xticks([0, 1])
        ax3.set_xticklabels(["clustered", "not clustered"], rotation=45)
    fig.colorbar(i_pot, cax=fig.add_subplot(gs[1, i+1]), label="P(pot|cond) - P(pot) /\n P(pot|cond) + P(pot)")
    fig.colorbar(i_dep, cax=fig.add_subplot(gs[2, i+1]), label="P(dep|cond) - P(pot) /\n P(dep|cond) + P(pot)")
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_nx2_cond_probs(probs, fracs, pot_matrix, dep_matrix, post_assembly_id, fig_name):
    """For cross assembly plots pie chart with total changes in sample neurons and 2 matrices
    with the cond. prob. of potentiation and depression (in a Nx2 grid)"""
    plt.rcParams["patch.edgecolor"] = "black"
    neg_colors = plt.cm.Greys_r(np.linspace(0, 1, 128))
    pot_colors = plt.cm.Reds(np.linspace(0, 1, 128))
    dep_colors = plt.cm.Blues(np.linspace(0, 1, 128))
    pot_cmap = LinearSegmentedColormap.from_list("pot_cmap", np.vstack((neg_colors, pot_colors)))
    dep_cmap = LinearSegmentedColormap.from_list("dep_cmap", np.vstack((neg_colors, dep_colors)))
    pot_cmap.set_bad(color="tab:pink")
    dep_cmap.set_bad(color="tab:pink")
    pot_extr = np.nanmax(np.abs(pot_matrix))
    dep_extr = np.nanmax(np.abs(dep_matrix))

    fig = plt.figure(figsize=(10, 6.5))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 3])
    ax = fig.add_subplot(gs[0, 0])
    ax.pie(probs, labels=["%.2f%%" % (prob * 100) for prob in probs], colors=[RED, "lightgray", BLUE], normalize=True)
    ax.set_title("post assembly %i" % post_assembly_id)
    tmp = _sort_keys(list(fracs.keys()))
    ys = np.append(np.arange(len(tmp[:-1])), -1)
    yticklabels = ["pre assembly %i" % i for i in tmp[:-1]] + ["non-assembly"]
    width = [fracs[key] for key in tmp]
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(ys, width, color="gray")
    ax2.set_yticks(ys)
    ax2.set_yticklabels(yticklabels)
    ax2.set_xlabel("Synapse ratio")
    sns.despine(ax=ax2, offset=2)
    ax3 = fig.add_subplot(gs[1, 0])
    i_pot = ax3.imshow(pot_matrix, cmap=pot_cmap, aspect="auto", vmin=-pot_extr, vmax=pot_extr)
    fig.colorbar(i_pot, label="P(pot|cond) - P(pot) /\n P(pot|cond) + P(pot)")
    ax3.set_yticks([i for i in range(len(yticklabels))])
    ax3.set_yticklabels(yticklabels)
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["clustered", "not clustered"], rotation=45)
    ax4 = fig.add_subplot(gs[1, 1])
    i_dep = ax4.imshow(dep_matrix, cmap=dep_cmap, aspect="auto", vmin=-dep_extr, vmax=dep_extr)
    fig.colorbar(i_dep, label="P(dep|cond) - P(pot) /\n P(dep|cond) + P(pot)")
    ax4.set_yticks([])
    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(["clustered", "not clustered"], rotation=45)
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_diffs_stats(pot_df, dep_df, pot_pairs, dep_pairs, pot_p_vals, dep_p_vals, fig_name):
    """Box plot and significance test of assembly synapse cluster on assembly neurons"""
    order = ["a-c", "a-nc", "na-c", "na-nc"]
    fig = plt.figure(figsize=(10, 6.5))
    dep_df.loc[:, "delta_rho"] = dep_df["delta_rho"].abs()  # to be able to show them on the same axis
    ax = fig.add_subplot(1, 2, 1)
    ax.set_ylim([0, 1])
    sns.boxplot(data=dep_df, x="groups", y="delta_rho", order=order, fliersize=0, color=BLUE, ax=ax)
    sns.stripplot(data=dep_df, x="groups", y="delta_rho", order=order,
                  dodge=True, size=3, color="black", edgecolor=None, ax=ax)
    ax.set_xlabel("")
    if dep_pairs is not None:
        annotator = Annotator(ax, pairs=dep_pairs, data=dep_df, x="groups", y="delta_rho", order=order)
        annotator.configure(loc="outside").set_pvalues(pvalues=dep_p_vals).annotate()
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_ylim([0, 1])
    sns.boxplot(data=pot_df, x="groups", y="delta_rho", order=order, fliersize=0, color=RED, ax=ax2)
    sns.stripplot(data=pot_df, x="groups", y="delta_rho", order=order,
                  dodge=True, size=3, color="black", edgecolor=None, ax=ax2)
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    if pot_pairs is not None:
        annotator = Annotator(ax2, pairs=pot_pairs, data=pot_df, x="groups", y="delta_rho", order=order)
        annotator.configure(loc="outside").set_pvalues(pvalues=pot_p_vals).annotate()
    sns.despine(bottom=True)
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_bglibpy_trace(t, v, t_nd, v_nd, fig_name):
    """Plots soma voltage traces from original (Neurodamus) and BGLibPy sims"""
    fig = plt.figure(figsize=(10, 6.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t, v, 'r-', label="BGLibPy")
    ax.plot(t_nd, v_nd, 'k-', label="Neurodamus")
    ax.legend(frameon=False)
    ax.set_xlim([t[0], t[-1]])
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("V (mV)")
    sns.despine(offset=2)
    fig.savefig(fig_name, bbox_inches="tight", dpi=100)
    plt.close(fig)