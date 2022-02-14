"""
Loads synapse clusters saved by `assemblyfire` and checks their total changes in the synapse report
author: András Ecker, last update: 02.2022
"""

import os
from copy import deepcopy
import numpy as np
import pandas as pd
import utils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

sns.set(style="ticks", context="notebook")
FIGS_DIR = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/figures/v7_assemblies"
RED, BLUE = "#e32b14", "#3271b8"


def _sort_keys(key_list):
    """Sort keys of assembly idx. If -1 is part of the list (standing for non-assembly) then that comes last"""
    if -1 not in key_list:
        return np.sort(key_list)
    else:
        keys = np.array(key_list)
        return np.concatenate((np.sort(keys[keys >= 0]), np.array([-1])))


def get_change_probs(syn_clusters, diffs):
    """Gets probabilities of potentiated, unchanged, and depressed synapses"""
    probs = {}
    for assembly_id, syn_cluster in syn_clusters.items():
        assembly_diffs = diffs.loc[syn_cluster.index]
        n_syns = len(assembly_diffs)
        potentiated, depressed = len(assembly_diffs[assembly_diffs > 0]), len(assembly_diffs[assembly_diffs < 0])
        assembly_probs = np.array([potentiated, 0., depressed]) / n_syns
        assembly_probs[1] = 1 - np.sum(assembly_probs)
        probs[assembly_id] = assembly_probs
    return probs


def group_diffs(syn_clusters, diffs):
    """
    Groups efficacy differences (rho at last time step - rho at first time step) in sample neurons from assemblies
    based on 2 criteria: 1) synapse is coming from assembly neuron vs. non-assembly neuron (fist key (-1 for non-assembly))
                         2) synapse is part of a synapse cluster (clustering done in `assemblyfire`) vs. not (second key)
    """
    grouped_diffs = {}
    for assembly_id, syn_cluster in syn_clusters.items():
        pre_asssembly_idx = [int(label.split("assembly")[1]) for label in syn_cluster.columns.to_list()
                             if label.split("assembly")[0] == '']
        assembly_diffs = {pre_assembly_id: {} for pre_assembly_id in pre_asssembly_idx + [-1]}  # -1 for non-assembly
        for pre_assembly_id in pre_asssembly_idx:
            assembly_syns = syn_cluster["assembly%i" % pre_assembly_id]
            assembly_diffs[pre_assembly_id][1] = diffs.loc[assembly_syns[assembly_syns >= 0].index]
            assembly_diffs[pre_assembly_id][0] = diffs.loc[assembly_syns[assembly_syns == -1].index]
        non_assembly_syns = syn_cluster["non_assembly"]
        assembly_diffs[-1] = {1: diffs.loc[non_assembly_syns[non_assembly_syns >= 0].index],
                              0: diffs.loc[non_assembly_syns[non_assembly_syns == -1].index]}
        grouped_diffs[assembly_id] = assembly_diffs
    return grouped_diffs


def get_fracs(syn_clusters):
    """Gets fraction of synapses coming from assemblies (and for non-assembly saved with key: -1)"""
    fracs = {}
    for assembly_id, syn_cluster in syn_clusters.items():
        pre_asssembly_idx = [int(label.split("assembly")[1]) for label in syn_cluster.columns.to_list()
                             if label.split("assembly")[0] == '']
        assembly_fracs = {pre_assembly_id: {} for pre_assembly_id in pre_asssembly_idx + [-1]}  # -1 for non-assembly
        for pre_assembly_id in pre_asssembly_idx:
            assembly_syns = syn_cluster["assembly%i" % pre_assembly_id]
            assembly_fracs[pre_assembly_id] = len(assembly_syns[assembly_syns >= -1]) / len(assembly_syns)
        non_assembly_syns = syn_cluster["non_assembly"]
        assembly_fracs[-1] = len(non_assembly_syns[non_assembly_syns >= -1]) / len(non_assembly_syns)
        fracs[assembly_id] = assembly_fracs
    return fracs


def _print_user_target_blocks(gids):
    """Print extra user.target blocks for detailed reporting of `gids` (that can be copy pasted)"""
    user_target_blocks = "\nTarget Cell DetailedReport\n{\n"
    for gid in gids:
        user_target_blocks += "a%i " % gid
    user_target_blocks += "\n}\n\nTarget Compartment Compartments_DetailedReport\n{\nDetailedReport\n}\n"
    print(user_target_blocks)


def get_grouped_diffs(seed, sim_path, report_name, late_assembly=False, print_ut=False):
    """Wrapper of other functions that together load and pre-calculate/group stuff for statistic tests and plotting"""
    syn_clusters, gids = utils.load_synapse_clusters(seed, sim_path, late_assembly)
    if print_ut:
        _print_user_target_blocks(gids)
    diffs = utils.get_synapse_changes(sim_path, report_name, gids)
    probs = get_change_probs(syn_clusters, diffs)
    fracs = get_fracs(syn_clusters)
    grouped_diffs = group_diffs(syn_clusters, diffs)
    return probs, fracs, grouped_diffs


def get_michelson_contrast(probs, grouped_diffs):
    """Gets Michelson contrast (aka. visibility) defined as:
    P(pot/dep | condition) - P(pot/dep) / (P(pot/dep | condition) + P(pot/dep))"""
    pot_contrasts, dep_contrasts = {}, {}
    for assembly_id, uncond_probs in probs.items():
        p_pot, p_dep = uncond_probs[0], uncond_probs[2]  # unchanged is the 2nd element, which we won't use here
        pre_assembly_idx = _sort_keys(list(grouped_diffs[assembly_id].keys()))
        pot_contrast, dep_contrast = np.zeros((len(pre_assembly_idx), 2)), np.zeros((len(pre_assembly_idx), 2))
        for i, pre_assembly_id in enumerate(pre_assembly_idx):
            for j, clustered in enumerate([1, 0]):  # looks useless, but this way clustered comes first in the plots...
                df = grouped_diffs[assembly_id][pre_assembly_id][clustered]
                n_syns = len(df)
                if n_syns:
                    p_pot_cond = len(df[df > 0]) / n_syns
                    pot_contrast[i, j] = (p_pot_cond - p_pot) / (p_pot_cond + p_pot)
                    p_dep_cond = len(df[df < 0]) / n_syns
                    dep_contrast[i, j] = (p_dep_cond - p_dep) / (p_dep_cond + p_dep)
                else:
                    pot_contrast[i, j], dep_contrast[i, j] = np.nan, np.nan
        pot_contrasts[assembly_id], dep_contrasts[assembly_id] = pot_contrast, dep_contrast
    return pot_contrasts, dep_contrasts


def assembly_syn_cluster_diffs2df(grouped_diffs):
    """Creates a DataFrame from assembly synapse cluster diffs (easier to plot and merge with extra morph. features)"""
    pre_assembly_idx = _sort_keys(list(grouped_diffs.keys()))[:-1]  # don't use non-assembly synapses
    dfs = []
    for pre_assembly_id in pre_assembly_idx:
        df = grouped_diffs[pre_assembly_id][1].to_frame()  # [1]: clustered synapses
        df["pre_assembly"] = pre_assembly_id
        dfs.append(df)
    return pd.concat(dfs).sort_index()


def plot_2x2_cond_probs(probs, pot_matrices, dep_matrices, fig_name):
    """For every assembly plots pie chart with total changes in sample neurons and 2 matrices
    with the cond. prob. of potentiation and depression (in a 2x2 grid)"""
    plt.rcParams["patch.edgecolor"] = "black"
    neg_colors = plt.cm.Greys_r(np.linspace(0, 1, 128))
    pot_colors = plt.cm.Reds(np.linspace(0, 1, 128))
    dep_colors = plt.cm.Blues(np.linspace(0, 1, 128))
    pot_cmap = LinearSegmentedColormap.from_list("pot_cmap", np.vstack((neg_colors, pot_colors)))
    dep_cmap = LinearSegmentedColormap.from_list("pot_cmap", np.vstack((neg_colors, dep_colors)))
    pot_extr = np.max([np.max(np.abs(pot_matrix)) for _, pot_matrix in pot_matrices.items()])
    dep_extr = np.max([np.max(np.abs(dep_matrix)) for _, dep_matrix in dep_matrices.items()])

    assembly_idx = _sort_keys(list(probs.keys()))
    n = len(assembly_idx)
    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(3, n+1, width_ratios=[10 for i in range(n)] + [1])
    for i, assembly_id in enumerate(assembly_idx):
        ax = fig.add_subplot(gs[0, i])
        ax.pie(probs[assembly_id], labels=["%.3f" % prob for prob in probs[assembly_id]],
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
    fig.savefig(fig_name, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_nx2_cond_probs(probs, fracs, pot_matrix, dep_matrix, fig_name):
    """For a late assembly plots pie chart with total changes in sample neurons and 2 matrices
    with the cond. prob. of potentiation and depression (in a Nx2 grid)"""
    plt.rcParams["patch.edgecolor"] = "black"
    neg_colors = plt.cm.Greys_r(np.linspace(0, 1, 128))
    pot_colors = plt.cm.Reds(np.linspace(0, 1, 128))
    dep_colors = plt.cm.Blues(np.linspace(0, 1, 128))
    pot_cmap = LinearSegmentedColormap.from_list("pot_cmap", np.vstack((neg_colors, pot_colors)))
    dep_cmap = LinearSegmentedColormap.from_list("pot_cmap", np.vstack((neg_colors, dep_colors)))
    pot_extr = np.max(np.abs(pot_matrix))
    dep_extr = np.max(np.abs(dep_matrix))

    yticklabels = ["assembly %i" % i for i in range(pot_matrix.shape[0] - 1)] + ["non-assembly"]
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 5])
    ax = fig.add_subplot(gs[0, 0])
    ax.pie(probs, labels=["%.3f" % prob for prob in probs], colors=[RED, "lightgray", BLUE], normalize=True)
    ax.set_title("assembly 0")
    y = _sort_keys(list(fracs.keys()))
    width = [fracs[key] for key in y]
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(y, width, color="gray")
    ax2.set_yticks(y)
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
    fig.savefig(fig_name, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_morph_features(df, fig_name):
    """Plot morphological features (detailed scatter plot)"""
    df.loc[df["loc"].isin(["oblique", "trunk", "tuft"]), "loc"] = "apical"
    cmap = plt.cm.get_cmap("tab20", df["pre_assembly"].max() + 1)
    palette = {i: cmap(i) for i in df["pre_assembly"].unique()}
    fig = plt.figure(figsize=(7, 11))
    ax = fig.add_subplot(1, 1, 1)
    sns.scatterplot(data=df, x="diam", y="dist", hue="pre_assembly", hue_order=np.sort(list(palette.keys())),
                    palette=palette, style="loc", edgecolor="none", ax=ax)  # size="br_ord"
    # ax.set_xlim([df["diam"].min(), df["diam"].max()])
    # ax.set_ylim([df["dist"].min(), df["dist"].max()])
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, frameon=False)
    sns.despine(offset=2, trim=True)
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main(project_name):
    report_name = "rho"
    sim_paths = utils.load_sim_paths(project_name)
    level_names = sim_paths.index.names
    assert len(level_names) == 1 and level_names[0] == "seed"
    utils.ensure_dir(os.path.join(FIGS_DIR, project_name))

    for seed, sim_path in sim_paths.iteritems():
        probs, _, grouped_diffs = get_grouped_diffs(seed, sim_path, report_name)
        pot_contrasts, dep_contrasts = get_michelson_contrast(probs, grouped_diffs)
        fig_name = os.path.join(FIGS_DIR, project_name, "syn_clust_plast_seed%i.png" % seed)
        plot_2x2_cond_probs(probs, pot_contrasts, dep_contrasts, fig_name)

    morph_df = utils.load_extra_morph_features(["loc", "dist", "diam", "br_ord"])
    for seed, sim_path in sim_paths.iteritems():
        print_ut = True if seed == 31 else False  # seed 31 is totally hand selected...
        probs, fracs, grouped_diffs = get_grouped_diffs(seed, sim_path, report_name,
                                                        late_assembly=True, print_ut=print_ut)
        pot_contrasts, dep_contrasts = get_michelson_contrast(probs, grouped_diffs)
        fig_name = os.path.join(FIGS_DIR, project_name, "late_assembly_syn_clust_plast_seed%i.png" % seed)
        # late assembly_id is 0 (it just happens to be... and is hard coded in `assemblyfire` as well)
        # and as the rest of the functions create dicts, one has to select it by the `0` key here...
        plot_nx2_cond_probs(probs[0], fracs[0], pot_contrasts[0], dep_contrasts[0], fig_name)
        df = assembly_syn_cluster_diffs2df(deepcopy(grouped_diffs[0]))
        df = pd.concat([df, morph_df.loc[df.index]], axis=1)
        # fig_name = os.path.join(FIGS_DIR, project_name, "late_assembly_syn_clust_morph_seed%i.png" % seed)
        # plot_morph_features(df, fig_name)


if __name__ == "__main__":
    project_name = "e0fbb0c8-07a4-49e0-be7d-822b2b2148fb"
    main(project_name)
