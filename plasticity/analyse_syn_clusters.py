"""
Loads synapse clusters saved by `assemblyfire` and checks their total changes in the synapse report
author: András Ecker, last update: 01.2022
"""

import os
import numpy as np
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


def get_change_probs(syn_clusters, diffs):
    """Gets probabilities of potentiated, unchanged, and depressed synapses"""
    probs = {}
    for assembly_label, syn_cluster in syn_clusters.items():
        assembly_diffs = diffs.loc[syn_cluster.index]
        n_syns = len(assembly_diffs)
        potentiated, depressed = len(assembly_diffs[assembly_diffs > 0]), len(assembly_diffs[assembly_diffs < 0])
        assembly_probs = np.array([potentiated, 0., depressed]) / n_syns
        assembly_probs[1] = 1 - np.sum(assembly_probs)
        probs[assembly_label] = assembly_probs
    return probs


def group_diffs(syn_clusters, diffs):
    """
    Groups efficacy differences (rho at last time step - rho at first time step) in sample neurons from assemblies
    based on 2 criteria: 1) synapse is coming from assembly neuron vs. non-assembly neuron (fist +/-)
                         2) synapse is part of a synapse cluster (clustering done in `assemblyfire`) vs. not (second +/-)
    """
    assembly_labels = np.sort(list(syn_clusters.keys()))
    grouped_diffs = {assembly_label: {} for assembly_label in assembly_labels}
    for assembly_label in assembly_labels:
        # get diffs for all 4 cases
        assembly_syns = syn_clusters[assembly_label]["assembly%i" % assembly_label]
        non_assembly_syns = syn_clusters[assembly_label]["non_assembly"]
        grouped_diffs[assembly_label]["++"] = diffs.loc[assembly_syns[assembly_syns >= 0].index]
        grouped_diffs[assembly_label]["+-"] = diffs.loc[assembly_syns[assembly_syns == -1].index]
        grouped_diffs[assembly_label]["-+"] = diffs.loc[non_assembly_syns[non_assembly_syns >= 0].index]
        grouped_diffs[assembly_label]["--"] = diffs.loc[non_assembly_syns[non_assembly_syns == -1].index]
    return grouped_diffs


def get_michelson_contrast(probs, group_diffs):
    """Gets Michelson contrast (aka. visibility) defined as:
    P(pot/dep | condition) - P(pot/dep) / (P(pot/dep | condition) + P(pot/dep))"""
    pot_contrasts, dep_contrasts = {}, {}
    for assembly_label, uncond_probs in probs.items():
        p_pot, p_dep = uncond_probs[0], uncond_probs[2]  # unchanged is the 2nd element, which we won't use here
        pot_contrast, dep_contrast = np.zeros((2, 2)), np.zeros((2, 2))
        for i, assembly in enumerate(["+", "-"]):
            for j, clustered in enumerate(["+", "-"]):
                df = group_diffs[assembly_label][assembly + clustered]
                n_syns = len(df)
                p_pot_cond = len(df[df > 0]) / n_syns
                pot_contrast[i, j] = (p_pot_cond - p_pot) / (p_pot_cond + p_pot)
                p_dep_cond = len(df[df < 0]) / n_syns
                dep_contrast[i, j] = (p_dep_cond - p_dep) / (p_dep_cond + p_dep)
        pot_contrasts[assembly_label], dep_contrasts[assembly_label] = pot_contrast, dep_contrast
    return pot_contrasts, dep_contrasts


def plot_grouped_diffs(probs, pot_matrices, dep_matrices, fig_name):
    """For every assembly plots pie chart with total changes in sample neurons and 2 matrices
    with the cond. prob. of potentiation and depression (in a 2x2 grid - see `group_diffs()` above)"""
    plt.rcParams["patch.edgecolor"] = "black"
    neg_colors = plt.cm.Greys_r(np.linspace(0, 1, 128))
    pot_colors = plt.cm.Reds(np.linspace(0, 1, 128))
    dep_colors = plt.cm.Blues(np.linspace(0, 1, 128))
    pot_cmap = LinearSegmentedColormap.from_list("pot_cmap", np.vstack((neg_colors, pot_colors)))
    dep_cmap = LinearSegmentedColormap.from_list("pot_cmap", np.vstack((neg_colors, dep_colors)))
    pot_extr = np.max([np.max(np.abs(pot_matrix)) for _, pot_matrix in pot_matrices.items()])
    dep_extr = np.max([np.max(np.abs(dep_matrix)) for _, dep_matrix in dep_matrices.items()])

    assembly_labels = np.sort(list(probs.keys()))
    n = len(assembly_labels)
    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(3, n+1, width_ratios=[10 for i in range(n)] + [1])
    for i, assembly_label in enumerate(assembly_labels):
        ax = fig.add_subplot(gs[0, i])
        ax.pie(probs[assembly_label], labels=["%.3f" % prob for prob in probs[assembly_label]],
               colors=[RED, "lightgray", BLUE], normalize=True)
        ax.set_title("assembly %i" % assembly_label)
        ax2 = fig.add_subplot(gs[1, i])
        i_pot = ax2.imshow(pot_matrices[assembly_label], cmap=pot_cmap, aspect="auto", vmin=-pot_extr, vmax=pot_extr)
        ax3 = fig.add_subplot(gs[2, i])
        i_dep = ax3.imshow(dep_matrices[assembly_label], cmap=dep_cmap, aspect="auto", vmin=-dep_extr, vmax=dep_extr)
        if i == 0:
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(["assembly", "non-assembly"])
            ax3.set_yticks([0, 1])
            ax3.set_yticklabels(["assembly", "non assembly"])
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


def main(project_name):
    report_name = "rho"
    sim_paths = utils.load_sim_paths(project_name)
    level_names = sim_paths.index.names
    assert len(level_names) == 1 and level_names[0] == "seed"
    utils.ensure_dir(os.path.join(FIGS_DIR, project_name))

    for seed, sim_path in sim_paths.iteritems():
        syn_clusters, gids = utils.load_synapse_clusters(seed, sim_path)
        diffs = utils.get_synapse_changes(sim_path, report_name, gids)
        probs = get_change_probs(syn_clusters, diffs)
        grouped_diffs = group_diffs(syn_clusters, diffs)

        pot_contrasts, dep_contrast = get_michelson_contrast(probs, grouped_diffs)
        fig_name = os.path.join(FIGS_DIR, project_name, "syn_clust_plast_seed%i.png" % seed)
        plot_grouped_diffs(probs, pot_contrasts, dep_contrast, fig_name)


if __name__ == "__main__":
    project_name = "e0fbb0c8-07a4-49e0-be7d-822b2b2148fb"
    main(project_name)
