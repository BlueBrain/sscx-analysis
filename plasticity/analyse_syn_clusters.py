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
import seaborn as sns

sns.set(style="ticks", context="notebook")
FIGS_DIR = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/figures/v7_assemblies"
RED, BLUE = "#e32b14", "#3271b8"


def group_diffs(syn_clusters, diffs, nsyn_samples=100):
    """
    Groups efficacy differences (rho at last time step - rho at first time step) on samples neurons based on
    2 criteria: 1) synapse is coming from assembly neuron vs. non-assembly neuron
                2) synapse is part of a synapse cluster (clustering done in `assemblyfire`) vs. not
    (It's neither the prettiest nor the most flexible function ever... but it does the job)
    """
    assembly_labels = np.sort(list(syn_clusters.keys()))
    sizes, pot_matrices, dep_matrices = {}, {}, {}
    for assembly_label in assembly_labels:
        # get ratios of potentiated vs. depressed synapses for a given assembly
        assembly_diffs = diffs.loc[syn_clusters[assembly_label].index]
        n_syns = len(assembly_diffs)
        potentiated, depressed = len(assembly_diffs[assembly_diffs > 0]), len(assembly_diffs[assembly_diffs < 0])
        sizes[assembly_label] = np.array([potentiated, n_syns - (potentiated + depressed), depressed])
        # get diffs for all 4 cases
        assembly_syns = syn_clusters[assembly_label]["assembly%i" % assembly_label]
        non_assembly_syns = syn_clusters[assembly_label]["non_assembly"]
        assembly_cluster_diffs = diffs.loc[assembly_syns[assembly_syns > 0].index]
        assembly_nc_diffs = diffs.loc[assembly_syns[assembly_syns == -1].index]
        non_assembly_cluster_diffs = diffs.loc[non_assembly_syns[non_assembly_syns > 0].index]
        non_assembly_nc_diffs = diffs.loc[non_assembly_syns[non_assembly_syns == -1].index]
        # get samples for potentiation and depression for all 4 cases
        pot_matrix, dep_matrix = np.zeros((2, 2)), np.zeros((2, 2))
        pot_matrix[0, 0] = assembly_cluster_diffs[assembly_cluster_diffs > 0].mean()
        pot_matrix[0, 1] = assembly_nc_diffs[assembly_nc_diffs > 0].mean()
        pot_matrix[1, 0] = non_assembly_cluster_diffs[non_assembly_cluster_diffs > 0].mean()
        pot_matrix[1, 1] = non_assembly_nc_diffs[non_assembly_nc_diffs > 0].mean()
        dep_matrix[0, 0] = -1 * assembly_cluster_diffs[assembly_cluster_diffs < 0].mean()
        dep_matrix[0, 1] = -1 * assembly_nc_diffs[assembly_nc_diffs < 0].mean()
        dep_matrix[1, 0] = -1 * non_assembly_cluster_diffs[non_assembly_cluster_diffs < 0].mean()
        dep_matrix[1, 1] = -1 * non_assembly_nc_diffs[non_assembly_nc_diffs < 0].mean()
        pot_matrices[assembly_label], dep_matrices[assembly_label] = pot_matrix, dep_matrix
    return sizes, pot_matrices, dep_matrices


def plot_grouped_diffs(sizes, pot_matrices, dep_matrices, fig_name):
    """For every assembly plots pie chart with total changes and 2 matrices
    with the ammount of potentiation and depression (in a 2x2 grid - see `group_diffs()` above)"""
    plt.rcParams["patch.edgecolor"] = "black"
    assembly_labels = np.sort(list(sizes.keys()))
    n = len(assembly_labels)
    min_pot = np.min([np.min(pot_matrix) for _, pot_matrix in pot_matrices.items()])
    max_pot = np.max([np.max(pot_matrix) for _, pot_matrix in pot_matrices.items()])
    min_dep = np.min([np.min(dep_matrix) for _, dep_matrix in dep_matrices.items()])
    max_dep = np.max([np.max(dep_matrix) for _, dep_matrix in dep_matrices.items()])

    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(3, n+1, width_ratios=[10 for i in range(n)] + [1])
    for i, assembly_label in enumerate(assembly_labels):
        ax = fig.add_subplot(gs[0, i])
        n_syns = np.sum(sizes[assembly_label])
        ax.pie(sizes[assembly_label], labels=["%.2f%%" % ratio for ratio in (100 * sizes[assembly_label] / n_syns)],
               colors=[RED, "lightgray", BLUE])
        ax.set_title("assembly %i\n(n = %i syns\nfrom 10 neurons)" % (assembly_label, n_syns))
        ax2 = fig.add_subplot(gs[1, i])
        i_pot = ax2.imshow(pot_matrices[assembly_label], cmap="Reds", aspect="auto", vmin=min_pot, vmax=max_pot)
        ax3 = fig.add_subplot(gs[2, i])
        i_dep = ax3.imshow(dep_matrices[assembly_label], cmap="Blues", aspect="auto", vmin=min_dep, vmax=max_dep)
        if i == 0:
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(["assembly", "non-assembly"])
            ax3.set_yticks([0, 1])
            ax3.set_yticklabels(["from assembly", "non assembly"])
        else:
            ax2.set_yticks([])
            ax3.set_yticks([])
        ax2.set_xticks([])
        ax3.set_xticks([0, 1])
        ax3.set_xticklabels(["clustered", "not clustered"], rotation=45)
    fig.colorbar(i_pot, cax=fig.add_subplot(gs[1, i+1]), label="Potentiated")
    fig.colorbar(i_dep, cax=fig.add_subplot(gs[2, i+1]), label="Depressed")
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
        sizes, pot_matrices, dep_matrices = group_diffs(syn_clusters, diffs)
        fig_name = os.path.join(FIGS_DIR, project_name, "syn_clust_plast_seed%i.png" % seed)
        plot_grouped_diffs(sizes, pot_matrices, dep_matrices, fig_name)


if __name__ == "__main__":
    project_name = "e0fbb0c8-07a4-49e0-be7d-822b2b2148fb"
    main(project_name)
