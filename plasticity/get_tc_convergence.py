"""
Convergence of TC fibers on cortical EXC cells
author: András Ecker, last update: 11.2021
"""

import os
from tqdm import tqdm
import numpy as np
from bluepy import Circuit
from bluepy.enums import Cell, Synapse
import utils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="ticks", context="notebook")
FIGS_DIR = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/figures/sscx-analysis"


def get_tc_convergence(pattern_gids, projection_name, circuit_target):
    """Gets convergence (number of synapses on postsynaptic cell) of TC fibers on cortical EXC cells"""
    exc_gids = c.cells.ids({"$target": circuit_target, Cell.SYNAPSE_CLASS: "EXC"})
    n_syns = {}
    for pattern_name, t_gids in tqdm(pattern_gids.items(), desc="Iterating over patterns"):
        post_gids = c.projection(projection_name).pathway_synapses(t_gids, exc_gids, [Synapse.POST_GID]).to_numpy()
        _, counts = np.unique(post_gids, return_counts=True)
        n_syns[pattern_name] = counts
    return n_syns


def _get_nsyn_range(n_syns_dict):
    """Concatenates convergence results and return overall min, max and 95% percentile"""
    pattern_names = list(n_syns_dict.keys())
    n_syns = n_syns_dict[pattern_names[0]]
    for pattern_name in pattern_names[1:]:
        n_syns = np.concatenate((n_syns, n_syns_dict[pattern_name]))
    return np.min(n_syns), np.max(n_syns), np.percentile(n_syns, 95)


def plot_tc_convergence(n_syns_dict, patterns_dir):
    """Plots TC convergence histograms"""
    utils.ensure_dir(patterns_dir)
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


if __name__ == "__main__":

    project_name = "cdf61143-0299-4a41-928d-b2cf0577d543"
    sim_paths = utils.load_sim_path(project_name)  # just to have a circuit
    c = Circuit(sim_paths.iloc[0])
    for seed in [12, 28]:
        pattern_gids, _, _, metadata = utils.load_patterns(project_name, seed)
        n_syns = get_tc_convergence(pattern_gids, metadata[0], metadata[1])
        plot_tc_convergence(n_syns, os.path.join(FIGS_DIR, project_name, "patterns_seed%i" % seed))






