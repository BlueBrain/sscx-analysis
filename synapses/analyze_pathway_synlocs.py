"""
Gets synapse (and optionally apoosition) path distances and Sholl analyis of the postsyn population
last modified: András Ecker 09.2022
"""

import os
from tqdm import tqdm
import numpy as np
import neurom as nm
from bluepy import Circuit
from bluepy.enums import Cell, Synapse
from bluepy.utils import take_n
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="ticks", context="notebook")
FIGS_PATH = "/gpfs/bbp.cscs.ch/project/proj83/home/ecker/figures"


def get_pathdists(c, pre_gids, post_gids):
    """Gets path distances (from soma to synapse) for a given pre-post pathway"""
    return c.connectome.pathway_synapses(pre_gids, post_gids, Synapse.POST_NEURITE_DISTANCE).to_numpy()


def sholl_analysis(c, gids, bins):
    """Performs sholl analysis with pathdistance (instead of Euclidean)"""
    crossings = np.array(nm.get("sholl_crossings", nm.load_morphology(c.morph.get_filepath(gids[0])),
                                radii=bins, distance_type="path"))
    for gid in tqdm(gids[1:]):  # TODO: try to do this w/o iterating over stuff
        crossings += np.array(nm.get("sholl_crossings", nm.load_morphology(c.morph.get_filepath(gid)),
                                 radii=bins, distance_type="path"))
    return crossings


def plot_nsyns_and_sholl(dists, struct_dists, crossings, bin_edges, fig_name):
    """Histplot number of synapses and Sholl crossings vs. path distances"""
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    fig = plt.figure(figsize=(9., 8.))
    ax = fig.add_subplot(3, 1, 1)
    ax.hist(struct_dists, bins=bin_edges)
    ax.set_ylabel("#Appositions")
    ax.set_title("L5 MC - L5 TTPC")
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.hist(dists, bins=bin_edges, label="functional")
    ax2.set_ylabel("#Synapses")
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.bar(bin_centers, crossings[1:], width=bin_edges[1])
    ax3.set_xlabel("Path distance (um)")
    ax3.set_ylabel("#Sholl crossings")
    sns.despine(trim=True)
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    c = Circuit("/gpfs/bbp.cscs.ch/project/proj83/circuits/Bio_M/20200805/CircuitConfig")
    nsamples = 100
    pre_gids = c.cells.ids({"$target": "Mosaic", Cell.MTYPE: "L5_MC"})
    post_gids = take_n(c.cells.ids({"$target": "central_column_4_region_700um",
                                    Cell.MTYPE: {"$regex": "L5_TPC:(A|B)"}}), nsamples)
    dists = get_pathdists(c, pre_gids, post_gids)
    bins = np.arange(0, 1400, 50)
    crossings = sholl_analysis(c, post_gids, bins)
    c = Circuit("/gpfs/bbp.cscs.ch/project/proj83/circuits/Bio_M/20200805/CircuitConfig_struct")
    struct_dists = get_pathdists(c, pre_gids, post_gids)
    plot_nsyns_and_sholl(dists, struct_dists, crossings, bins,
                         os.path.join(FIGS_PATH, "L5_MC-L5_TTPC_nsysns_Sholl.png"))
