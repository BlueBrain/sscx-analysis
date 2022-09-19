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


def get_pathdists(c, pre_mtype="L5_MC", post_mtype={"$regex": "L5_TPC:(A|B)"}, nsamples=100):
    """Gets path distances (from soma to synapse) for a given pre-post pathway"""
    pre_gids = c.cells.ids({"$target": "Mosaic", Cell.MTYPE: pre_mtype})
    sample_gids = take_n(c.cells.ids({"$target": "central_column_4_region_700um", Cell.MTYPE: post_mtype}), nsamples)
    dists = c.connectome.pathway_synapses(pre_gids, sample_gids, Synapse.POST_NEURITE_DISTANCE).to_numpy()
    return sample_gids, dists


def sholl_analysis(gids, bins):
    """Performs sholl analysis with pathdistance (instead of Euclidean)"""
    crossings = np.array(nm.get("sholl_crossings", nm.load_morphology(c.morph.get_filepath(gids[0])),
                                radii=bins, distance_type="path"))
    for gid in tqdm(gids[1:]):  # TODO: try to do this w/o iterating over stuff
        crossings += np.array(nm.get("sholl_crossings", nm.load_morphology(c.morph.get_filepath(gid)),
                                 radii=bins, distance_type="path"))
    return crossings


def plot_nsyns_and_sholl(dists, crossings, bin_edges, fig_name):
    """Histplot number of synapses and Sholl crossings vs. path distances"""
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    fig = plt.figure(figsize=(9., 6.))
    ax = fig.add_subplot(2, 1, 1)
    ax.hist(dists, bins=bin_edges)
    ax.set_ylabel("#Synapses")
    ax.set_title("L5 MC - L5 TTPC")
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.bar(bin_centers, crossings[1:], width=bin_edges[1])
    ax2.set_xlabel("Path distance (um)")
    ax2.set_ylabel("#Sholl crossings")
    sns.despine(trim=True)
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    c = Circuit("/gpfs/bbp.cscs.ch/project/proj83/circuits/Bio_M/20200805/CircuitConfig")
    bins = np.arange(0, 1500, 75)
    gids, dists = get_pathdists(c)
    crossings = sholl_analysis(gids, bins)
    plot_nsyns_and_sholl(dists, crossings, bins, os.path.join(FIGS_PATH, "L5_MC-L5_TTPC_nsysns_Sholl.png"))
