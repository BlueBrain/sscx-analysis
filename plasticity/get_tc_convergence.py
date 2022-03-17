"""
Convergence of TC fibers on cortical EXC cells
author: András Ecker, last update: 11.2021
"""

import os
from tqdm import tqdm
import numpy as np
from bluepy import Circuit
from bluepy.enums import Cell, Synapse
from utils import load_sim_paths, load_patterns
from plots import plot_tc_convergence

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


if __name__ == "__main__":
    project_name = "cdf61143-0299-4a41-928d-b2cf0577d543"
    sim_paths = load_sim_paths(project_name)  # just to have a circuit
    c = Circuit(sim_paths.iloc[0])
    for seed in [12, 28]:
        pattern_gids, _, _, metadata = load_patterns(project_name, seed)
        n_syns = get_tc_convergence(pattern_gids, metadata[0], metadata[1])
        plot_tc_convergence(n_syns, os.path.join(FIGS_DIR, project_name, "patterns_seed%i" % seed))






