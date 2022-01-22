"""
Loads synapse clusters saved by `assemblyfire` and checks their total changes in the synapse report
author: András Ecker, last update: 01.2022
"""

import os
import utils
'''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns

sns.set(style="ticks", context="notebook")
'''
FIGS_DIR = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/figures/sscx-analysis"
RED, BLUE = "#e32b14", "#3271b8"


def main(project_name):
    report_name = "rho"
    sim_paths = utils.load_sim_paths(project_name)
    level_names = sim_paths.index.names
    assert len(level_names) == 1 and level_names[0] == "seed"
    utils.ensure_dir(os.path.join(FIGS_DIR, project_name))

    for seed, sim_path in sim_paths.iteritems():
        syn_clusters, gids = utils.load_synapse_clusters(seed, sim_path)
        diffs = utils.get_synapse_changes(sim_path, report_name, gids)


if __name__ == "__main__":
    project_name = "e0fbb0c8-07a4-49e0-be7d-822b2b2148fb"
    main(project_name)
