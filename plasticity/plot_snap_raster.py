"""
Plot SSCx hex_O1 rasters
author: András Ecker, last update: 06.2022
"""

import os
import time
import pickle
import numpy as np
from bluepysnap import Simulation
import utils
from plot_raster import plot_raster

FIGS_DIR = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/figures/sscx-analysis"


def get_tc_rates(sim_config, t_start, t_end):
    """Read VPM and POm spikes from spike_file(s)"""
    proj_spikes = utils.get_snap_tc_spikes(sim_config, t_start, t_end)
    proj_rates = {proj_name: utils.calc_rate(data["spike_times"], len(np.unique(data["spiking_gids"])), t_start, t_end)
                  for proj_name, data in proj_spikes.items()}
    if len(proj_rates):
        return proj_spikes, proj_rates
    else:
        return None, None


if __name__ == "__main__":
    project_name = "p_scan/P_FR0p6"
    t_start = 1500
    node_pop = "S1nonbarrel_neurons"

    sim_paths = utils.load_sim_paths(project_name)
    level_names = sim_paths.index.names
    utils.ensure_dir(os.path.join(FIGS_DIR, project_name))
    with open("raster_asth_Zenodo_O1.pkl", "rb") as f:
        raster_asthetics = pickle.load(f)

    for idx, sim_path in sim_paths.items():
        sim = Simulation(sim_path)
        t_end = sim.time_stop
        spike_times, spiking_gids = utils.get_spikes(sim, t_start, t_end, node_pop=node_pop)
        _, proj_rates = get_tc_rates(sim.config, t_start, t_end)
        fig_name = os.path.join(FIGS_DIR, project_name, "%sraster.png" % utils.midx2str(idx, level_names))
        plot_raster(spike_times, spiking_gids, proj_rates, raster_asthetics, t_start, t_end, fig_name)







