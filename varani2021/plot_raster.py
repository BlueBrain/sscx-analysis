"""
Plot SSCx (hex_O1) rasters
author: András Ecker, last update: 02.2022
"""

import os
import pickle
import numpy as np
from bluepy import Simulation
import utils
from plots import plot_raster

FIGS_DIR = "/gpfs/bbp.cscs.ch/project/proj83/home/ecker/figures/sscx-analysis"


def get_tc_rates(sim, t_start, t_end):
    """Read VPM and POm spikes from spike replay file"""
    spike_times, spiking_gids = utils.get_tc_spikes(sim, t_start, t_end)
    vpm_gids, pom_gids = utils.load_tc_gids(os.path.split(sim.config.Run_Default.CurrentDir)[0])
    proj_rate_dict = {}
    for proj_name, proj_gids in zip(["VPM", "POm"], [vpm_gids, pom_gids]):
        if proj_gids is not None:
            mask = np.isin(spiking_gids, proj_gids)
            if mask.sum() > 0:
                proj_spike_times, proj_spiking_gids = spike_times[mask], spiking_gids[mask]
                proj_rate_dict[proj_name] = utils.calc_rate(proj_spike_times, len(np.unique(proj_spiking_gids)),
                                                            t_start, t_end)
    return proj_rate_dict if len(proj_rate_dict) else None


if __name__ == "__main__":
    project_name = "84f11a52-5b9d-42a3-9765-cedece9771a4"
    t_start = 1500

    sim_paths = utils.load_sim_paths(project_name)
    level_names = sim_paths.index.names
    utils.ensure_dir(os.path.join(FIGS_DIR, project_name))
    with open("raster_asth.pkl", "rb") as f:
        raster_asthetics = pickle.load(f)

    for idx, sim_path in sim_paths.iteritems():
        sim = Simulation(sim_path)
        t_end = sim.t_end
        spike_times, spiking_gids = utils.get_spikes(sim, None, t_start, t_end)
        proj_rate_dict = get_tc_rates(sim, t_start, t_end)
        fig_name = os.path.join(FIGS_DIR, project_name, "%sraster.png" % utils.midx2str(idx, level_names))
        plot_raster(spike_times, spiking_gids, proj_rate_dict, raster_asthetics, t_start, t_end, fig_name)






