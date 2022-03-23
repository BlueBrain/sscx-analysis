"""
Reruns selected cells in BGLibPy to have a better temporal resolution of the synapse report
(and reports 'spine' voltage as well - to be able to look for nonlinear dendritic events)
author: András Ecker, last update: 03.2023
"""

import os
import numpy as np
import bglibpy
import utils
from plots import plot_bglibpy_trace

FIGS_DIR = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/figures/sscx-analysis"


def rerun_single_cell(bc, gid, t_stop=None, return_orig_v=False):
    """Reruns a single cell with all its inputs from the network simulation"""
    ssim = bglibpy.SSim(bc, record_dt=0.1)
    # manually add TC spikes (as BGLibPy doesn't load spikes from the SpikeFile in the BlueConfig)
    proj_spike_trains = utils.get_stim_spikes(ssim.bc)
    # instead of all the cell's synapses use only the ones that originate from the sim's target
    # and the ones from the active TC fibers (TC fibers don't have minis - so no need to add all TC synapses)
    pre_gids = np.concatenate([ssim.bc_simulation.target_gids, np.asarray(list(proj_spike_trains.keys()))])
    # instantiate gid with replay on all synapses and the same stim as it gets in the network simulation
    ssim.instantiate_gids([gid], add_synapses=True, add_projections=True, add_minis=True, intersect_pre_gids=pre_gids,
                          add_stimuli=True, add_replay=True, pre_spike_trains=proj_spike_trains)
    ssim.run(t_stop=t_stop)
    t = ssim.get_time_trace()
    v = ssim.get_voltage_trace(gid)
    if return_orig_v:
        t_nd = ssim.get_mainsim_time_trace()
        v_nd = ssim.get_mainsim_voltage_trace(gid)
        ssim.delete()
        if t_stop is not None:
            idx = np.where(t_nd <= t_stop)[0]
            t_nd, v_nd = t_nd[idx], v_nd[idx]
        return t, v, t_nd, v_nd
    else:
        ssim.delete()
        return t, v


def main(project_name):
    sim_paths = utils.load_sim_paths(project_name)
    seed = 31
    gid = 3652313
    sim_path = sim_paths.loc[31]
    t, v, t_nd, v_nd = rerun_single_cell(sim_path, gid, t_stop=5000, return_orig_v=True)
    fig_name = os.path.join(FIGS_DIR, project_name, "seed%i_a%i_bglibpy_trace.png" % (seed, gid))
    plot_bglibpy_trace(t, v, t_nd, v_nd, fig_name)


if __name__ == "__main__":
    project_name = "e0fbb0c8-07a4-49e0-be7d-822b2b2148fb"
    main(project_name)






