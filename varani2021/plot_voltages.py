"""
Plots voltage collage and subthreshold EPSP amplitudes of L2/3 PCs (to replicate the "analysis" of Varani et al. 2021)
author: András Ecker, last update: 01.2023
"""

import os
import numpy as np
from bluepy import Simulation
from bluepy.enums import Cell
import utils
from plots import plot_voltages, plot_all_voltages, plot_epsps_amplitudes

FIGS_DIR = "/gpfs/bbp.cscs.ch/project/proj83/home/ecker/figures/sscx-analysis"


def filter_tc_gids(sim_config, t_start, stim_dur, t_end):
    """Finds VPM gids that elicit at least 1 spike for stim. and 1 rebound/'off' one for stim. withould"""
    spike_times, spiking_gids = utils.get_tc_spikes(sim_config, t_start, t_end)
    gids = np.unique(spiking_gids[spike_times < t_start + stim_dur])
    off_gids, off_counts = np.unique(spiking_gids[spike_times > t_start + stim_dur], return_counts=True)
    return gids[np.in1d(gids, off_gids[off_counts == 1], assume_unique=True)]


def get_epsp_amplitudes(t, voltages, t_windows):
    """Get EPSP amplitudes in `vs` in `t_window` using `efel`"""
    t_start = t[0] if t_windows[0] < t[0] else t_windows[0]
    t_end = t[-1] if t_windows[2] > t[-1] else t_windows[2]
    baselines = np.mean(voltages[:, (t_start <= t) & (t <= t_windows[1])], axis=1)
    peaks = np.max(voltages[:, (t_windows[1] <= t) & (t <= t_end)], axis=1)
    return peaks - baselines


def get_n_highest_epsp_voltages(gids, voltages, epsps, n=10):
    """Returns the gids and voltage traces corresponding to the `n` highest EPSP amplitudes"""
    sort_idx = np.argsort(epsps)[::-1]
    gids, voltages = gids[sort_idx], voltages[sort_idx, :]
    return gids[:n], voltages[:n, :]


def main(sim, fig_name_tag, t_start=2000, stim_dur=500, t_end=3000):
    # get L23 PC spikes and rate
    c = sim.circuit
    gids = c.cells.ids({"$target": sim.target, Cell.LAYER: [2, 3], Cell.SYNAPSE_CLASS: "EXC"})
    spike_times, spiking_gids = utils.get_spikes(sim, gids, t_start, t_end)
    rate = utils.calc_rate(spike_times, len(gids), t_start, t_end, bin_size=1)
    # split L23 PCs and get soma voltages for both groups
    spiking_gids = np.unique(spiking_gids)
    non_spiking_gids = gids[np.isin(gids, spiking_gids, assume_unique=True, invert=True)]
    _, _, v_spiking = utils.get_voltages(sim, spiking_gids, t_start, t_end, True)
    non_spiking_gids, t, v_subtreshold = utils.get_voltages(sim, non_spiking_gids, t_start, t_end, False)
    fig_name = os.path.join(FIGS_DIR, project_name, "%sL23PC_vs.png" % fig_name_tag)
    plot_all_voltages(v_spiking, v_subtreshold, rate, t_start, t_end, fig_name)

    # filter L23 PCs that get synapses for properly spiking VPM fibers
    conns = c.projection("Thalamocortical_input_VPM")
    tc_gids = filter_tc_gids(sim.config, t_start, stim_dur, t_end)
    innervated_gids = conns.pathway_synapses(tc_gids, non_spiking_gids, Synapse.POST_GID).unique()
    idx = np.in1d(non_spiking_gids, innervated_gids, assume_unique=True)
    non_spiking_gids, v_subtreshold = non_spiking_gids[idx], v_subtreshold[idx, :]
    
    '''
    # calculate EPSP amplitudes for the non-spiking group and plots the voltage trace corresponding to the 10 highest
    epsp_t_windows = [1960, 2010, 2060]
    epsps = get_epsp_amplitudes(t, v_subtreshold, epsp_t_windows)
    fig_name = os.path.join(FIGS_DIR, project_name, "%sL23PC_subth_epsps.png" % fig_name_tag)
    plot_epsps_amplitudes(epsps, fig_name)
    plot_gids, plot_vs = get_n_highest_epsp_voltages(non_spiking_l23pc_gids, v_subtreshold, epsps)
    fig_name = os.path.join(FIGS_DIR, project_name, "%sL23PC_subth_vsamples.png" % fig_name_tag)
    plot_voltages(plot_gids, t, plot_vs, fig_name)
    '''


if __name__ == "__main__":
    project_name = "50756f8e-641e-4a39-9d22-6b1d9cbef323"
    sim_paths = utils.load_sim_paths(project_name)
    level_names = sim_paths.index.names
    utils.ensure_dir(os.path.join(FIGS_DIR, project_name))

    for idx, sim_path in sim_paths.items():
        sim = Simulation(sim_path)
        fig_name_tag = utils.midx2str(idx, level_names)
        main(sim, fig_name_tag)



