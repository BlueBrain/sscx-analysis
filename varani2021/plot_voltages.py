"""
Plots voltage collage and subthreshold EPSP amplitudes of L2/3 PCs (to replicate the "analysis" of Varani et al. 2021)
author: András Ecker, last update: 01.2023
"""

import os
import numpy as np
from scipy.signal import find_peaks
from bluepy import Simulation
from bluepy.enums import Cell, Synapse
import utils
from plots import plot_all_voltages, plot_selected_voltages, plot_dvdt

FIGS_DIR = "/gpfs/bbp.cscs.ch/project/proj83/home/ecker/figures/sscx-analysis"


def filter_tc_gids(sim_config, t_spikes, t_end):
    """Finds VPM gids that elicit at least 1 spike for stim. and 1 least one for rebound/'off' one for stim. withould"""
    spike_times, spiking_gids = utils.get_tc_spikes(sim_config, t_spikes[0], t_end)
    gids = np.unique(spiking_gids[spike_times < t_spikes[1]])
    off_gids = np.unique(spiking_gids[t_spikes[1] < spike_times])
    return gids[np.in1d(gids, off_gids, assume_unique=True)]


def find_dvdt_th_crossings(t, voltages, th=1):
    """Locates threshold crossings in the derivative of the voltage traces"""
    dt = t[1] - t[0]
    peak_times = []
    for i in range(voltages.shape[0]):
        peak_idx, _ = find_peaks(np.gradient(voltages[i, :], dt), height=th)
        peak_times.append(t[peak_idx])
    return peak_times


def main(sim, fig_name_tag, t_start=1950, t_end=3000, t_spikes=[2000, 2500], t_window=20, debug=False):
    if debug:
        fig_dir = os.path.join(FIGS_DIR, project_name, "debug")
        utils.ensure_dir(fig_dir)
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
    tc_gids = filter_tc_gids(sim.config, t_spikes, t_end)
    conns = c.projection("Thalamocortical_input_VPM")
    innervated_gids = conns.pathway_synapses(tc_gids, non_spiking_gids, Synapse.POST_GID).unique()
    idx = np.in1d(non_spiking_gids, innervated_gids, assume_unique=True)
    non_spiking_gids, v_subtreshold = non_spiking_gids[idx], v_subtreshold[idx, :]
    # filter L23 PCs that have detectable EPSPs following the VPM spikes
    dvdt_peaks = find_dvdt_th_crossings(t, v_subtreshold)
    responsive_gids = []
    for (gid, voltage, peaks) in zip(non_spiking_gids, v_subtreshold, dvdt_peaks):
        if len(peaks) == 2:
            if peaks[0] < t_spikes[0] + t_window and t_spikes[1] < peaks[1] and peaks[1] < t_spikes[1] + t_window:
                responsive_gids.append(gid)
                if debug:
                    fig_name = os.path.join(fig_dir, "%sa%i.png" % (fig_name_tag, gid))
                    plot_dvdt(t, voltage, fig_name)

    assert len(responsive_gids), "no gids remained after filtering steps"
    idx = np.in1d(non_spiking_gids, responsive_gids, assume_unique=True)
    non_spiking_gids, v_subtreshold = non_spiking_gids[idx], v_subtreshold[idx, :]
    mean_voltage = np.mean(v_subtreshold, axis=0)  # TODO: figure out how to work with this...
    fig_name = os.path.join(FIGS_DIR, project_name, "%sselected_L23PC_vs.png" % fig_name_tag)
    plot_selected_voltages(v_subtreshold, t_start, t_end, fig_name)


if __name__ == "__main__":
    project_name = "50756f8e-641e-4a39-9d22-6b1d9cbef323"
    sim_paths = utils.load_sim_paths(project_name)
    level_names = sim_paths.index.names
    utils.ensure_dir(os.path.join(FIGS_DIR, project_name))

    for idx, sim_path in sim_paths.items():
        sim = Simulation(sim_path)
        fig_name_tag = utils.midx2str(idx, level_names)
        main(sim, fig_name_tag)



