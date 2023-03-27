"""
Aims to replicate the analysis of Varani et al. 2021:
Select L23 PCs with subthreshold responses to VPM stim, average their traces,
and see how this mean voltage trace changes when L4 PCs are optogenetically inhibited
author: András Ecker, last update: 03.2023
"""

import os
import numpy as np
from scipy.signal import find_peaks
from bluepy import Simulation
from bluepy.enums import Cell, Synapse
import utils
from plots import plot_all_voltages, plot_selected_voltages, plot_dvdt, plot_mean_voltages

FIGS_DIR = "/gpfs/bbp.cscs.ch/project/proj83/home/ecker/figures/sscx-analysis"


def split_traces(sim, gids, t_start, t_end):
    """Loads in voltage traces and splits them to spiking and non-spiking groups"""
    spike_times, spiking_gids = utils.get_spikes(sim, gids, t_start, t_end)
    rate = utils.calc_rate(spike_times, len(gids), t_start, t_end, bin_size=1)
    spiking_gids = np.unique(spiking_gids)
    non_spiking_gids = gids[np.isin(gids, spiking_gids, assume_unique=True, invert=True)]
    _, _, v_spiking = utils.get_voltages(sim, spiking_gids, t_start, t_end, True)
    non_spiking_gids, t, v_subtreshold = utils.get_voltages(sim, non_spiking_gids, t_start, t_end, False)
    return rate, spiking_gids, v_spiking, non_spiking_gids, v_subtreshold, t


def get_tc_gids(sim_config, t_start, t_end):
    """Finds VPM gids that elicit at least one spike"""
    _, spiking_gids = utils.get_tc_spikes(sim_config, t_start, t_end)
    return np.unique(spiking_gids)


def _find_dvdt_th_crossings(t, voltages, th=1):
    """Locates threshold crossings in the derivative of the voltage traces"""
    dt = t[1] - t[0]
    peak_times = []
    for i in range(voltages.shape[0]):
        peak_idx, _ = find_peaks(np.gradient(voltages[i, :], dt), height=th)
        peak_times.append(t[peak_idx])
    return peak_times


def get_responsive_gids(non_spiking_gids, t, v_subtreshold, stim_start, t_window, fig_dir=None, fig_name_tag=""):
    """Return gids which have EPSP like peaks in their subthreshold voltages (within `t_window` after `stim_start`)"""
    dvdt_peaks = _find_dvdt_th_crossings(t, v_subtreshold)
    if fig_dir:
        utils.ensure_dir(fig_dir)
    responsive_gids = []
    for (gid, voltage, peaks) in zip(non_spiking_gids, v_subtreshold, dvdt_peaks):
        if len(peaks):
            idx = np.where((stim_start < peaks) & (peaks < stim_start + t_window))[0]
            if len(idx):
                responsive_gids.append(gid)
                if fig_dir:
                    plot_dvdt(t, voltage, os.path.join(fig_dir, "%sa%i.png" % (fig_name_tag, gid)))
    return responsive_gids


def main(project_name, sim, t_start, stim_start, t_end, fig_name_tag, debug=False, t_window=20):
    c = sim.circuit
    # get subthreshold L23 PC voltages
    gids = c.cells.ids({"$target": sim.target, Cell.LAYER: [2, 3], Cell.SYNAPSE_CLASS: "EXC"})
    rate, _, v_spiking, non_spiking_gids, v_subtreshold, t = split_traces(sim, gids, t_start, t_end)
    fig_name = os.path.join(FIGS_DIR, project_name, "%sL23PC_vs.png" % fig_name_tag)
    plot_all_voltages(rate, v_spiking, v_subtreshold, t_start, t_end, fig_name)

    # filter L23 PCs that get synapses from spiking VPM fibers
    tc_gids = get_tc_gids(sim.config, stim_start, t_end)
    conns = c.projection("Thalamocortical_input_VPM")
    innervated_gids = conns.pathway_synapses(tc_gids, non_spiking_gids, Synapse.POST_GID).unique()
    idx = np.in1d(non_spiking_gids, innervated_gids, assume_unique=True)
    non_spiking_gids, v_subtreshold = non_spiking_gids[idx], v_subtreshold[idx, :]
    # filter L23 PCs that have detectable EPSPs following VPM stimulus onset
    fig_dir = os.path.join(FIGS_DIR, project_name) if debug else None
    responsive_gids = get_responsive_gids(non_spiking_gids, t, v_subtreshold, stim_start, t_window, fig_dir, fig_name_tag)
    assert len(responsive_gids), "No gids remained after filtering steps"
    idx = np.in1d(non_spiking_gids, responsive_gids, assume_unique=True)
    non_spiking_gids, v_subtreshold = non_spiking_gids[idx], v_subtreshold[idx, :]
    fig_name = os.path.join(FIGS_DIR, project_name, "%sL23PC_selected_vs.png" % fig_name_tag)
    plot_selected_voltages(v_subtreshold, t_start, t_end, fig_name)

    return non_spiking_gids, t, np.mean(v_subtreshold, axis=0)


def opto_main(project_name, sim, selected_gids, t_start, t_end, plot_l4, fig_name_tag):
    c = sim.circuit
    if plot_l4:  # check the effect of optogenetic stim. on L4 PCs
        gids = c.cells.ids({"$target": sim.target, Cell.LAYER: 4, Cell.SYNAPSE_CLASS: "EXC"})
        rate, _, v_spiking, _, v_subtreshold, _ = split_traces(sim, gids, t_start, t_end)
        fig_name = os.path.join(FIGS_DIR, project_name, "%sL4PC_vs.png" % fig_name_tag)
        plot_all_voltages(rate, v_spiking, v_subtreshold, t_start, t_end, fig_name)
    # get subthreshold L23 PC voltages (just to make sure that the `selected_gids` are still subthreshold)
    gids = c.cells.ids({"$target": sim.target, Cell.LAYER: [2, 3], Cell.SYNAPSE_CLASS: "EXC"})
    _, _, _, non_spiking_gids, v_subtreshold, _ = split_traces(sim, gids, t_start, t_end)
    v_subtreshold = v_subtreshold[np.in1d(non_spiking_gids, selected_gids, assume_unique=True), :]
    fig_name = os.path.join(FIGS_DIR, project_name, "%sL23PC_selected_vs.png" % fig_name_tag)
    plot_selected_voltages(v_subtreshold, t_start, t_end, fig_name)
    return np.mean(v_subtreshold, axis=0)


if __name__ == "__main__":
    project_name = "84305a68-0537-4d55-807d-a03700c94204"  # "34f953d9-3521-42f7-87d9-f766acd2cfa7"
    t_start, stim_start, t_end = 1800, 2000, 3000
    sim_paths = utils.load_sim_paths(project_name)
    level_names = sim_paths.index.names
    assert level_names[0] in ["opto_depol_pct", "from_layer"], "Atm. `opto_depol_pct` or `from_layer`" \
                                                               "has to be the first index in the campaign"
    tag = level_names[0]
    plot_l4 = True if tag == "opto_depol_pct" else False
    cm_name = "Greens" if tag == "opto_depol_pct" else ""  # leaving it empty will use seaborn's default colormap
    utils.ensure_dir(os.path.join(FIGS_DIR, project_name))

    if len(level_names) == 1:
        idx = sim_paths.index.to_numpy()
        assert 0 in idx, "baseline sim (without modifications) has to be present"
        sim = Simulation(sim_paths.loc[0])
        selected_gids, t, mean_ctrl_voltage = main(project_name, sim, t_start, stim_start, t_end, "%s0_" % tag)

        mean_opto_voltages = {}
        for tag_val in idx[np.nonzero(idx)]:
            sim = Simulation(sim_paths.loc[tag_val])
            mean_opto_voltages[tag_val] = opto_main(project_name, sim, selected_gids, t_start, t_end, plot_l4,
                                                    "%s%i_" % (tag, tag_val))
        plot_mean_voltages(t, mean_ctrl_voltage, mean_opto_voltages, cm_name,
                           os.path.join(FIGS_DIR, project_name, "mean_L23PC_voltages.png"))

    elif len(level_names) == 2:
        tag_vals = np.unique(sim_paths.index.get_level_values(tag))
        key = level_names[1]
        vals = np.unique(sim_paths.index.get_level_values(key))
        for val in vals:
            sim = Simulation(sim_paths.loc[0, val])
            selected_gids, t, mean_ctrl_voltage = main(project_name, sim, t_start, stim_start, t_end,
                                                       "%s0_%s%i_" % (tag, key, val))

            mean_opto_voltages = {}
            for tag_val in tag_vals[np.nonzero(tag_vals)]:
                sim = Simulation(sim_paths.loc[tag_val, val])
                mean_opto_voltages[tag_val] = opto_main(project_name, sim, selected_gids, t_start, t_end, plot_l4,
                                                        "%s%i_%s%i_" % (tag, tag_val, key, val))
            plot_mean_voltages(t, mean_ctrl_voltage, mean_opto_voltages, cm_name,
                               os.path.join(FIGS_DIR, project_name, "%s%i_mean_L23PC_voltages.png" % (key, val)))

    # TODO: deal with more than one extra coordinates ...






