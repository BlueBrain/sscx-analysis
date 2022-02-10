"""
Plots voltage collage and subthreshold EPSP amplitudes of L2/3 PCs (to replicate the "analysis" of Varani et al. 2021)
author: András Ecker, last update: 02.2022
"""

import os
import numpy as np
from efel import getFeatureValues
from bluepy import Simulation
from bluepy.enums import Cell
from plots import plot_voltages, plot_all_voltages, plot_epsps_amplitudes

SPIKE_TH = -30  # -30 mV is NEURON's built in spike threshold
SIMS_DIR = "/gpfs/bbp.cscs.ch/project/proj96/scratch/home/ecker/simulations"
FIGS_DIR = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/figures/sscx-analysis"


def _ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_spikes(sim, gids, t_start, t_end):
    """Extracts spikes with bluepy"""
    if gids is not None:
        spikes = sim.spikes.get(gids=gids, t_start=t_start, t_end=t_end)
    else:
        spikes = sim.spikes.get(t_start=t_start, t_end=t_end)
    return spikes.index.to_numpy(), spikes.to_numpy()


def calc_rate(spike_times, N, t_start, t_end, bin_size):
    """Calculates populational firing rate"""
    t_bins = np.arange(t_start, t_end+bin_size, bin_size)
    rate, _ = np.histogram(spike_times, t_bins)
    return rate / (N*1e-3*bin_size)  # *1e-3 ms to s conversion


def _clean_voltages(gids, voltages, spiking):
    """Removes non-spiking voltage traces if spiking is True or the spiking ones if it's False
    (currently we detect spikes at the AIS and report the voltage of the soma...)
    and reorders the remaining ones based on the mean value for better plotting"""
    max_v = np.max(voltages, axis=0)
    idx = np.where(max_v >= SPIKE_TH)[0] if spiking else np.where(max_v < SPIKE_TH)[0]
    voltages = voltages[:, idx]
    gids = gids[idx]
    sort_idx = np.argsort(np.mean(voltages, axis=0))
    return gids[sort_idx], voltages[:, sort_idx]


def get_voltages(sim, gids, t_start, t_end, spiking):
    """Extracts reported soma voltages with bluepy and cleanes them (see `_clean_voltages()` above)"""
    report = sim.report("soma")
    if gids is not None:
        # data = report.get(gids=gids, t_start=t_start, t_end=t_end)
        data = report.get(t_start=t_start, t_end=t_end)
        data = data[gids]
    else:
        data = report.get(t_start=t_start, t_end=t_end)
    t = data.index.to_numpy()
    voltages = data.to_numpy()
    gids, voltages =  _clean_voltages(gids, voltages, spiking)
    return gids, t, voltages.T


def efel_traces(t, voltages, t_window):
    """Get traces in the format expected by efel.getFeatureValues."""
    t_window[0] = t[0] if t_window[0] < t[0] else t_window[0]
    t_window[1] = t[-1] if t_window[1] > t[-1] else t_window[1]
    return [{"T": t, "V": v, "stim_start": [t_window[0]], "stim_end": [t_window[1]]} for v in voltages]


def get_epsp_amplitudes(t, voltages, t_window):
    """Get EPSP amplitudes in `vs` in `t_window` using `efel`"""
    traces = efel_traces(t, voltages, t_window)
    traces_results = getFeatureValues(traces, ["maximum_voltage", "voltage_base"])
    epsps = [np.abs(trace_result["maximum_voltage"][0] - trace_result["voltage_base"][0])
             for trace_result in traces_results]
    return np.asarray(epsps)


def get_n_highest_epsp_voltages(gids, voltages, epsps, n=10):
    """Returns the gids and voltage traces corresponding to the `n` highest EPSP amplitudes"""
    sort_idx = np.argsort(epsps)[::-1]
    gids, voltages = gids[sort_idx], voltages[sort_idx, :]
    return gids[:n], voltages[:n, :]


def main():
    project_name = "LayerWiseEShotNoise_WhiskerFlick"
    t_start, t_end = 2000, 2050
    _ensure_dir(os.path.join(FIGS_DIR, project_name))

    # get L23 PC spikes and rate
    sim = Simulation(os.path.join(SIMS_DIR, project_name, "BlueConfig"))
    l23pc_gids = sim.circuit.cells.ids({"$target": sim.target, Cell.LAYER: [2, 3], Cell.SYNAPSE_CLASS: "EXC"})
    spike_times, spiking_gids = get_spikes(sim, l23pc_gids, t_start, t_end)
    l23pc_rate = calc_rate(spike_times, len(l23pc_gids), t_start, t_end, bin_size=1)
    # split L23 PCs and get soma voltages for both groups
    spiking_l23pc_gids = np.unique(spiking_gids)
    non_spiking_l23pc_gids = l23pc_gids[np.isin(l23pc_gids, spiking_l23pc_gids, assume_unique=True, invert=True)]
    _, _, v_spiking = get_voltages(sim, spiking_l23pc_gids, t_start, t_end, True)
    non_spiking_l23pc_gids, t, v_subtreshold = get_voltages(sim, non_spiking_l23pc_gids, t_start, t_end, False)
    fig_name = os.path.join(FIGS_DIR, project_name, "L23PC_voltages.png")
    plot_all_voltages(v_spiking, v_subtreshold, l23pc_rate, t_start, t_end, fig_name)
    # calculate EPSP amplitudes for the non-spiking group and plots the voltage trace corresponding to the 10 highest
    v_subtreshold = v_subtreshold[:5000, :]  # sample because it would run forever
    epsps = get_epsp_amplitudes(t, v_subtreshold, [2010, 2020])
    fig_name = os.path.join(FIGS_DIR, project_name, "L23PC_subth_epsps.png")
    plot_epsps_amplitudes(epsps, fig_name)
    plot_gids, plot_vs = get_n_highest_epsp_voltages(non_spiking_l23pc_gids, v_subtreshold, epsps)
    fig_name = os.path.join(FIGS_DIR, project_name, "most_active_subth_L23PC_vs.png")
    plot_voltages(plot_gids, t, plot_vs, fig_name)


if __name__ == "__main__":
    main()



