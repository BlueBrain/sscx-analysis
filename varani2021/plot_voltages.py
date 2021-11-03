# -*- coding: utf-8 -*-
"""
Plots voltage collage (of L2/3 PCs to replicate the "analysis" of Variani et al. 2021)
author: András Ecker, last update: 11.2021
"""

import os
import numpy as np
from bluepy import Simulation
from bluepy.enums import Cell
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set(style="ticks", context="notebook")
RED = "#e32b14"
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
    return spikes.index.to_numpy(), spikes.values


def get_voltages(sim, gids, t_start, t_end):
    """Extracts reported soma voltages with bluepy"""
    report = sim.report("soma")
    if gids is not None:
        # data = report.get(gids=gids, t_start=t_start, t_end=t_end)
        data = report.get(t_start=t_start, t_end=t_end)
        data = data[gids]
    else:
        data = report.get(t_start=t_start, t_end=t_end)
    return data.index.to_numpy(), data.values


def _clean_voltages(voltages, spiking):
    """Removes non-spiking voltage traces if spiking is True or the spiking ones if it's False
    (currently we detect spikes at the AIS and report the voltage of the soma...)
    and reorders the remaining ones based on the mean value for better plotting"""
    max_v = np.max(voltages, axis=0)
    if spiking:
        voltages = voltages[:, max_v > SPIKE_TH]
    else:
        voltages = voltages[:, max_v < SPIKE_TH]
    idx = np.argsort(np.mean(voltages, axis=0))
    return voltages[:, idx]


def calc_rate(spike_times, N, t_start, t_end, bin_size):
    """Calculates populational firing rate"""
    t_bins = np.arange(t_start, t_end+bin_size, bin_size)
    rate, _ = np.histogram(spike_times, t_bins)
    return rate / (N*1e-3*bin_size)  # *1e-3 ms to s conversion


def plot_voltages(v_spiking, v_subtreshold, rate, t_start, t_end, fig_name):
    """Plots voltage collages of spiking and non-spiking cells"""
    t_rate = np.linspace(t_start, t_end, len(rate))
    xticks = np.linspace(t_start, t_end, 6).astype(int)
    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 4, 15], width_ratios=[69, 1])
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t_rate, rate, color=RED)
    ax.fill_between(t_rate, np.zeros_like(t_rate), rate, color=RED, alpha=0.1)
    ax.set_xticks(xticks)
    ax.set_xlim(t_start, t_end)
    ax.set_ylabel("Rate (Hz)")
    sns.despine(ax=ax)
    ax2 = fig.add_subplot(gs[1, 0])
    i2 = ax2.imshow(v_spiking.T, cmap="inferno", aspect="auto", origin="lower")
    plt.colorbar(i2, cax=fig.add_subplot(gs[1, 1]))
    ax2.set_ylabel("Spiking gids")
    ax2.set_xticks(np.linspace(0, v_spiking.shape[0], 6).astype(int))
    ax2.set_xticklabels(xticks)
    ax3 = fig.add_subplot(gs[2, 0])
    i3 = ax3.imshow(v_subtreshold.T, cmap="inferno", aspect="auto", origin="lower")
    cbar = plt.colorbar(i3, cax=fig.add_subplot(gs[2, 1]))
    cbar.set_label("Voltage (mV)")
    ax3.set_ylabel("Non-spiking gids")
    ax3.set_xlabel("Time (ms)")
    ax3.set_xticks(np.linspace(0, v_subtreshold.shape[0], 6).astype(int))
    ax3.set_xticklabels(xticks)
    fig.align_ylabels()
    gs.tight_layout(fig, h_pad=0.2, w_pad=0.2)
    fig.savefig(fig_name, bbox_inches="tight", dpi=100)
    plt.close(fig)


def main():
    project_name = "LayerWiseEShotNoise_WhiskerFlick"
    t_start, t_end = 2000, 2050
    _ensure_dir(os.path.join(FIGS_DIR, project_name))

    sim = Simulation(os.path.join(SIMS_DIR, project_name, "BlueConfig"))
    l23pc_gids = sim.circuit.cells.ids({"$target": sim.target, Cell.LAYER: [2, 3], Cell.SYNAPSE_CLASS: "EXC"})
    spike_times, spiking_gids = get_spikes(sim, l23pc_gids, t_start, t_end)
    l23pc_rate = calc_rate(spike_times, len(l23pc_gids), t_start, t_end, bin_size=1)
    spiking_l23pc_gids = np.unique(spiking_gids)
    non_spiking_l23pc_gids = l23pc_gids[np.isin(l23pc_gids, spiking_l23pc_gids, assume_unique=True, invert=True)]
    _, v_spiking = get_voltages(sim, spiking_l23pc_gids, t_start, t_end)
    _, v_subtreshold = get_voltages(sim, non_spiking_l23pc_gids, t_start, t_end)

    fig_name = os.path.join(FIGS_DIR, project_name, "L23PC_voltages.png")
    plot_voltages(_clean_voltages(v_spiking, True), _clean_voltages(v_subtreshold, False), l23pc_rate,
                  t_start, t_end, fig_name)


if __name__ == "__main__":
    main()



