"""
SSCx related utility functions (most of them deal with the custom directory and file structure)
author: András Ecker, last update: 01.2023
"""

import os
import warnings
import pickle
import numpy as np
import pandas as pd

SIMS_DIR = "/gpfs/bbp.cscs.ch/project/proj83/scratch/home/ecker/simulations"
SPIKE_TH = -30  # -30 mV is NEURON's built in spike threshold

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load_sim_paths(project_name):
    """Loads in simulation paths as pandas MultiIndex DataFrame generated by bbp-workflow"""
    pklf_name = os.path.join(SIMS_DIR, project_name, "analyses", "simulations.pkl")
    if os.path.isfile(pklf_name):
        return pd.read_pickle(pklf_name)
    else:
        bc_path = os.path.join(SIMS_DIR, project_name, "BlueConfig")
        if os.path.isfile(bc_path):
            # warnings.warn("No bbp-workflow generated pandas MI DF found. Creating one with a single entry.")
            return pd.Series(data=bc_path)
        else:
            raise RuntimeError("Neither `analyses/simulations.pkl` nor `BlueConfig` found under %s" %
                               os.path.join(SIMS_DIR, project_name))


def _idx2str(idx, level_name):
    """Helper function to convert pandas.Index to string"""
    value = ("%.2f" % idx).replace('.', 'p') if isinstance(idx, float) else "%s" % idx
    return "%s%s" % (level_name, value)


def midx2str(midx, level_names):
    """Helper function to convert pandas.MultiIndex to string"""
    if len(level_names) == 1:  # it's not actually a MultiIndex
        if level_names[0] is not None:  # bbp-worflow generated
            return _idx2str(midx, level_names[0]) + "_"
        else:
            return ""
    elif len(level_names) > 1:
        str_ = ""
        for i, level_name in enumerate(level_names):
            str_ += _idx2str(midx[i], level_name) + "_"
        return str_
    else:
        raise RuntimeError("Incorrect level_names passed")


def get_spikes(sim, gids, t_start, t_end):
    """Extracts spikes with bluepy"""
    if gids is not None:
        spikes = sim.spikes.get(gids=gids, t_start=t_start, t_end=t_end)
    else:
        spikes = sim.spikes.get(t_start=t_start, t_end=t_end)
    return spikes.index.to_numpy(), spikes.to_numpy()


def calc_rate(spike_times, N, t_start, t_end, bin_size=10):
    """Calculates populational firing rate"""
    t_bins = np.arange(t_start, t_end+bin_size, bin_size)
    rate, _ = np.histogram(spike_times, t_bins)
    return rate / (N*1e-3*bin_size)  # *1e-3 ms to s conversion


def load_tc_gids(project_name):
    """Loads in VPM and POM gids from saved files"""
    vpm_gids, pom_gids = None, None
    proj_dir = os.path.join(project_name, "projections")
    if os.path.isdir(proj_dir):
        for f_name in os.listdir(proj_dir):
            if f_name[-4:] == ".txt" and "VPM" in f_name:
                vpm_gids = np.loadtxt(os.path.join(proj_dir, f_name))[:, 0].astype(int)
            if f_name[-4:] == ".txt" and "POM" in f_name:
                pom_gids = np.loadtxt(os.path.join(proj_dir, f_name))[:, 0].astype(int)
    return vpm_gids, pom_gids


def _get_spikef_name(config):
    """Gets the name of the SpikeFile from bluepy.Simulation.config object"""
    f_name = None
    stims = config.typed_sections("Stimulus")
    for stim in stims:
        if hasattr(stim, "SpikeFile"):
            f_name = stim.SpikeFile
            break  # atm. it handles only a single (the first in order) SpikeFile... TODO: extend this
    if f_name is not None:
        f_name = f_name if os.path.isabs(f_name) else os.path.join(config.Run["CurrentDir"], f_name)
    return f_name


def get_tc_spikes(sim_config, t_start, t_end):
    """Loads in input spikes (on projections) using the bluepy.Simulation.config object.
    Returns the format used for plotting rasters and population rates"""
    f_name = _get_spikef_name(sim_config)
    if f_name is not None:
        tmp = np.loadtxt(f_name, skiprows=1)
        spike_times, spiking_gids = tmp[:, 0], tmp[:, 1].astype(int)
        idx = np.where((t_start < spike_times) & (spike_times < t_end))[0]
        return spike_times[idx], spiking_gids[idx]
    else:
        warnings.warn("No SpikeFile found in the BlueConfig, returning empty arrays.")
        return np.array([]), np.array([], dtype=int)


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



