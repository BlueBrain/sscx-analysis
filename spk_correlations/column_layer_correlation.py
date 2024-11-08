"""
Within column/hexagon E-I and across-column/hexagon cross correlations split by layer
author: Daniela Egas Santander, last update: 03.2023
"""

import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate, correlation_lags
from bluepy import Simulation
import itertools
import pickle

# 240 columns (last colummn is empy and column cells in ``column`` 0 are cells that couldn't be flatmapped)
NRRD_FNAME = "/gpfs/bbp.cscs.ch/project/proj83/home/reimann/subvolumes/column_identities.nrrd"  


def get_nrndf_nrrd(c):
    """Gets neuron DataFrame from hexes defined in an nrrd file (using `conntility`)"""
    from conntility.circuit_models import neuron_groups
    load_cfg = {"loading": {"base_target": "Mosaic",
                            "properties": ["x", "y", "z", "synapse_class", "layer"],
                            "atlas": [{"data": NRRD_FNAME, "properties": ["column_id"]}]},
                "grouping": [{"method": "group_by_properties", "columns": ["column_id"]}]}
    nrn_df = neuron_groups.load_group_filter(c, load_cfg)
    return nrn_df[["gid", "column_id", "synapse_class", "layer"]].reset_index(drop=True)


def _get_spikes(sim, t_start, t_end, gids=None):
    """Extracts spikes with `bluepy`"""
    spikes = sim.spikes.get(gids, t_start, t_end)
    return spikes.index.to_numpy(), spikes.to_numpy()


def bin_column_spikes(sim, hexes_from, bin_size, t_start=1000,layers="all"):
    """Bins EXC and INH spikes per layer in all columns"""
    assert hexes_from in ["nrrd"], "Hex targets can be only read from nrrd file"
    nrn_df = get_nrndf_nrrd(sim.circuit)
    if layers == "all":
        layers=nrn_df["layer"].unique()
    spike_times, spiking_gids = _get_spikes(sim, t_start, sim.t_end)
    bins = np.arange(t_start, sim.t_end + bin_size, bin_size)
    spike_counts = {f'layer_{i}':{"EXC": {}, "INH": {}} for i in layers}
    for column_id in nrn_df["column_id"].unique():
        for i, type_ in itertools.product(layers,["EXC", "INH"]):

            gids = nrn_df.loc[((nrn_df["column_id"] == column_id) & 
                               (nrn_df["synapse_class"] == type_)&
                               (nrn_df["layer"] == i)), "gid"].to_numpy()
            spike_counts_, _ = np.histogram(spike_times[np.in1d(spiking_gids, gids)], bins)
            spike_counts[f'layer_{i}'][type_][column_id] = spike_counts_
    return spike_counts, layers


def correlate_spike_counts(spike_counts_1, spike_counts_2, kernel_std=1):
    """Correlation between Gaussian smoothed spike counts"""
    spike_counts_1 = gaussian_filter1d(spike_counts_1.astype(float), kernel_std)
    spike_counts_2 = gaussian_filter1d(spike_counts_2.astype(float), kernel_std)
    return np.corrcoef(spike_counts_1, spike_counts_2)[0, 1]


def cross_correlate_spike_counts(spike_counts_1, spike_counts_2, bin_size, kernel_std=1):
    """Max and (lag) of cross-correlation between Gaussian smoothed and mean centered spike counts
    (Normalized to be Pearson correlation not the `scipy.signal` dot product...)"""
    spike_counts_1 = gaussian_filter1d(spike_counts_1.astype(float), kernel_std)
    spike_counts_2 = gaussian_filter1d(spike_counts_2.astype(float), kernel_std)
    if np.std(spike_counts_1) != 0 and np.std(spike_counts_2) != 0:
        norm_spike_counts_1 = spike_counts_1 - np.mean(spike_counts_1)
        norm_spike_counts_2 = spike_counts_2 - np.mean(spike_counts_2)
        corrs = correlate(norm_spike_counts_1, norm_spike_counts_2)
        corrs /= (np.std(spike_counts_1) * np.std(spike_counts_2) * len(spike_counts_2))
        lags = correlation_lags(len(spike_counts_1), len(spike_counts_2)) * bin_size
        return np.max(corrs), lags[np.argmax(corrs)]
    else:
        return np.nan, np.nan


def column_ei_correlations(spike_counts, layers):
    """E-I correlation of within column spikes for the layers in layers"""
    ei_corrs={f'layer_{i}':None for i in layers}
    for i in layers:
        n_cols = len(spike_counts[f'layer_{i}']["EXC"].keys())
        ei_corrs[f'layer_{i}']=np.array([correlate_spike_counts(spike_counts[f'layer_{i}']["EXC"][j], spike_counts[f'layer_{i}']["INH"][j]) for j in range(n_cols)])
    return ei_corrs

def column_cross_correlations(spike_counts, bin_size, layers):
    """Cross correlation of spikes across (pairs of) columns for the layers in layers"""
    cross_corrs={f'layer_{i}':{} for i in layers}
    for i in layers:
        n_cols = len(spike_counts[f'layer_{i}']["EXC"].keys())
        corrs = np.zeros((n_cols, n_cols))
        lags = np.zeros_like(corrs)
        for r, t in itertools.product(range(n_cols),range(n_cols)):
            corr, lag = cross_correlate_spike_counts(spike_counts[f'layer_{i}']["EXC"][r] + spike_counts[f'layer_{i}']["INH"][r],
                                                     spike_counts[f'layer_{i}']["EXC"][t] + spike_counts[f'layer_{i}']["INH"][t], 
                                                     bin_size)
            corrs[r, t], lags[r, t] = corr, lag
        cross_corrs[f'layer_{i}']['corrs']=corrs
        cross_corrs[f'layer_{i}']['lags']=lags
    return cross_corrs


def main(sim_root, hexes_from, bin_size=3,layers="all"):
    sim = Simulation(os.path.join(sim_root, "BlueConfig"))
    spike_counts, layers = bin_column_spikes(sim, hexes_from, bin_size,layers=layers)
    ei_corrs = column_ei_correlations(spike_counts,layers)
    cross_corrs= column_cross_correlations(spike_counts, bin_size,layers)
    #Saving
    path_out=os.path.join(sim_root, "ei_corrs_per_layer.pkl")
    with open(path_out, 'wb') as fp:
        pickle.dump(ei_corrs, fp)
    path_out=os.path.join(sim_root, "cross_corrs_per_layer.pkl")
    with open(path_out, 'wb') as fp:
        pickle.dump(cross_corrs, fp)
    return ei_corrs, cross_corrs


if __name__ == "__main__":
    hexes_from = "nrrd"
    #sim_root = "/gpfs/bbp.cscs.ch/project/proj83/scratch/bbp_workflow/sscx_calibration_mgfix/5-FullCircuit/" \
    ##           "5-FullCircuit-2-BetterMinis-FprScan/5d83d4c2-693c-4ecc-a9da-c8dd2c8100c3/4/"
    sim_root = "/gpfs/bbp.cscs.ch/project/proj83/scratch/bbp_workflow/sscx_calibration_mgfix/5-FullCircuit/" \
                "5-FullCircuit-2-BetterMinis-Fpr15-StimScan-10x/bb16bd9f-3d21-4a35-8296-d6aec4c55bf7/2/"
    main(sim_root, hexes_from)
