"""
Within column/hexagon E-I and across-column/hexagon cross correlations
author: András Ecker, last update: 03.2023
"""

import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
from scipy.signal import correlate, correlation_lags
from bluepy import Simulation

N_HEX_TARGETS = 78  # 78 hexes defined in Sirio's user.target
NRRD_FNAME = "/gpfs/bbp.cscs.ch/project/proj83/home/reimann/subvolumes/column_identities.nrrd"  # 240 columns


def get_nrndf_nrrd(c):
    """Gets neuron DataFrame from hexes defined in an nrrd file (using `conntility`)"""
    from conntility.circuit_models import neuron_groups
    load_cfg = {"loading": {"base_target": "Mosaic",
                            "properties": ["x", "y", "z", "synapse_class"],
                            "atlas": [{"data": NRRD_FNAME, "properties": ["column_id"]}]},
                "grouping": [{"method": "group_by_properties", "columns": ["column_id"]}]}
    nrn_df = neuron_groups.load_group_filter(c, load_cfg)
    return nrn_df[["gid", "column_id", "synapse_class"]].reset_index(drop=True)


def get_nrndf_usertarget(c):
    """Gets neuron DataFrame from hexes defined in user.target"""
    hex_gids = [c.cells.ids("hex%i" % i) for i in range(N_HEX_TARGETS)]
    hex_idx = np.concatenate([i * np.ones_like(gids) for i, gids in enumerate(hex_gids)])
    data = np.vstack([np.concatenate(hex_gids), hex_idx]).T
    nrn_df = pd.DataFrame(data=data, columns=["gid", "column_id"])
    nrn_df["synapse_class"] = c.cells.get(data[:, 0], "synapse_class").to_numpy()
    return nrn_df


def _get_spikes(sim, t_start, t_end, gids=None):
    """Extracts spikes with `bluepy`"""
    spikes = sim.spikes.get(gids, t_start, t_end)
    return spikes.index.to_numpy(), spikes.to_numpy()


def bin_column_spikes(sim, hexes_from, bin_size, t_start=1000):
    """Bins EXC and INH spikes in all hexes/columns"""
    assert hexes_from in ["nrrd", "user_target"], "Hex targets can be only read from nrrd file or user.target"
    nrn_df = get_nrndf_nrrd(sim.circuit) if hexes_from == "nrrd" else get_nrndf_usertarget(sim.circuit)

    spike_times, spiking_gids = _get_spikes(sim, t_start, sim.t_end)
    bins = np.arange(t_start, sim.t_end + bin_size, bin_size)
    spike_counts = {"EXC": {}, "INH": {}}
    for column_id in nrn_df["column_id"].unique():
        for type_ in ["EXC", "INH"]:
            gids = nrn_df.loc[(nrn_df["column_id"] == column_id) & (nrn_df["synapse_class"] == type_), "gid"].to_numpy()
            spike_counts_, _ = np.histogram(spike_times[np.in1d(spiking_gids, gids)], bins)
            spike_counts[type_][column_id] = spike_counts_
    return spike_counts


def correlate_spike_counts(spike_counts_1, spike_counts_2, kernel_std=1):
    """R-value of linear regression between Gaussian smoothed and mean centered spike counts"""
    spike_counts_1 = gaussian_filter1d(spike_counts_1, kernel_std)
    spike_counts_2 = gaussian_filter1d(spike_counts_2, kernel_std)
    if np.max(spike_counts_1) != 0 and np.max(spike_counts_2) != 0:
        norm_spike_counts_1 = spike_counts_1 - np.mean(spike_counts_1)
        norm_spike_counts_2 = spike_counts_2 - np.mean(spike_counts_2)
        return linregress(norm_spike_counts_1, norm_spike_counts_2).rvalue
    else:
        return np.nan


def cross_correlate_spike_counts(spike_counts_1, spike_counts_2, bin_size, kernel_std=1):
    """Max and (lag) of cross-correlation between Gaussian smoothed and mean centered spike counts
    (Normalized to be Pearson correlation not the `scipy.signal` dot product...)"""
    spike_counts_1 = gaussian_filter1d(spike_counts_1, kernel_std)
    spike_counts_2 = gaussian_filter1d(spike_counts_2, kernel_std)
    if np.std(spike_counts_1) != 0 and np.std(spike_counts_2) != 0:
        norm_spike_counts_1 = spike_counts_1 - np.mean(spike_counts_1)
        norm_spike_counts_2 = spike_counts_2 - np.mean(spike_counts_2)
        corrs = correlate(norm_spike_counts_1, norm_spike_counts_2)
        corrs /= (np.std(spike_counts_1) * np.std(spike_counts_2) * len(spike_counts_2))
        lags = correlation_lags(len(spike_counts_1), len(spike_counts_2)) * bin_size
        return np.max(corrs), lags[np.argmax(corrs)]
    else:
        return np.nan, np.nan


def column_ei_correlations(spike_counts):
    """E-I correlation of within column spikes"""
    n_cols = len(spike_counts["EXC"].keys())
    return np.array([correlate_spike_counts(spike_counts["EXC"][i], spike_counts["INH"][i]) for i in range(n_cols)])


def column_cross_correlations(spike_counts, bin_size):
    """Cross correlation of spikes across (pairs of) columns"""
    n_cols = len(spike_counts["EXC"].keys())
    corrs = np.zeros((n_cols, n_cols))
    lags = np.zeros_like(corrs)
    for i in range(n_cols):
        for j in range(n_cols):
            corr, lag = cross_correlate_spike_counts(spike_counts["EXC"][i] + spike_counts["INH"][i],
                                                     spike_counts["EXC"][j] + spike_counts["INH"][j], bin_size)
            corrs[i, j], lags[i, j] = corr, lag
    return corrs, lags


def main(sim_root, hexes_from, bin_size=3):
    sim = Simulation(os.path.join(sim_root, "BlueConfig"))
    spike_counts = bin_column_spikes(sim, hexes_from, bin_size)
    ei_corrs = column_ei_correlations(spike_counts)
    corrs, lags = column_cross_correlations(spike_counts, bin_size)
    npzf_name = os.path.join(sim_root, "corrs_%ihexes.npz" % len(spike_counts["EXC"].keys()))
    np.savez(npzf_name, ei_corrs=ei_corrs, corrs=corrs, lags=lags)


if __name__ == "__main__":
    hexes_from = "nrrd"
    sim_root = "/gpfs/bbp.cscs.ch/project/proj83/scratch/bbp_workflow/sscx_calibration_mgfix/5-FullCircuit/" \
               "5-FullCircuit-2-BetterMinis-FprScan/5d83d4c2-693c-4ecc-a9da-c8dd2c8100c3/4/"
    # sim_root = "/gpfs/bbp.cscs.ch/project/proj83/scratch/bbp_workflow/sscx_calibration_mgfix/5-FullCircuit/" \
    ##           "5-FullCircuit-2-BetterMinis-Fpr15-StimScan-10x/bb16bd9f-3d21-4a35-8296-d6aec4c55bf7/2/"
    main(sim_root, hexes_from)