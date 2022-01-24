"""
Plastic SSCx related utility functions (most of them deal with the custom directory and file structure)
author: András Ecker, last update: 01.2022
"""

import os
import warnings
import pickle
from tqdm.contrib import tzip
import numpy as np
import pandas as pd
import numba
import multiprocessing as mp
from libsonata import ElementReportReader


SIMS_DIR = "/gpfs/bbp.cscs.ch/project/proj96/scratch/home/ecker/simulations"
MAPPING_SYNF_NAME = "/gpfs/bbp.cscs.ch/project/proj96/circuits/plastic_v1/syn_idx.pkl"
NONREP_SYNF_NAME = "/gpfs/bbp.cscs.ch/project/proj96/circuits/plastic_v1/nonrep_syn_df.pkl"
MORPH_FF_NAME = "/gpfs/bbp.cscs.ch/project/proj96/circuits/plastic_v1/morph_features.pkl"


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


def get_spikes(sim, t_start, t_end):
    """Extracts spikes with bluepy"""
    spikes = sim.spikes.get(t_start=t_start, t_end=t_end)
    return spikes.index.to_numpy(), spikes.values


def calc_rate(spike_times, N, t_start, t_end, bin_size=10):
    """Calculates populational firing rate"""
    t_bins = np.arange(t_start, t_end+bin_size, bin_size)
    rate, _ = np.histogram(spike_times, t_bins)
    return rate / (N*1e-3*bin_size)  # *1e-3 ms to s conversion


def get_tc_spikes(sim, t_start, t_end):
    """Extracts spikes from BC stimulus block 'spikeReplay'"""
    spikef_name = sim.config.Stimulus_spikeReplay.SpikeFile  # TODO: add some error handling...
    f_name = spikef_name if os.path.isabs(spikef_name) else os.path.join(sim.config.Run_Default.CurrentDir, spikef_name)
    tmp = np.loadtxt(f_name, skiprows=1)
    spike_times, spiking_gids = tmp[:, 0], tmp[:, 1].astype(int)
    idx = np.where((t_start < spike_times) & (spike_times < t_end))[0]
    return spike_times[idx], spiking_gids[idx]


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


def load_patterns(project_name, seed=None):
    """Loads in patterns from saved files (pretty custom structure and has a bunch of hard coded parts)"""
    pklf_name = None
    seed_str = "seed%i" % seed if seed is not None else "seed"
    for f_name in os.listdir(os.path.join(SIMS_DIR, project_name, "input_spikes")):
        if f_name[-4:] == ".pkl" and seed_str in f_name and "pattern_gids" in f_name:
            pklf_name = f_name
            with open(os.path.join(SIMS_DIR, project_name, "input_spikes", pklf_name), "rb") as f:
                pattern_gids = pickle.load(f)
    if pklf_name:
        metadata = pklf_name.split('__')[:-1]
        for f_name in os.listdir(os.path.join(SIMS_DIR, project_name, "projections")):
            if pklf_name.split("__nc")[0] + ".txt" == f_name:
                tmp = np.loadtxt(os.path.join(SIMS_DIR, project_name, "projections", f_name))
                gids, pos = tmp[:, 0].astype(int), tmp[:, 1:]
                return pattern_gids, gids, pos, metadata
            else:
                warnings.warn("Couldn't find saved positions in %s/projections" % project_name)
                return pattern_gids, None, None, metadata
    else:
        raise RuntimeError("Couldn't find saved *pattern_gids*.pkl in %s/input_spikes" % project_name)


def load_synapse_clusters(seed, sim_path):
    """Loads synapse clusters for given `seed` saved by `assemblyfire`"""
    base_dir = os.path.join(os.path.split(os.path.split(sim_path)[0])[0], "analyses", "seed%i_syn_clusters" % seed)
    syn_clusters, gids = {}, np.array([])
    for f_name in os.listdir(base_dir):
        if f_name[-4:] == ".pkl":
            df = pd.read_pickle(os.path.join(base_dir, f_name))
            syn_clusters[int(f_name.split("assembly")[1].split(".pkl")[0])] = df
            gids = np.concatenate((gids, df["post_gid"].unique()))
    return syn_clusters, gids


def load_synapse_report(h5f_name, t_start=None, t_end=None, t_step=None, gids=None, return_idx=False):
    """Fast, pure libsonata, in line implementation of report.get()"""
    report = ElementReportReader(h5f_name)
    report = report[list(report.get_population_names())[0]]
    time = np.arange(*report.times)
    if t_start == 0 and t_end == -1:  # special case of loading only the first and last time steps
        t_start = time[0]; t_end = time[-1]
        t_step = t_end - t_start
    if t_end == 0:  # special case of loading only the first time step
        t_start = t_end = time[0]
    if t_start == -1:  # special case of loading only the last time step
        t_start = t_end = time[-1]
    t_stride = round(t_step/report.times[2]) if t_step is not None else None
    report_gids = np.asarray(report.get_node_ids()) + 1
    node_idx = gids[np.isin(gids, report_gids, assume_unique=True)] - 1 if gids is not None else report_gids - 1
    if gids is not None and len(node_idx) < len(gids):
        warnings.warn("Not all gids are reported")
    view = report.get(node_ids=node_idx.tolist(), tstart=t_start, tstop=t_end, tstride=t_stride)
    if return_idx:
        col_idx = view.ids
        col_idx[:, 0] += 1  # get back gids from node_ids
        col_idx = pd.MultiIndex.from_arrays(col_idx.transpose(), names=["post_gid", "local_syn_idx"])
        return pd.DataFrame(data=view.data, index=pd.Index(view.times, name="time"),
                            columns=col_idx)
    else:
        return view.times, view.data


def reindex_report(data, mapping_df):
    """Re-indexes synapse report from (Neurodamus style) post_gid & local_syn_idx MultiIndex
    to (bluepy style) single global_syn_idx"""
    data.sort_index(axis=1, inplace=True)  # sort report columns to have the same ordering as the mapping df
    data.columns = mapping_df.index
    data = data.transpose(copy=False)
    return data


def load_mapping_df(features=None):
    """Loads bluepy style synapse DataFrame of pre_gids, post_gid & (Neurodamus style) local_syn_idx"""
    mapping_df = pd.read_pickle(MAPPING_SYNF_NAME)
    return mapping_df[features] if features is not None else mapping_df


def calc_mapping_df(sonata_fn, mi, features=None):
    """Uses conntility to get synapse mapping (bluepy style synapse DataFrame of pre_gids, post_gig
    & (Neurodamus style) local_syn_idx) for report's pd.MuliIndex (`mi`)"""
    from conntility.io.synapse_report import get_presyn_mapping
    mapping_df = get_presyn_mapping(sonata_fn, mi)
    return mapping_df[features] if features is not None else mapping_df


def load_nonrep_syn_df(report_name=None):
    """Loads bluepy style synapse DataFrame of non-reported (EXC) synapses (on L6 PCs in hex_O1)"""
    syn_df = pd.read_pickle(NONREP_SYNF_NAME)
    return syn_df[report_name] if report_name is not None else syn_df


def load_extra_morph_features(features=None):
    """Loads bluepy style synapse DataFrame of non-reported (EXC) synapses (on L6 PCs in hex_O1)"""
    df = pd.read_pickle(MORPH_FF_NAME)
    return df[features] if features is not None else df


def get_all_synapses_at_t(sim_path, report_name, t):
    """Loads reported and non-reported synapses (ordered by global_syn_id) at given time t"""
    # load time step
    h5f_name = os.path.join(os.path.split(sim_path)[0], "%s.h5" % report_name)
    if t == 0:
        data = load_synapse_report(h5f_name, t_end=0, return_idx=True)
    elif t == -1:
        data = load_synapse_report(h5f_name, t_start=-1, return_idx=True)
    else:
        data = load_synapse_report(h5f_name, t_start=t, t_end=t, return_idx=True)
    report_t = data.index.to_numpy()[0]
    # reindex it, transpose it and rename column index
    mapping_df = load_mapping_df()  # load mapping df, which has (and is ordered by) the global_syn_idx
    data = reindex_report(data, mapping_df)
    data.columns = pd.Index([report_name])
    # load non-reported values, convert them to float DF and merge the 2 datasets
    nonrep_data = load_nonrep_syn_df(report_name)
    data = data.append(nonrep_data.to_frame().astype(np.float64))
    data.sort_index(inplace=True)
    return report_t, data


def get_synapse_changes(sim_path, report_name, gids):
    """Gets total change (last reported t - first reported t) of synapses on `gids`"""
    from bluepy import Circuit
    c = Circuit(sim_path)
    # load first and last time steps
    h5f_name = os.path.join(os.path.split(sim_path)[0], "%s.h5" % report_name)
    data = load_synapse_report(h5f_name, t_start=0, t_end=-1, gids=gids, return_idx=True)
    # reindex, transpose and make diff
    mapping_df = calc_mapping_df(c.config["connectome"], data.columns)
    data = reindex_report(data, mapping_df)
    return pd.Series(data=np.diff(data.to_numpy(), axis=1).reshape(-1),
                     index=data.index, name="delta_%s" % report_name)


def split_synapse_report(c, data, split_by):
    """Splits `data` (synapse report in DataFrame) into chunks, organized by `split_by` property of (post) gids"""
    gids = data.columns.get_level_values(0).unique().to_numpy()
    categories = c.cells.get(gids, split_by)
    split_data = {}
    unique_cats = np.sort(categories.unique())
    for cat in unique_cats:
        split_data[cat] = data[categories[categories == cat].index].to_numpy()
    return split_data


def update_split_data(c, report_name, split_data, split_by):
    """Updates split data with constant, non-reported (but saved elsewhere for hex_O1) values"""
    nonrep_df = load_nonrep_syn_df(["post_gid", report_name])
    gids = nonrep_df["post_gid"].unique()
    categories = c.cells.get(gids, split_by)
    unique_cats = np.sort(categories.unique())
    n = split_data[unique_cats[0]].shape[0]
    for cat in unique_cats:
        nonrep_data = nonrep_df.loc[nonrep_df["post_gid"].isin(categories[categories == cat].index),
                                    report_name].to_numpy()
        split_data[cat] = np.concatenate((split_data[cat], np.tile(nonrep_data, (n, 1))), axis=1)
    return split_data


@numba.njit
def numba_hist(values, bins, bin_range):
    """Dummy function for numba decorator..."""
    return np.histogram(values, bins=bins, range=bin_range)


def get_synapse_report_hist(h5f_name, n_chunks=5, bins=10, bin_range=(0, 1)):
    """Similar to `load_synapse_report()` above, but is designed to load the full report
    and return histograms at each time points (which are easier to handle)"""
    report = ElementReportReader(h5f_name)
    report = report[list(report.get_population_names())[0]]
    time = np.arange(*report.times)
    n_bins = bins if isinstance(bins, int) else len(bins) - 1
    binned_report = np.zeros((n_bins, len(time)), dtype=int)
    node_idx = np.asarray(report.get_node_ids())
    idx = np.linspace(0, len(node_idx), n_chunks+1, dtype=int)
    for start_id, end_id in tzip(idx[:-1], idx[1:], desc="Loading chunks of data"):
        view = report.get(node_ids=node_idx[start_id:end_id].tolist())
        data = view.data
        for i in range(len(time)):
            hist, bin_edges = numba_hist(data[i, :], bins, bin_range)
            binned_report[:, i] += hist
        del data, view
    return bin_edges, time, binned_report


def update_hist_data(report_name, data, bin_edges):
    """Updates binned report with constant, non-reported (but saved elsewhere for hex_O1) values"""
    nonrep_data = load_nonrep_syn_df(report_name).to_numpy()
    hist, _ = np.histogram(nonrep_data, bins=bin_edges)
    for i in range(data.shape[1]):
        data[:, i] += hist
    return data


def coarse_binning(bin_edges, data, new_nbins):
    """Re-bins data from synapse report on a lower resolution (more coarse binning)"""
    orig_nbins = data.shape[0]
    q, m = np.divmod(orig_nbins, new_nbins)
    assert m == 0, "Cannot divide original %i bins into %i new ones" % (orig_nbins, new_nbins)
    new_data = np.zeros((int(orig_nbins / q), data.shape[1]), dtype=int)
    idx = np.arange(0, orig_nbins + q, q)
    for i, (start_id, end_id) in enumerate(zip(idx[:-1], idx[1:])):
        new_data[i, :] = np.sum(data[start_id:end_id, :], axis=0)
    return bin_edges[::q], new_data




