"""
Analyzes V_m distribution and spectrum, based on A.Dexteshe et al. (2001,2003)
In vivo-like should be normally distributed with a high (~ -60mV) mean, and its spectrum as colored (pink) noise
author: András Ecker, last update: 08.2022
"""

import os
from tqdm import tqdm
from copy import deepcopy
import numpy as np
from scipy.stats import normaltest
from scipy.signal import welch
import pandas as pd
from bluepy import Simulation
from utils import parse_stim_blocks, stim2str
from plots import plot_vm_dist_spect, plot_heatmap_grid, plot_corrs

SPIKE_TH = -30  # mV (NEURON's built in spike threshold)
SIGN_TH = 0.05  # alpha level for significance tests
FIGS_DIR = "/gpfs/bbp.cscs.ch/project/proj83/home/ecker/figures/vm_analysis"
BASE_SIMS_DIR = "/gpfs/bbp.cscs.ch/project/proj83/home/bolanos/Bernstein2022/singlecell/"
EXC_MTYPES = {"L23E": ["L2_IPC", "L2_TPC:A", "L2_TPC:B", "L3_TPC:A", "L3_TPC:C"],
              "L4E": ["L4_SSC", "L4_TPC", "L4_UPC"], "L5E": ["L5_TPC:A", "L5_TPC:B", "L5_TPC:C", "L5_UPC"],
              "L6E": ["L6_BPC", "L6_HPC", "L6_IPC", "L6_TPC:A", "L6_TPC:C", "L6_UPC"]}


def sample_metypes(c, gids, nsamples):
    """Samples `nsamples` (or as many as possible) gids for all me_types"""
    df = c.cells.get(gids, ["mtype", "etype"]).reset_index()
    # replace=True and then .unique() is just a workaround to sample as many as there is from the given group...
    sample_gids = df.groupby(["mtype", "etype"]).sample(n=nsamples, replace=True)["index"].unique()
    return c.cells.get(sample_gids, ["mtype", "etype"]).sort_index()


def get_spikes(sim, gids):
    """Loads spikes"""
    spikes = sim.spikes.get(gids=gids)
    return spikes.index.to_numpy(), spikes.to_numpy()


def calc_rate(df, spike_times, spiking_gids, t_start, t_end):
    """Calculates single cell firing rate (and adds it to `df`)"""
    df["rate"] = 0.
    norm = (t_end - t_start) / 1000
    v, c = np.unique(spiking_gids[(t_start < spike_times) & (spike_times <= t_end)], return_counts=True)
    df.loc[v, "rate"] =  c / norm
    return df


def analyze_v_dist(df, v):
    """Analyzes V_m distribution (no stat. test for normality any more) and adds results to `df`"""
    v[v > SIGN_TH] = np.nan  # get rid of spikes
    df["V_mean"] = np.nanmean(v, axis=0)
    df["V_std"] = np.nanstd(v, axis=0)
    return df


def analyze_v_spectrum(v, fs, freq_window):
    """Analyzes the spectrum of V_m TODO: get rid of spikes (properly)"""
    f, pxx = welch(v, fs=fs)
    # cut low freq. part before fitting a line to log-log data
    idx = np.where((freq_window[0] < f) & (f < freq_window[1]))[0]
    coeffs = np.polyfit(np.log10(f[idx]), np.log10(pxx[idx]), deg=1)
    return f, pxx, coeffs


def pool_results(df, input_cols=["pattern", "mode", "mean", "std", "tau", "mtype"],
                 feature_cols=["V_mean", "V_std", "rate"]):
    """Pools results (e.g. from different seeds or gids) and report their mean"""
    agg_df = df.groupby(input_cols, as_index=False)[feature_cols].agg("mean")
    return agg_df.loc[~agg_df.isnull().any(axis=1)]


def pool_mtypes(unique_mtypes, mtypes_dict):
    """Pools `unique_mtypes` based on the `mtypes_dict` passed"""
    data = np.repeat("other", len(unique_mtypes))
    for mtype_group, mtypes in mtypes_dict.items():
        for mtype in mtypes:
            data[unique_mtypes == mtype] = mtype_group
    return pd.Series(data=data, index=unique_mtypes)


def main(sim, nsamples=10, t_start_offset=200, freq_window=[10, 500], plot_results=False):
    # load reports with bluepy
    report = sim.report("soma")
    fs = 1 / (report.meta["time_step"] / 1000)
    gids = sample_metypes(sim.circuit, report.gids, nsamples)
    tmp = report.get(gids=gids.index.to_numpy())
    t, v = tmp.index.to_numpy().reshape(-1), tmp.to_numpy()
    spike_times, spiking_gids = get_spikes(sim, gids.index.to_numpy())
    # parse stim blocks and iterate over them
    stims = parse_stim_blocks(sim.config)
    results = []
    for _, stim in stims.iterrows():
        t_start, t_end = stim["t_start"] + t_start_offset, stim["t_end"]
        df = calc_rate(deepcopy(gids), spike_times, spiking_gids, t_start, t_end)
        v_window = v[(t_start < t) & (t <= t_end), :]
        df = analyze_v_dist(df, deepcopy(v_window))
        if freq_window is not None:
            psd_slopes = np.zeros(len(gids), dtype=np.float32)
            for i in range(v_window.shape[1]):  # try to speed this up and don't do it one-by-one
                f, pxx, coeffs = analyze_v_spectrum(v_window[:, i], fs, freq_window)
                psd_slopes[i] = coeffs[1]
                if plot_results:
                    fig_name = os.path.join(FIGS_DIR, "individual", "%s_a%i.png" % (stim2str(stim), gids.index[i]))
                    plot_vm_dist_spect(v_window[:, i], df.loc[gids.index[i], "V_mean"], df.loc[gids.index[i], "V_std"],
                                       df.loc[gids.index[i], "rate"], f, pxx, coeffs, freq_window, fig_name)
            df["PSD_slope"] = np.abs(psd_slopes)
            # df.loc[df["rate"] > 0., "PSD_slope"] = -1  # only keep fits to subthreshold traces
        results.append(df)
    results = pd.concat(results, ignore_index=True, axis=0)
    return pd.concat([stims.loc[stims.index.repeat(len(gids))].reset_index(drop=True), results], axis=1)


if __name__ == "__main__":
    results = []
    for seed in tqdm(["seed174345", "seed253057", "seed312643", "seed497313", "seed639057",
                      "seed654929", "seed777747", "seed852251", "seed966958", "seed983777"]):
        for std in ["sdperc3", "sdperc6", "sdperc9", "sdperc12", "sdperc15", "sdperc18"]:
            bc = os.path.join(BASE_SIMS_DIR, "mtype_sample", "seed174345", "unique_emorphos_0.1_8196", "Conductance",
                              "RelativeOrnsteinUhlenbeck_E", "meanperc3_3", "tau3", std, "BlueConfig")
            results.append(main(Simulation(bc)))
            bc = os.path.join(BASE_SIMS_DIR, "mtype_sample", "seed174345", "unique_emorphos_0.1_8196", "Conductance",
                              "RelativeShotNoise_E", "meanperc3_3", "tau0.4_4", "ampcv0.5", std, "BlueConfig")
            results.append(main(Simulation(bc)))
        # TODO: merge these 2 below once the paths get fixed...
        for std in ["sdperc5", "sdperc10", "sdperc15", "sdperc20", "sdperc25", "sdperc30"]:
            bc = os.path.join(BASE_SIMS_DIR, "mtype_sample", "seed174345", "unique_emorphos_0.1_8196", "Current",
                              "RelativeOrnsteinUhlenbeck_E", "meanperc10_15", "tau3", std, "BlueConfig")
            results.append(main(Simulation(bc)))
            bc = os.path.join(BASE_SIMS_DIR, "mtype_sample", "seed174345", "unique_emorphos_0.1_8196", "Current",
                              "RelativeShotNoise_E", "meanperc10_15", "tau0.4_4", "ampcv0.5", std, "BlueConfig")
            results.append(main(Simulation(bc)))
        for std in ["sdperc35", "sdperc40"]:
            bc = os.path.join(BASE_SIMS_DIR, "mtype_sample", "seed174345", "unique_emorphos_0.1_8196", "Current",
                              "RelativeOrnsteinUhlenbeck_E", "meanperc10_15", "tau3", std, "BlueConfig")
            results.append(main(Simulation(bc)))
            bc = os.path.join(BASE_SIMS_DIR, "mtype_sample", "seed174345", "unique_emorphos_0.1_8196", "Current",
                              "RelativeShotNoise_E", "meanperc10_15", "tau0.4_4", "fskew0.5", std, "BlueConfig")
            results.append(main(Simulation(bc)))
    df = pd.concat(results, axis=0, ignore_index=True).drop(columns=["t_start", "t_end"])
    df.to_pickle("vm.pkl")

    agg_df = pool_results(df)
    drop_df = agg_df.loc[(agg_df["mode"] == "Current") & (agg_df["std"].isin([5, 10]))]
    agg_df.drop(index=drop_df.index, inplace=True)  # get rid of these for consistent plots...
    mean_min, mean_max = -85, -65  # agg_df["V_mean"].quantile(0.01), agg_df["V_mean"].quantile(0.99)
    std_min, std_max = 0, 9  # agg_df["V_std"].quantile(0.01), agg_df["V_std"].quantile(0.99)
    mtypes = agg_df["mtype"].unique().to_numpy()
    agg_df["mtype_group"] = pool_mtypes(mtypes, EXC_MTYPES).loc[agg_df["mtype"]].to_numpy()
    for mtype_group in list(EXC_MTYPES.keys()) + ["other"]:
        df_plot = agg_df.loc[(agg_df["mtype_group"] == mtype_group)]
        plot_heatmap_grid(df_plot, "V_mean", "mode", "pattern", "rate", mean_min, mean_max, "viridis",
                          os.path.join(FIGS_DIR, "%s_V_mean.png" % mtype_group))
        plot_heatmap_grid(df_plot, "V_std", "mode", "pattern", "rate", std_min, std_max, "magma",
                          os.path.join(FIGS_DIR, "%s_V_std.png" % mtype_group))
    '''
    for mtype in mtypes:
        df_plot = agg_df.loc[(agg_df["mtype"] == mtype)]
        plot_heatmap_grid(df_plot, "V_mean", "mode", "pattern", "rate", mean_min, mean_max, "viridis",
                          os.path.join(FIGS_DIR, "%s_V_mean.png" % mtype))
        plot_heatmap_grid(df_plot, "V_std", "mode", "pattern", "rate", std_min, std_max, "magma",
                          os.path.join(FIGS_DIR, "%s_V_std.png" % mtype))
    '''





