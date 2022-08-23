"""
Analyzes V_m distribution and spectrum, based on A.Dexteshe et al. (2001,2003)
In vivo-like should be normally distributed with a high (~ -60mV) mean, and its spectrum as colored (pink) noise
author: András Ecker, last update: 08.2022
"""

import os
from tqdm import tqdm
import numpy as np
from scipy.stats import normaltest
from scipy.signal import welch
import pandas as pd
from bluepy import Simulation
from utils import parse_stim_blocks, stim2str
from plots import plot_vm_dist_spect, plot_heatmap_line, plot_heatmap_grid, plot_corrs

SPIKE_TH = -30  # mV (NEURON's built in spike threshold)
SIGN_TH = 0.05  # alpha level for significance tests
FIGS_DIR = "/gpfs/bbp.cscs.ch/project/proj83/home/ecker/figures/vm_analysis"
BASE_SIMS_DIR = "/gpfs/bbp.cscs.ch/project/proj83/home/bolanos/Bernstein2022/singlecell/"


def analyze_v_dist(v):
    """Analyzes V_m distribution"""
    v = v[v < SIGN_TH]  # get rid of spikes
    # _, p = normaltest(v)  # there are a bunch of other tests for normality...
    # normal = True if p > SIGN_TH else False
    return np.mean(v), np.std(v) #, normal


def analyze_v_spectrum(v, fs, freq_window):
    """Analyzes the spectrum of V_m TODO: get rid of spikes (properly)"""
    f, pxx = welch(v, fs=fs)
    # cut low freq. part before fitting a line to log-log data
    idx = np.where((freq_window[0] < f) & (f < freq_window[1]))[0]
    coeffs = np.polyfit(np.log10(f[idx]), np.log10(pxx[idx]), deg=1)
    return f, pxx, coeffs


def pool_results(df, input_cols=["pattern", "mode", "mean", "std", "tau", "amp_cv"],
                 feature_cols=["V_mean", "V_std", "rate"], mi=False):
    """Pools results (e.g. from different seeds or gids) and report their mean"""
    agg_df = df.groupby(input_cols)[feature_cols].agg("mean")
    return agg_df if mi else agg_df.reset_index()
    

def main(sim, t_start_offset=200, freq_window=[10, 5000], plot_results=False):
    # load report with bluepy
    report = sim.report("soma")
    fs = 1 / (report.meta["time_step"] / 1000)
    assert len(report.gids) == 1, "Works with single reported gid atm." \
                                  "(Either add gid selection to the code or report only from a single gid)"
    tmp = report.get()
    t, v = tmp.index.to_numpy().reshape(-1), tmp.to_numpy().reshape(-1)
    spike_times = sim.spikes.get().index.to_numpy()
    # parse stim blocks and iterate over them
    stims = parse_stim_blocks(sim.config)
    results = np.zeros((len(stims), 5), dtype=np.float32)
    for i, (row_id, stim) in enumerate(stims.iterrows()):
        t_start, t_end = stim["t_start"] + t_start_offset, stim["t_end"]
        rate = len(spike_times[(t_start < spike_times) & (spike_times <= t_end)]) / ((t_end - t_start) / 1000)
        v_window = v[(t_start < t) & (t <= t_end)]
        mean, std = analyze_v_dist(v_window)
        if freq_window is not None:
            f, pxx, coeffs = analyze_v_spectrum(v_window, fs, freq_window)
            if plot_results:
                fig_name = os.path.join(FIGS_DIR, "individual", "%s.png" % stim2str(stim))
                plot_vm_dist_spect(v_window, mean, std, spiking, f, pxx, coeffs, freq_window, fig_name)
            results[i, :] = [row_id, mean, std, rate, coeffs[0]]
        else:
            results[i, :-1] = [row_id, mean, std, rate]
    if freq_window is not None:
        results[results[:, 3] == 0., 4] = np.nan  # only keep fits to subth. traces
        results = pd.DataFrame(data=results[:, 1:], index=results[:, 0].astype(int),
                               columns=["V_mean", "V_std", "rate", "PSD_slope"])
    else:
        results = pd.DataFrame(data=results[:, 1:-1], index=results[:, 0].astype(int),
                               columns=["V_mean", "V_std", "rate"])
    return pd.concat([stims, results], axis=1)


if __name__ == "__main__":
    results = []
    for std in ["sdperc3", "sdperc6", "sdperc9", "sdperc12", "sdperc15", "sdperc18"]:
        sim = Simulation(os.path.join(BASE_SIMS_DIR, "mtype_sample", "seed174345", "unique_emorphos_0.1_8196",
                                      "Conductance", "RelativeOrnsteinUhlenbeck_E", "tau3", std, "BlueConfig"))
        results.append(main(sim, freq_window=None))
        sim = Simulation(os.path.join(BASE_SIMS_DIR, "mtype_sample", "seed174345", "unique_emorphos_0.1_8196",
                                      "Conductance", "RelativeShotNoise_E", "tau0.4_4", "ampcv0.5", std, "BlueConfig"))

    for std in ["sdperc5", "sdperc10", "sdperc15", "sdperc20", "sdperc25", "sdperc30"]:
        sim = Simulation(os.path.join(BASE_SIMS_DIR, "mtype_sample", "seed174345", "unique_emorphos_0.1_8196",
                                      "Current", "RelativeOrnsteinUhlenbeck_E", "tau3", std, "BlueConfig"))
        results.append(main(sim, freq_window=None))
        sim = Simulation(os.path.join(BASE_SIMS_DIR, "mtype_sample", "seed174345", "unique_emorphos_0.1_8196",
                                      "Current", "RelativeShotNoise_E", "tau0.4_4", "ampcv0.5", std, "BlueConfig"))
    df = pd.concat(results, axis=0, ignore_index=True)
    df = pool_results(df.drop(columns=["t_start", "t_end"], inplace=True))

    for pattern in ["RelativeShotNoise", "RelativeOrnsteinUhlenbeck"]:
        for mode in ["Current", "Conductance"]:
            df_plot = df.loc[(df["pattern"] == pattern) & (df["mode"] == mode)]
            if "ShotNoise" in pattern:
                plot_heatmap_grid(df_plot, "V_mean", os.path.join(FIGS_DIR, "%s_%s_V_mean.png" % (pattern, mode)))
                plot_heatmap_grid(df_plot, "V_std", os.path.join(FIGS_DIR, "%s_%s_V_std.png" % (pattern, mode)))
            else:
                plot_heatmap_line(df_plot, "V_mean", os.path.join(FIGS_DIR, "%s_%s_V_mean.png" % (pattern, mode)))
                plot_heatmap_line(df_plot, "V_std", os.path.join(FIGS_DIR, "%s_%s_V_std.png" % (pattern, mode)))
        plot_corrs(df.loc[df["pattern"] == pattern], ["mean", "std"], ["V_mean", "V_std"], mode, None,
                   os.path.join(FIGS_DIR, "%s_corrs" % pattern))




