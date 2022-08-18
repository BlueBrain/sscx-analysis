"""
Analyzes V_m distribution and spectrum, based on A.Dexteshe et al. (2001,2003)
In vivo-like should be normally distributed with a high (~ -60mV) mean, and its spectrum as colored (pink) noise
author: András Ecker, last update: 08.2022
"""

import os
import numpy as np
from scipy.stats import normaltest
from scipy.signal import welch
import pandas as pd
from bluepy import Simulation
from utils import parse_stim_blocks, stim2str
from plots import plot_vm_dist_spect, plot_heatmap_line, plot_heatmap_grid, plot_corrs

SPIKE_TH = -30  # mV (NEURON's built in spike threshold)
SIGN_TH = 0.1  # alpha level for significance tests
FIGS_DIR = "/gpfs/bbp.cscs.ch/project/proj83/home/ecker/figures/vm_analysis"
BASE_SIMS_DIR = "/gpfs/bbp.cscs.ch/project/proj83/home/bolanos/Bernstein2022/singlecell/L5TPC_exemplar/Current/"


def analyze_v_dist(v):
    """Analyzes V_m distribution"""
    v = v[v < SIGN_TH]  # get rid of spikes
    _, p = normaltest(v)  # there are a bunch of other tests for normality...
    normal = True if p > SIGN_TH else False
    return np.mean(v), np.std(v), normal


def analyze_v_spectrum(v, fs, freq_window):
    """Analyzes the spectrum of V_m TODO: get rid of spikes (properly)"""
    f, pxx = welch(v, fs=fs)
    # cut low freq. part before fitting a line to log-log data
    idx = np.where((freq_window[0] < f) & (f < freq_window[1]))[0]
    coeffs = np.polyfit(np.log10(f[idx]), np.log10(pxx[idx]), deg=1)
    return f, pxx, coeffs


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
    results_dict = {}
    for row_id, stim in stims.iterrows():
        t_start, t_end = stim["t_start"] + t_start_offset, stim["t_end"]
        rate = len(spike_times[(t_start < spike_times) & (spike_times <= t_end)]) / ((t_end - t_start) / 1000)
        v_window = v[(t_start < t) & (t <= t_end)]
        mean, std, normal = analyze_v_dist(v_window)
        f, pxx, coeffs = analyze_v_spectrum(v_window, fs, freq_window)
        results_dict[row_id] = [mean, std, normal, rate, coeffs[0]]
        if plot_results:
            fig_name = os.path.join(FIGS_DIR, "individual", "%s.png" % stim2str(stim))
            plot_vm_dist_spect(v_window, mean, std, spiking, f, pxx, coeffs, freq_window, fig_name)
    results = pd.DataFrame.from_dict(results_dict, orient="index",
                                     columns=["V_mean", "V_std", "V_normal", "rate", "PSD_slope"])
    results.loc[results["rate"] == 0., "PSD_slope"] = np.nan
    return pd.concat([stims, results], axis=1)


if __name__ == "__main__":
    results = []
    for tau in ["tau_fast", "tau_slow"]:
        for amp_cv in ["ampcv0.25", "ampcv0.5", "ampcv0.75", "ampcv1.0", "ampcv1.25"]:
            for sigma in ["sigma0.010", "sigma0.020", "sigma0.030", "sigma0.040",
                          "sigma0.050", "sigma0.060", "sigma0.070"]:
                sim = Simulation(os.path.join(BASE_SIMS_DIR, "AbsoluteShotNoise", "seed161981",
                                              tau, amp_cv, sigma, "BlueConfig"))
                results.append(main(sim))
            for sigma in ["sigma10", "sigma15", "sigma20", "sigma25", "sigma30"]:
                sim = Simulation(os.path.join(BASE_SIMS_DIR, "RelativeShotNoise", "seed161981",
                                              tau, amp_cv, sigma, "BlueConfig"))
                results.append(main(sim))
    for tau in ["tau1", "tau2.5", "tau4", "tau5.5", "tau7"]:
        for sigma in ["sigma0.010", "sigma0.020", "sigma0.030", "sigma0.040",
                      "sigma0.050", "sigma0.060", "sigma0.070"]:
            sim = Simulation(os.path.join(BASE_SIMS_DIR, "OrnsteinUhlenbeck", "seed161981",
                                          tau, sigma, "BlueConfig"))
            results.append(main(sim))
        for sigma in ["sigma10", "sigma15", "sigma20", "sigma25", "sigma30"]:
            sim = Simulation(os.path.join(BASE_SIMS_DIR, "RelativeOrnsteinUhlenbeck", "seed161981",
                                          tau, sigma, "BlueConfig"))
            results.append(main(sim))
    df = pd.concat(results, axis=0, ignore_index=True)
    df.drop(columns=["t_start", "t_end"], inplace=True)
    df.to_pickle("vm.pkl")

    plot_heatmap_grid(df.loc[df["pattern"] == "AbsoluteShotNoise"], "V_mean",
                      os.path.join(FIGS_DIR, "AbsoluteShotNoiseCurrent_V_mean.png"))
    plot_heatmap_grid(df.loc[df["pattern"] == "AbsoluteShotNoise"], "V_std",
                      os.path.join(FIGS_DIR, "AbsoluteShotNoiseCurrent_V_std.png"))
    plot_heatmap_grid(df.loc[df["pattern"] == "RelativeShotNoise"], "V_mean",
                      os.path.join(FIGS_DIR, "RelativeShotNoiseCurrent_V_mean.png"))
    plot_heatmap_grid(df.loc[df["pattern"] == "RelativeShotNoise"], "V_std",
                      os.path.join(FIGS_DIR, "RelativeShotNoiseCurrent_V_std.png"))
    plot_corrs(df.loc[df["pattern"] == "RelativeShotNoise"], ["mean", "std", "amp_cv", "tau"],
               ["V_mean", "V_std"], os.path.join(FIGS_DIR, "RelativeShotNoiseCurrent_corrs.png"))
    plot_heatmap_line(df.loc[df["pattern"] == "AbsoluteOrnsteinUhlenbeck"], "V_mean",
                      os.path.join(FIGS_DIR, "AbsoluteOrnsteinUhlenbeckCurrent_V_mean.png"))
    plot_heatmap_line(df.loc[df["pattern"] == "AbsoluteOrnsteinUhlenbeck"], "V_std",
                      os.path.join(FIGS_DIR, "AbsoluteOrnsteinUhlenbeckCurrent_V_std.png"))
    plot_heatmap_line(df.loc[df["pattern"] == "RelativeOrnsteinUhlenbeck"], "V_mean",
                      os.path.join(FIGS_DIR, "RelativeOrnsteinUhlenbeckCurrent_V_mean.png"))
    plot_heatmap_line(df.loc[df["pattern"] == "RelativeOrnsteinUhlenbeck"], "V_std",
                      os.path.join(FIGS_DIR, "RelativeOrnsteinUhlenbeckCurrent_V_std.png"))






