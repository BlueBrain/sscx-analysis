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
from plots import plot_vm_dist_spect

SPIKE_TH = -30  # mV (NEURON's built in spike threshold)
SIGN_TH = 0.05  # alpha level for significance tests
FIGS_DIR = "/gpfs/bbp.cscs.ch/project/proj83/home/ecker/figures/sscx-analysis/vm_analysis"


def analyze_v_dist(v):
    """Analyzes V_m distribution"""
    _, p = normaltest(v)  # there are a bunch of other tests for normality...
    normal = True if p > SIGN_TH else False
    spiking = np.any(v > SPIKE_TH)
    return np.mean(v), np.std(v), normal, spiking


def analyze_v_spectrum(v, fs, freq_window):
    """Analyzes the spectrum of V_m"""
    f, pxx = welch(v, fs=fs)
    # cut low freq. part before fitting a line to log-log data
    idx = np.where((freq_window[0] < f) & (f < freq_window[1]))[0]
    coeffs = np.polyfit(np.log10(f[idx]), np.log10(pxx[idx]), deg=1)
    return f, pxx, coeffs


def main(sim, t_start_offset=300, freq_window=[10, 5000], plot_results=True):
    # load report with bluepy
    report = sim.report("soma")
    fs = 1 / (report.meta["time_step"] / 1000)
    assert len(report.gids) == 1, "Works with single reported gid atm." \
                                  "(Either add gid selection to the code or report only from a single gid)"
    tmp = report.get()
    t, v = tmp.index.to_numpy().reshape(-1), tmp.to_numpy().reshape(-1)
    # parse stim blocks and iterate over them
    stims = parse_stim_blocks(sim.config)
    results_dict = {}
    for row_id, stim in stims.iterrows():
        v_window = v[(stim["t_start"] + t_start_offset < t) & (t <= stim["t_end"])]
        mean, std, normal, spiking = analyze_v_dist(v_window)
        f, pxx, coeffs = analyze_v_spectrum(v_window, fs, freq_window)
        results_dict[row_id] = [mean, std, normal, spiking, coeffs[0]]
        if plot_results:
            fig_name = os.path.join(FIGS_DIR, "individual", "%s.png" % stim2str(stim))
            plot_vm_dist_spect(v_window, mean, std, spiking, f, pxx, coeffs, freq_window, fig_name)
    results = pd.DataFrame.from_dict(results_dict, orient="index",
                                     columns=["V_mean", "V_std", "V_normal", "V_spiking", "PSD_slope"])
    results.loc[results["V_spiking"] == True, "PSD_slope"] = np.nan
    return pd.concat([stims, results], axis=1)


if __name__ == "__main__":
    bc_path = "/gpfs/bbp.cscs.ch/project/proj83/home/bolanos/Bernstein2022/singlecell/L5TPC_exemplar/" \
              "Current/AbsoluteShotNoise/seed161981/tau_fast/ampcv0.25/sigma0.010/BlueConfig"
    sim = Simulation(bc_path)
    results = main(sim)
    print(results)





