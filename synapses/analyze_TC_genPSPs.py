"""
Basic stats (for 1 pathway) of PSPs (latency, rise time, half-width, dtc, ampl, CV, failures)
author: Andras Ecker, last update: 06.2021
"""

import os
import h5py
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUT_PATH = "/gpfs/bbp.cscs.ch/project/proj83/home/ecker/psp-validation/out"
FIG_PATH = "/gpfs/bbp.cscs.ch/project/proj83/home/ecker/psp-validation/figures/"
SPIKE_TH = -30  # (mV) NEURON's built in spike threshold
PSP_DETECTION_TH = 0.01  # mV (below which we consider the PSP undetected)
MIN_TRIALS = 5  # min number of correct traces (after spike and failure filters)
PROJS = ["VPM", "POM"]
LAYERS = [1, 2, 3, 4, 5, 6]
TYPES = ["EXC", "PV", "Sst", "5HT3aR"]
# sns.set(style="ticks", context="notebook")


def load_traces(h5f_name, time_range):
    """Loads in traces from .h5 dump and returns dict with (zoomed in) sweeps for each pairs"""
    pair_data = {}
    with h5py.File(h5f_name, "r") as h5f:
        if len(h5f) != 1:
            raise RuntimeError("Unexpected HDF5 layout")
        root = next(iter(h5f.values()))
        if root.name != "/traces":
            raise RuntimeError("Unexpected HDF5 layout")
        for pair in iter(root.values()):
            pre_gid = pair.attrs["pre_gid"]
            post_gid = pair.attrs["post_gid"]
            for k, trial in enumerate(pair["trials"]):
                i, t_tmp = trial
                if k == 0:
                    idx = np.where((time_range[0] <= t_tmp) & (t_tmp < time_range[1]))
                    traces = i.reshape(1, i.shape[0])[0, idx]
                else:
                    traces = np.concatenate((traces, i.reshape(1, i.shape[0])[0, idx]), axis=0)
            pair_data["%s-%s" % (pre_gid, post_gid)] = traces
    t = t_tmp[idx]
    return t, pair_data


def prepare_data(h5f_name, t_stim, time_range, exc):
    """Loads in traces and transfers them to have 0 mean and positive peaks (flips IPSPs to look like EPSPs)"""
    t, pair_data = load_traces(h5f_name, time_range)
    idx_to_stim = np.where(t < t_stim)[0]
    idx_from_stim = np.where(t_stim < t)[0]
    transformed_pair_data = {}
    for pair_id, traces in pair_data.items():
        # filter out spiking trials
        non_spiking_trial_idx = np.where(np.max(traces, axis=1) <= SPIKE_TH)[0]
        if len(non_spiking_trial_idx) >= MIN_TRIALS:
            traces = traces[non_spiking_trial_idx, :]
            mean_trace = np.mean(traces, axis=0)
            voltage_baseline = np.mean(mean_trace[idx_to_stim])
            transformed_trace = traces[:, idx_from_stim] - voltage_baseline
            tmp = 1 if exc else -1  # flip IPSPs
            transformed_trace = tmp * transformed_trace
            transformed_trace[np.where(transformed_trace < 0.)] = 0.
            transformed_pair_data[pair_id] = transformed_trace
        else:
            print("pair: %s couldn't be analyzed due to spiking" % pair_id)
    return t[np.where(t_stim < t)[0]] - t_stim, transformed_pair_data


def _time_of_percent_ampl_rise(percent, t, trace, t_peak, peak_ampl):
    """Calculates time point when rising PSP reaches a given percent"""
    taget_ampl = percent/100. * peak_ampl
    shifted_trace = trace - taget_ampl
    return t[np.argmin(np.abs(shifted_trace[np.where(t < t_peak)[0]]))]


def _time_of_percent_ampl_decay(percent, t, trace, t_peak, peak_ampl):
    """Calculates time point when decaying PSP reaches a given percent"""
    taget_ampl = percent/100. * peak_ampl
    shifted_trace = trace - taget_ampl
    return t[np.argmin(np.abs(shifted_trace[np.where(t_peak < t)[0]]))] + t_peak


def _exp_func(t, tau):
    """Dummy exp. function to pass to curve_fit"""
    return np.exp(-t/tau)


def fit_dtc(t, trace, t_peak, peak_ampl):
    """Fits decaying phase of PSP (till it drops to 80%) and returns time constant"""
    t_80p = _time_of_percent_ampl_decay(20, t, trace, t_peak, peak_ampl)
    idx = np.where((t_peak < t) & (t < t_80p))[0]
    n = len(idx)
    decay_to_fit = trace[idx]
    decay_to_fit /= peak_ampl
    popt, _ = curve_fit(_exp_func, np.linspace(0, t_80p-t_peak, n), decay_to_fit)
    return popt[0]


def extract_features(t, pair_data):
    """Extracts amplitude, latency, rise time, decay time constant, CV and failure rate features of PSPs"""
    peak_amplitudes = []; latencies = []; rise_times = []
    tau_decays = []; half_widths = []; cvs = []; failure_rates = []
    for pair_id, traces in pair_data.items():
        # features extracted from all trials (CV and failure rate)
        n = traces.shape[0]
        peak_ampls = np.max(traces, axis=1)
        idx = np.where(peak_ampls < PSP_DETECTION_TH)[0]
        peak_ampls[idx] = np.nan
        failure_rates.append(len(idx) * 100. / n)
        cvs.append(np.nanstd(peak_ampls) / np.nanmean(peak_ampls))
        # features extracted from the mean trace (excluding failures)
        if len(idx) <= n - MIN_TRIALS:
            detectable_traces = np.delete(traces, idx, axis=0)
            mean_trace = np.mean(detectable_traces, axis=0)
            peak_ampl = np.max(mean_trace)
            t_peak = t[np.argmax(mean_trace)]
            # sanity check if peak is not in the beginning nor at the second half of the interval
            if (t_peak > 1) and (t_peak <= 50):
                peak_amplitudes.append(peak_ampl)
                latencies.append(_time_of_percent_ampl_rise(5, t, mean_trace, t_peak, peak_ampl))
                t_10p = _time_of_percent_ampl_rise(10, t, mean_trace, t_peak, peak_ampl)
                t_90p = _time_of_percent_ampl_rise(90, t, mean_trace, t_peak, peak_ampl)
                rise_times.append(t_90p - t_10p)
                t_50p_rise = _time_of_percent_ampl_rise(50, t, mean_trace, t_peak, peak_ampl)
                t_50p_decay = _time_of_percent_ampl_decay(50, t, mean_trace, t_peak, peak_ampl)
                half_widths.append(t_50p_decay - t_50p_rise)
                tau_decays.append(fit_dtc(t, mean_trace, t_peak, peak_ampl))
            else:
                print("pair: %s has a suspicious peak" % pair_id)
                peak_amplitudes.append(np.nan)
                latencies.append(np.nan)
                rise_times.append(np.nan)
                half_widths.append(np.nan)
                tau_decays.append(np.nan)
        else:
            print("pair: %s contains only failures" % pair_id)
            peak_amplitudes.append(np.nan)
            latencies.append(np.nan)
            rise_times.append(np.nan)
            half_widths.append(np.nan)
            tau_decays.append(np.nan)
    '''
    results = {"amplitudes": np.asarray(peak_amplitudes), "latencies": np.asarray(latencies),
               "rise_times": np.asarray(rise_times), "tau_decays": np.asarray(tau_decays),
               "half_widths": np.asarray(half_widths), "cvs": np.asarray(cvs),
               "failure_rates": np.asarray(failure_rates)}
    '''
    return peak_amplitudes, latencies, rise_times, tau_decays, half_widths, cvs, failure_rates


def create_feature_df():
    """Extracts features for all projections and creates big DataFrame"""
    all_features = {"nuclei": [], "layer": [], "type": [],  "amplitude": [], "latency": [],
                    "rise_time": [], "tau_decay": [], "half_width": [], "cv": [], "failure_rate": []}
    for proj in PROJS:
        for layer in LAYERS:
            for type in TYPES:
                h5f_name = os.path.join(OUT_PATH, "%s-L%i_%s.traces.h5" % (proj, layer, type))
                try:
                    print(h5f_name)
                    t, pair_data = prepare_data(h5f_name, t_stim=800., time_range=[700., 900.], exc=True)
                    amps, latencies, rts, tau_ds, hws, cvs, failure_rates = extract_features(t, pair_data)
                    all_features["nuclei"].extend([proj for _ in range(len(amps))])
                    all_features["layer"].extend([layer for _ in range(len(amps))])
                    all_features["type"].extend([type for _ in range(len(amps))])
                    all_features["amplitude"].extend(amps)
                    all_features["latency"].extend(latencies)
                    all_features["rise_time"].extend(rts)
                    all_features["tau_decay"].extend(tau_ds)
                    all_features["half_width"].extend(hws)
                    all_features["cv"].extend(cvs)
                    all_features["failure_rate"].extend(failure_rates)
                except:
                    print("No file: %s" % h5f_name)
    return pd.DataFrame.from_dict(all_features)


def plot_feature(df, feature, fig_name):
    """Plot selected feature across layers for all types"""
    fig = plt.figure(figsize=(12, 6.5))
    for i, type_ in enumerate(TYPES):
        ax = fig.add_subplot(1, 4, i+1)
        data = df[df["type"] == type_]
        sns.boxplot(data=data, x=feature, y="layer", hue="nuclei", hue_order=["VPM", "POM"],
                    orient="h", showfliers=False, notch=True, ax=ax)
        ax.axvline(data[data["nuclei"] == "VPM"][feature].mean(), color=sns.color_palette()[0], ls='--', alpha=0.5)
        ax.axvline(data[data["nuclei"] == "POM"][feature].mean(), color=sns.color_palette()[1], ls='--', alpha=0.5)
        ax.set_xlim([0, df.quantile(0.95)[feature]])
        ax.set_title("%s" % type_)
        if i != 0:
            ax.set_ylabel("")
        if i == 3:
            ax.legend(frameon=False, loc=1)
        else:
            ax.legend().remove()
    sns.despine(left=True, offset=5, trim=True)
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    df = create_feature_df()
    df.to_pickle("all_proj_features.pkl")
    # df = pd.read_pickle("all_proj_features.pkl")
    for feature in ["amplitude", "latency", "rise_time", "tau_decay", "half_width", "cv", "failure_rate"]:
        plot_feature(df, feature, os.path.join(FIG_PATH, "proj_%s.png" % feature))

