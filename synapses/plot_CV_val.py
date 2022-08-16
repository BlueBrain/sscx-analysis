"""
Calculates CV from traces saved by psp-validation and plots results from CV validation (in vitro vs. in silico CVs)
author: András Ecker, last update 01.2021
"""

import os
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from OU_generator import add_ou_noise

sns.set(style="ticks", context="notebook", font_scale=1.5)
FIGS_PATH = "/gpfs/bbp.cscs.ch/project/proj83/home/ecker/figures"


def load_traces(h5f_name, time_range=[700., 900.]):
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
                v, t_tmp = trial
                if k == 0:
                    idx = np.where((time_range[0] <= t_tmp) & (t_tmp < time_range[1]))
                    traces = v.reshape(1, v.shape[0])[0, idx]
                else:
                    traces = np.concatenate((traces, v.reshape(1, v.shape[0])[0, idx]), axis=0)
            pair_data["%s-%s" % (pre_gid, post_gid)] = traces
    t = t_tmp[idx]
    return t, pair_data


def _get_mean_voltages(t, traces, t_stim, jk):
    """Calculates the mean of the voltage traces before stimulation"""
    if not jk:
        return np.array([np.mean(traces[i, t < t_stim]) for i in range(traces.shape[0])])
    else:  # Jackknife correction
        means = []
        for i in range(traces.shape[0]):
            jk_trace = np.mean(np.delete(traces, i, axis=0), axis=0)  # delete 1 trace and average the rest
            means.append(np.mean(jk_trace[t < t_stim]))
        return np.asarray(means)


def _get_peak_voltages(t, traces, t_stim, exc, jk):
    """Calculates peak of the voltage traces after stimulation"""
    fn = np.max if exc else np.min
    if not jk:
        return np.array([fn(traces[i, t > t_stim]) for i in range(traces.shape[0])])
    else:  # Jackknife correction
        peaks = []
        for i in range(traces.shape[0]):
            jk_trace = np.mean(np.delete(traces, i, axis=0), axis=0)  # delete 1 trace and average the rest
            peaks.append(fn(jk_trace[t > t_stim]))
        return np.asarray(peaks)


def get_amplitudes(t, traces, t_stim, exc, jk):
    """Calculates PSP amplitudes for every sweeps"""
    means = _get_mean_voltages(t, traces, t_stim, jk)
    peaks = _get_peak_voltages(t, traces, t_stim, exc, jk)
    return np.abs(peaks - means)


def get_CVs(h5f_name, type, jk):
    """Calculates CVs for each pair in a pathway"""
    exc = True if type[0] == "e" else False
    t, pair_data = load_traces(h5f_name)
    cvs = []
    for pair_id, traces in pair_data.items():
        traces = add_ou_noise(t, traces)  # add OU noise to traces
        amplitudes = get_amplitudes(t, traces, t_stim=800, exc=exc, jk=jk)
        # amplitudes[amplitudes < 0.05] = np.nan  # excluding failures
        if not jk:
            cv = np.nanstd(amplitudes) / np.nanmean(amplitudes)
        else:  # has to be scaled with (n-1) in case of Jackknife correction
            scale_factor = np.sum(~np.isnan(amplitudes)) - 1
            cv = scale_factor * np.nanstd(amplitudes) / np.nanmean(amplitudes)
        if 0.05 < cv and cv < 1.5:  # just sanity check...
            cvs.append(cv)
    return cvs


def get_CVs_all_pathways(df, traces_path, jk):
    """Calculates CVs and adds them to the df (pre-loaded with the in vitro references)"""
    df["Model_mean"] = np.nan; df["Model_std"] = np.nan
    for id_ in df.index:
        pre_mtype = df.loc[id_, "Pre"]
        post_mtype = df.loc[id_, "Post"]
        h5f_name = os.path.join(traces_path, "%s-%s.traces.h5" % (pre_mtype, post_mtype))
        # print(pre_mtype, post_mtype)
        cvs = get_CVs(h5f_name, df.loc[id_, "Type"], jk=jk)
        # print(cvs)
        df.loc[id_, "Model_mean"] = np.mean(cvs)
        df.loc[id_, "Model_std"] = np.std(cvs)
    return df


def df2tex(df):
    """Converts df to table (rows) used in Latex"""
    for _, row in df.iterrows():
        print(r"%s & %s & %.2f$\pm$%.2f & %.2f$\pm$%.2f & \cite{} \\" % (row["Pre"], row["Post"],
              row["Bio_mean"], row["Bio_std"], row["Model_mean"], row["Model_std"]))


def plot_validation(df, fig_name):
    """Plots experimental vs. in silico PSPs"""
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(1, 1, 1)
    sns.despine()

    ax.plot([0, 1], [0, 1], "--", color="gray", zorder=1)  # diag line
    ax.errorbar(df["Bio_mean"], df["Model_mean"], yerr=df["Model_std"], color="blue", fmt="o", capsize=5, capthick=2, zorder=1)
    ax.errorbar(df["Bio_mean"], df["Model_mean"], xerr=df["Bio_std"], color="red", fmt="o", capsize=5, capthick=2, zorder=1)

    df_ee = df[df["Type"] == "ee"]
    ax.plot(df_ee["Bio_mean"], df_ee["Model_mean"], ls="", marker="D", color="magenta", markerfacecoloralt="magenta",
            markersize=12, fillstyle="left", markeredgecolor=(0.3, 0.3, 0.3, 0.5), zorder=2, label="E-E")
    df_ei = df[df["Type"] == "ei"]
    ax.plot(df_ei["Bio_mean"], df_ei["Model_mean"], ls="", marker="D", color="magenta", markerfacecoloralt="cyan",
            markersize=12, fillstyle="left", markeredgecolor=(0.3, 0.3, 0.3, 0.5), zorder=2, label="E-I")
    df_ie = df[df["Type"] == "ie"]
    ax.plot(df_ie["Bio_mean"], df_ie["Model_mean"], ls="", marker="D", color="cyan", markerfacecoloralt="magenta",
            markersize=12, fillstyle="left", markeredgecolor=(0.3, 0.3, 0.3, 0.5), zorder=2, label="I-E")

    ax.set_title("Validation of first PSP amplitudes' CVs")
    ax.set_xlabel("CV of PSP amplitude\nIn vitro", color="red")
    ax.set_xlim([0, 0.7])
    ax.spines["bottom"].set_color("red")
    ax.tick_params(axis="x", colors="red")
    ax.set_ylabel("In silico\nCV of PSP amplitude", color="blue")
    ax.set_ylim([0, 0.7])
    ax.spines["left"].set_color("blue")
    ax.tick_params(axis="y", colors="blue")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:5], labels[:5], frameon=False)
    ax.text(0.6, 0.08, "r = %.2f\n(n = %i)" % (np.corrcoef(df["Bio_mean"], df["Model_mean"])[0, 1], len(df)),
            fontsize="small")
    fig.savefig(fig_name, dpi=100, bbox_inches="tight", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    jk = False  # Jackknife correction (kinda bootstrapping) of CVs
    bio_path = "/gpfs/bbp.cscs.ch/project/proj83/home/ecker/psp-validation/usecases/sscx/CVs.tsv"
    if not jk:
        df = pd.read_csv(bio_path, sep=',', skiprows=1, names=["Pre", "Post", "Bio_mean", "Bio_std", "Type"],
                         usecols=[0, 1, 2, 4, 6])
    else:  # read in Jackknife corrected CVs (conversion is based on in silico data...)
        df = pd.read_csv(bio_path, sep=',', skiprows=1, names=["Pre", "Post", "Bio_mean", "Bio_std", "Type"],
                         usecols=[0, 1, 3, 4, 6])

    traces_path = "/gpfs/bbp.cscs.ch/project/proj83/home/ecker/psp-validation/out"
    df = get_CVs_all_pathways(df, traces_path, jk)
    print(df)
    print("r=%.2f (n=%i)" % (np.corrcoef(df["Bio_mean"], df["Model_mean"])[0, 1], len(df)))
    plot_validation(df, os.path.join(FIGS_PATH, "CV_validation.png"))
    # df2tex(df)
