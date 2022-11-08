"""
Analyze NMDA component of synapses (by comparing amplitudes of control synapses to synapses that lack NMDA)
aiming to reproduce Figure 9 C) from Markram et al. 1997 (JPhys.)
author: András Ecker, last update 11.2022
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
# from plot_CV_val import load_traces, get_amplitudes

sns.set(style="ticks", context="notebook", font_scale=1.5)
FIGS_PATH = "/gpfs/bbp.cscs.ch/project/proj83/home/ecker/figures"


def get_all_amplitudes(traces_path):
    """Load in traces from `psp-validations` HDF5 format and get amplitudes from both control and NMDA blocked traces
    (this is needed because `psp-validations` filters out anything above -30mV as spiking
    so we can't rely on the amplitudes saved)"""
    f_names = [f_name for f_name in os.listdir(os.path.join(traces_path, "AMPA_NMDA")) if ".traces.h5" in f_name]
    hold_vs = np.sort(np.array([int(f_name.split("L5_TTPC-L5_TTPC")[1].split("mV.traces.h5")[0])
                                for f_name in f_names]))

    hold_v_lst, psps, types = [], [], []
    for hold_v in hold_vs:
        t, pair_data = load_traces(os.path.join(traces_path, "AMPA", "L5_TTPC-L5_TTPC%imV.traces.h5" % hold_v))
        amplitudes = []
        for pair_id, traces in pair_data.items():
            amplitudes.append(np.mean(get_amplitudes(t, traces, t_stim=800, exc=True, jk=False)))
        hold_v_lst.extend([hold_v for _ in range(len(amplitudes))])
        psps.extend(amplitudes)
        types.extend(["D-AP5" for _ in range(len(amplitudes))])
        t, pair_data = load_traces(os.path.join(traces_path, "AMPA_NMDA", "L5_TTPC-L5_TTPC%imV.traces.h5" % hold_v))
        amplitudes = []
        for pair_id, traces in pair_data.items():
            amplitudes.append(np.mean(get_amplitudes(t, traces, t_stim=800, exc=True, jk=False)))
        hold_v_lst.extend([hold_v for _ in range(len(amplitudes))])
        psps.extend(amplitudes)
        types.extend(["ctrl." for _ in range(len(amplitudes))])

    return pd.DataFrame.from_dict({"hold_V": hold_v_lst, "EPSP": psps, "type": types})


def plot_psps_vs_hold_v(df, fig_name):
    """Plots PSPs recorded at different holding potentials"""

    hold_vs = df["hold_V"].unique()
    mean_ctrl_psps = [df.loc[(df["hold_V"] == hold_v) & (df["type"] == "ctrl."), "EPSP"].mean() for hold_v in hold_vs]
    mean_dap5_psps = [df.loc[(df["hold_V"] == hold_v) & (df["type"] == "D-AP5"), "EPSP"].mean() for hold_v in hold_vs]

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(hold_vs, mean_ctrl_psps, "o-", label="ctrl.")
    ax.plot(hold_vs, mean_dap5_psps, "o-", label="D-AP5")
    ax.legend(frameon=False)
    ax.set_ylabel("EPSP amplitude (mV)")
    ax2 = fig.add_subplot(2, 1, 2)
    sns.boxplot(data=df, x="hold_V", y="EPSP", hue="type", hue_order=["ctrl.", "D-AP5"], fliersize=0, ax=ax2)
    sns.stripplot(data=df, x="hold_V", y="EPSP", hue="type", hue_order=["ctrl.", "D-AP5"],
                  dodge=True, size=3, color="black", edgecolor=None, legend=False, ax=ax2)
    ax2.legend().set_title("")
    ax2.legend().get_frame().set_linewidth(0.)
    ax2.set_xlabel("Holding potential (mV)")
    ax2.set_ylabel("EPSP amplitude (mV)")
    sns.despine(trim=True, offset=5)
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    traces_path = "/gpfs/bbp.cscs.ch/project/proj83/home/ecker/psp-validation/out"
    df = get_all_amplitudes(traces_path)
    plot_psps_vs_hold_v(df, os.path.join(FIGS_PATH, "L5_TTPC_EPSP.png"))
