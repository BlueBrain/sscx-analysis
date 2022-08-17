"""
Counts number of afferent synapses (to L5 TTPCs) by layer and syn. type
last modified: András Ecker 09.2020
"""

import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from bluepy import Circuit
from bluepy.enums import Cell, Synapse
from bluepy.utils import take_n
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="ticks", context="notebook")
FIGS_PATH = "/gpfs/bbp.cscs.ch/project/proj83/home/ecker/figures"
LAYERS = [i+1 for i in range(6)]
COLUMNS = ["L%i_EXC" % l for l in LAYERS] + ["L%i_INH" % l for l in LAYERS]


def _counts_by_layers(c, gids):
    """Returns numer of gids per layer"""
    counts = np.zeros((6), dtype=int)
    tmp = c.cells.get(gids)["layer"].to_numpy()
    layer, count = np.unique(tmp, return_counts=True)
    counts[layer-1] = count
    return counts


def get_afferents(c, mtype={"$regex": "L5_TPC:(A|B)"}, TC=True, nsamples=10000):
    """Counts E and I afferent by layer"""
    results = np.zeros((nsamples, 1 + len(COLUMNS)), dtype=int)
    central_gids = c.cells.ids({"$target": "central_column_4_region_700um", Cell.MTYPE: mtype})
    sample_gids = take_n(central_gids, nsamples)
    for i, gid in enumerate(tqdm(sample_gids)):
        results[i, 0] = gid
        df = c.connectome.afferent_synapses(gid, [Synapse.TYPE, Synapse.PRE_GID])
        # process presynaptic E gids:
        pregids_exc = df.loc[df[Synapse.TYPE].values >= 100][Synapse.PRE_GID].to_numpy()
        results[i, 1:7] = _counts_by_layers(c, pregids_exc)
        # process presynaptic I gids:
        pregids_inh = df.loc[df[Synapse.TYPE].values < 100][Synapse.PRE_GID].to_numpy()
        results[i, 7:13] = _counts_by_layers(c, pregids_inh)
    df = pd.DataFrame(index=results[:, 0], data=results[:, 1:], columns=COLUMNS)
    df.drop(columns="L1_EXC", inplace=True)
    if TC:
        tc_results = np.zeros((nsamples, 3), dtype=int)
        for i, gid in enumerate(tqdm(sample_gids)):
            tc_results[i, 0] = gid
            tc_results[i, 1] = len(c.projection("Thalamocortical_input_VPM").afferent_synapses(gid))
            tc_results[i, 2] = len(c.projection("Thalamocortical_input_POM").afferent_synapses(gid))
        tc_df = pd.DataFrame(index=tc_results[:, 0], data=tc_results[:, 1:], columns=["VPM", "POm"])
        df = pd.concat((df, tc_df), axis=1)
    return df


def plot_afferents(df, fig_name):
    """Boxplot afferents"""
    # pd magic to get plotable df
    df_long = pd.melt(df, var_name="Source", value_name="#Afferents")
    df_long["Type"] = "EXC"
    df_long.loc[df_long.loc[:, "Source"].str.contains("INH"), "Type"] = "INH"
    df_long["Source"].replace("_EXC", "", regex=True, inplace=True)
    df_long["Source"].replace("_INH", "", regex=True, inplace=True)

    fig = plt.figure(figsize=(9., 6.))
    ax = fig.add_subplot(1, 1, 1)
    sns.despine()
    sns.boxplot(x="Source", y="#Afferents", hue="Type", hue_order=["EXC", "INH"],
                order=["L%i" % l for l in LAYERS] + ["VPM", "POm"], palette=sns.xkcd_palette(["red", "blue"]),
                showfliers=False, ax=ax, data=df_long)
    ax.set_title("L5 TTPC afferents")
    #ax.set_yscale("log")
    fig.savefig(fig_name, dpi=100, bbox_inches="tight", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    c = Circuit("/gpfs/bbp.cscs.ch/project/proj83/circuits/Bio_M/20200805/CircuitConfig_TC")
    df = get_afferents(c)
    plot_afferents(df, os.path.join(FIGS_PATH, "L5_TTPC_afferents.png"))
