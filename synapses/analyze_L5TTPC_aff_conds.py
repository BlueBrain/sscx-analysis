"""
Sums afferent conductances (of L5 TTPCs) by layer and syn. type
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


def _sum_by_layers(c, gids, g_syns):
    """Returns g_syn sums per layer"""
    sums = np.zeros((6), dtype=np.float32)
    tmp = c.cells.get(gids)["layer"].to_numpy()
    for i, layer in enumerate(LAYERS):
        sums[i] = np.sum(g_syns[tmp == layer])
    return sums


def sum_aff_conductances(c, mtype={"$regex": "L5_TPC:(A|B)"}, TC=True, nsamples=10000):
    """Sums afferent E and I conductances by layer"""
    results = np.zeros((nsamples, 1 + len(COLUMNS)), dtype=np.float32)
    central_gids = c.cells.ids({"$target": "central_column_4_region_700um", Cell.MTYPE: mtype})
    sample_gids = take_n(central_gids, nsamples)
    for i, gid in enumerate(tqdm(sample_gids)):
        results[i, 0] = gid
        df = c.connectome.afferent_synapses(gid, [Synapse.TYPE, Synapse.PRE_GID, Synapse.G_SYNX])
        # process presynaptic E gids:
        idx = df.loc[df[Synapse.TYPE].values >= 100].index
        results[i, 1:7] = _sum_by_layers(c, df.loc[idx, Synapse.PRE_GID].to_numpy(),
                                         df.loc[idx, Synapse.G_SYNX].to_numpy())
        # process presynaptic I gids:
        idx = df.loc[df[Synapse.TYPE].values < 100].index
        results[i, 7:13] = _sum_by_layers(c, df.loc[idx, Synapse.PRE_GID].to_numpy(),
                                          df.loc[idx, Synapse.G_SYNX].to_numpy())
    df = pd.DataFrame(index=results[:, 0], data=results[:, 1:], columns=COLUMNS)
    df.drop(columns="L1_EXC", inplace=True)
    if TC:
        tc_results = np.zeros((nsamples, 3), dtype=np.float32)
        for i, gid in enumerate(tqdm(sample_gids)):
            tc_results[i, 0] = gid
            tc_results[i, 1] = c.projection("Thalamocortical_input_VPM").afferent_synapses(gid,
                                            [Synapse.G_SYNX]).sum()[Synapse.G_SYNX]
            tc_results[i, 2] = c.projection("Thalamocortical_input_POM").afferent_synapses(gid,
                                            [Synapse.G_SYNX]).sum()[Synapse.G_SYNX]
        tc_df = pd.DataFrame(index=tc_results[:, 0], data=tc_results[:, 1:], columns=["VPM", "POm"])
        df = pd.concat((df, tc_df), axis=1)
    return df


def plot_conductances(df, fig_name):
    """Boxplot conductances"""
    # pd magic to get plotable df
    df_long = pd.melt(df, var_name="Source", value_name="G_SYNX")
    df_long["Type"] = "EXC"
    df_long.loc[df_long.loc[:, "Source"].str.contains("INH"), "Type"] = "INH"
    df_long["Source"].replace("_EXC", "", regex=True, inplace=True)
    df_long["Source"].replace("_INH", "", regex=True, inplace=True)

    fig = plt.figure(figsize=(9., 6.))
    ax = fig.add_subplot(1, 1, 1)
    sns.despine()
    sns.boxplot(x="Source", y="G_SYNX", hue="Type", hue_order=["EXC", "INH"],
                order=["L%i" % l for l in LAYERS] + ["VPM", "POm"], palette=sns.xkcd_palette(["red", "blue"]),
                showfliers=False, ax=ax, data=df_long)
    ax.set_title("L5 TTPC input conductances")
    ax.set_ylabel("total G_SYN (nS)")
    #ax.set_yscale("log")
    fig.savefig(fig_name, dpi=100, bbox_inches="tight", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    c = Circuit("/gpfs/bbp.cscs.ch/project/proj83/circuits/Bio_M/20200805/CircuitConfig_TC")
    df = sum_aff_conductances(c)
    plot_conductances(df, os.path.join(FIGS_PATH, "L5_TTPC_aff_cond.png"))
