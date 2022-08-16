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


def _sum_by_layers(c, gids, g_syns):
    """Returns g_syn sums per layer"""
    sums = np.zeros((6), dtype=int)
    tmp = c.cells.get(gids)["layer"].to_numpy()
    for i, layer in enumerate(LAYERS):
        sums[i] = np.sum(g_syns[tmp == layer])
    return sums


def sum_aff_conductances(c, mtype={"$regex": "L5_TPC:(A|B)"}, nsamples=10000):
    """Sums afferent E and I conductances by layer
    (Could be speed up by defining pre_gids, finding synapse idx, then returning PRE_GID *and* POST_GID
    and doing `groupby` on the properties df... A.E. 08.2022)"""
    # initialize dict to store results (will be converted to a nice df later)
    tmp = {layer:[] for layer in LAYERS}
    tmp["gid"] = []; tmp["Type"] = []
    central_gids = c.cells.ids({"$target": "central_column_4_region_700um", Cell.MTYPE: mtype})
    for gid in tqdm(take_n(central_gids, nsamples)):
        syn_idx = c.connectome.afferent_synapses(gid)  # this works only for a single gid...
        df = c.connectome.synapse_properties(syn_idx, [Synapse.TYPE, Synapse.PRE_GID, Synapse.G_SYNX])
        # process presynaptic E gids:
        idx = df.loc[df[Synapse.TYPE].values >= 100].index
        sums = _sum_by_layers(c, df.loc[idx, Synapse.PRE_GID].to_numpy(), df.loc[idx, Synapse.G_SYNX].to_numpy())
        tmp["gid"].append(gid); tmp["Type"].append("E")
        for i, sum in enumerate(sums):
            tmp[i+1].append(sum)
        # process presynaptic I gids:
        idx = df.loc[df[Synapse.TYPE].values < 100].index
        sums = _sum_by_layers(c, df.loc[idx, Synapse.PRE_GID].to_numpy(), df.loc[idx, Synapse.G_SYNX].to_numpy())
        tmp["gid"].append(gid); tmp["Type"].append("I")
        for i, sum in enumerate(sums):
            tmp[i+1].append(sum)
    return pd.DataFrame.from_dict(tmp)


def plot_conductances(df, fig_name):
    """Boxplot conductances"""
    # pd magic to get plotable df
    df_long = pd.melt(df, id_vars=["Type"], value_vars=LAYERS, value_name="G_SYNX", var_name="Layer")
    fig = plt.figure(figsize=(9., 6.))
    ax = fig.add_subplot(1, 1, 1)
    sns.despine()
    sns.boxplot(x="Layer", y="G_SYNX", hue="Type", hue_order=["E", "I"], order=LAYERS,
                palette=sns.xkcd_palette(["red", "blue"]), ax=ax, data=df_long)
    ax.set_title("L5 TTPC afferents")
    ax.set_ylabel("total G_SYN (nS)")
    #ax.set_yscale("log")
    fig.savefig(fig_name, dpi=100, bbox_inches="tight", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    c = Circuit("/gpfs/bbp.cscs.ch/project/proj83/circuits/Bio_M/20200805/CircuitConfig")
    df = sum_aff_conductances(c)
    plot_conductances(df, os.path.join(FIGS_PATH, "L5_TTPC_aff_cond.png"))
