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


def _counts_by_layers(c, gids):
    """Returns numer of gids per layer"""
    counts = np.zeros((6), dtype=int)
    tmp = c.cells.get(gids)["layer"].to_numpy()
    layer, count = np.unique(tmp, return_counts=True)
    counts[layer-1] = count
    return counts


def get_afferents(c, mtype={"$regex": "L5_TPC:(A|B)"}, nsamples=10000):
    """Counts E and I afferent by layer
    (Could be speed up by defining pre_gids, finding synapse idx, then returning PRE_GID *and* POST_GID
    and doing `groupby` on the properties df... A.E. 08.2022)"""
    # initialize dict to store results (will be converted to a nice df later)
    tmp = {layer:[] for layer in [1, 2, 3, 4, 5, 6]}
    tmp["gid"] = []; tmp["Type"] = []
    central_gids = c.cells.ids({"$target": "central_column_4_region_700um", Cell.MTYPE: mtype})
    for gid in tqdm(take_n(central_gids, nsamples)):
        syn_idx = c.connectome.afferent_synapses(gid)  # this works only for a single gid...
        df = c.connectome.synapse_properties(syn_idx, [Synapse.TYPE, Synapse.PRE_GID, Synapse.G_SYNX])
        # process presynaptic E gids:
        pregids_exc = df.loc[df[Synapse.TYPE].values >= 100][Synapse.PRE_GID].to_numpy()
        counts = _counts_by_layers(c, pregids_exc)
        tmp["gid"].append(gid); tmp["Type"].append("E")
        for i, count in enumerate(counts):
            tmp[i+1].append(count)
        # process presynaptic I gids:
        pregids_inh = df.loc[df[Synapse.TYPE].values < 100][Synapse.PRE_GID].to_numpy()
        counts = _counts_by_layers(c, pregids_inh)
        tmp["gid"].append(gid); tmp["Type"].append("I")
        for i, count in enumerate(counts):
            tmp[i+1].append(count)
    return pd.DataFrame.from_dict(tmp)


def plot_afferents(df, fig_name):
    """Boxplot afferents"""
    # pd magic to get plotable df
    df_long = pd.melt(df, id_vars=["Type"], value_vars=LAYERS, value_name="#Afferents", var_name="Layer")
    fig = plt.figure(figsize=(9., 6.))
    ax = fig.add_subplot(1, 1, 1)
    sns.despine()
    sns.boxplot(x="Layer", y="#Afferents", hue="Type", hue_order=["E", "I"], order=LAYERS,
                palette=sns.xkcd_palette(["red", "blue"]), ax=ax, data=df_long)
    ax.set_title("L5 TTPC afferents")
    #ax.set_yscale("log")
    fig.savefig(fig_name, dpi=100, bbox_inches="tight", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    c = Circuit("/gpfs/bbp.cscs.ch/project/proj83/circuits/Bio_M/20200805/CircuitConfig")
    df = get_afferents(c)
    plot_afferents(df, os.path.join(FIGS_PATH, "L5_TTPC_afferents.png"))
