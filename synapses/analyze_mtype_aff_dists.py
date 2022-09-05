"""
Gets synapse path distances (of L5 TTPCs) by markers
last modified: András Ecker 09.2022
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
PALETTE = {"EXC": "#e32b14", "PV": "#3271b8", "Sst": "#67b32e", "5HT3aR": "#c9a021", "VPM": "#4a4657", "POm": "#442e8a"}


def group_gids(c, target):
    """Groups gids based on 3 main inhibitory markers + EXC"""
    exc_gids = c.cells.ids({"$target": target, Cell.SYNAPSE_CLASS: "EXC"})
    pv_gids = c.cells.ids({"$target": target, Cell.MTYPE: {"$regex": "L(23|4|5|6)_(LBC|NBC|CHC)"}})
    sst_gids = np.concatenate((c.cells.ids({"$target": target, Cell.MTYPE: {"$regex": "L(23|4|5|6)_MC"}}),
                               c.cells.ids({"$target": target, Cell.MTYPE: {"$regex": "L(23|4|5|6)_(DBC|BTC)"},
                                            Cell.ETYPE: "cACint"})))
    other_gids = np.concatenate((c.cells.ids({"$target": target, Cell.MTYPE: {"$regex": "L(23|4|5|6)_(DBC|BTC)"},
                                              Cell.ETYPE: ["bNAC", "bAC", "cNAC", "dNAC", "cIR", "bIR", "bSTUT"]}),
                                 c.cells.ids({"$target": target, Cell.MTYPE: {"$regex": "L(23|4|5|6)_(SBC|BP|NGC)"}}),
                                 c.cells.ids({"$target": target, Cell.LAYER: 1})))
    return pd.Series(data=np.concatenate((np.repeat("EXC", len(exc_gids)), np.repeat("PV", len(pv_gids)),
                                          np.repeat("Sst", len(sst_gids)), np.repeat("5HT3aR", len(other_gids)))),
                     index=np.concatenate((exc_gids, pv_gids, sst_gids, other_gids))).sort_index()


def get_pathdists(c, mtype={"$regex": "L5_TPC:(A|B)"}, tc=True, nsamples=100):
    """Gets path distances (from soma to synapse)"""
    gid2marker = group_gids(c, "Mosaic")
    sample_gids = take_n(c.cells.ids({"$target": "central_column_4_region_700um", Cell.MTYPE: mtype}), nsamples)
    results = []
    for gid in tqdm(sample_gids):
        df = c.connectome.afferent_synapses(gid, [Synapse.PRE_GID, Synapse.POST_NEURITE_DISTANCE])
        df["marker"] = gid2marker.loc[df[Synapse.PRE_GID]].to_numpy()
        results.append(df.loc[:, [Synapse.POST_NEURITE_DISTANCE, "marker"]])
    results = pd.concat(results)
    if tc:
        tc_results = []
        for gid in tqdm(sample_gids):
            df = c.projection("Thalamocortical_input_VPM").afferent_synapses(gid, [Synapse.POST_NEURITE_DISTANCE])
            df["marker"] = "VPM"
            tc_results.append(df)
            df = c.projection("Thalamocortical_input_POM").afferent_synapses(gid, [Synapse.POST_NEURITE_DISTANCE])
            df["marker"] = "POm"
            tc_results.append(df)
        tc_results = pd.concat(tc_results)
        results = pd.concat((results, tc_results))
    return results.rename(columns={Synapse.POST_NEURITE_DISTANCE: "dist"})


def plot_dists(df, fig_name):
    """Histplot path distances"""
    fig = plt.figure(figsize=(9., 6.))
    ax = fig.add_subplot(1, 2, 1)
    sns.histplot(x="dist", hue="marker", hue_order=["EXC", "VPM", "POm"], bins=20, binrange=[0, 1500],
                 palette=PALETTE, multiple="stack", ax=ax, data=df)
    ax.set_xlabel("Path distance (um)")
    ax.set_title("L5 TTPC EXC path distances")
    ax2 = fig.add_subplot(1, 2, 2)
    sns.histplot(x="dist", hue="marker", hue_order=["PV", "Sst", "5HT3aR"], bins=20, binrange=[0, 1500],
                 palette=PALETTE, multiple="stack", ax=ax2, data=df)
    ax2.set_xlabel("Path distance (um)")
    ax2.set_title("L5 TTPC INH path distances")
    sns.despine(trim=True)
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    c = Circuit("/gpfs/bbp.cscs.ch/project/proj83/circuits/Bio_M/20200805/CircuitConfig_TC")
    df = get_pathdists(c)
    plot_dists(df, os.path.join(FIGS_PATH, "L5_TTPC_aff_dists.png"))
