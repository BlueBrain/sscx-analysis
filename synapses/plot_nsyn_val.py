"""
Compares nsyns/conns vs. experimental number of synapses per connections
last modified: András Ecker 01.2021
"""

import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from bluepy import Circuit
from bluepy.enums import Cell
from bluepy.utils import take_n
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="ticks", context="notebook", font_scale=1.5)
FIGS_PATH = "/gpfs/bbp.cscs.ch/project/proj83/home/ecker/figures"


def _parse_conn_type(pre_mtype, post_mtype):
    """Connection type (eg. Exc->Exc.: 'ee') parser"""
    pre = "e" if "PC" in pre_mtype or pre_mtype == "L4_SSC" else "i"
    post = "e" if "PC" in post_mtype or post_mtype == "L4_SSC" else "i"
    return pre + post


def get_synapse_count(c, df):
    """Counts synapses and adds them to the df (pre-loaded with the in vitro references)"""
    # add new columns to the df to store model results
    df["Model_mean"] = np.nan; df["Model_std"] = np.nan; df["Type"] = ""
    for id_ in tqdm(df.index):
        pre_mtype = df.loc[id_, "Pre"]
        post_mtype = df.loc[id_, "Post"]
        df.loc[id_, "Type"] = _parse_conn_type(pre_mtype, post_mtype)
        pre_gids = c.cells.ids({"$target": "Mosaic", Cell.MTYPE: pre_mtype})
        # post gids only from the "central column" to avoid boundary artifacts
        post_gids = c.cells.ids({"$target": "central_column_4_region_700um", Cell.MTYPE: post_mtype})
        # count synapses:
        #nsyns = c.stats.sample_pathway_synapse_count(1000, pre_gids, post_gids)
        it = c.connectome.iter_connections(pre_gids, post_gids, return_synapse_count=True, shuffle=True)
        nsyns = [conn[2] for conn in take_n(it, 100000)]  # sample pairs because it runs forever...
        if len(nsyns):
            df.loc[id_, "Model_mean"] = np.mean(nsyns)
            df.loc[id_, "Model_std"] = np.std(nsyns)
    return df


def get_TC_synapse_counts(c, df):
    """Counts TC synapses and adds them to the df"""
    connectome = c.projection("Thalamocortical_input_VPM")
    offset = len(df) + 1
    for i, post_mtype in enumerate(["L4_SSC", "L4_TPC", "L4_UPC"]):
        df.loc[offset + i, "Pre"] = "TC"
        df.loc[offset + i, "Post"] = "TC"
        df.loc[offset + i, "Type"] = "te"
        df.loc[offset + i, "Bio_mean"] = 7.  # hard coded from Gil et al. (1999)
        df.loc[offset + i, "Bio_std"] = 4.9  # hard coded from Gil et al. (1999)
        post_gids = c.cells.ids({"$target": "central_column_4_region_700um", Cell.MTYPE: post_mtype})
        it = connectome.iter_connections(None, post_gids, return_synapse_count=True, shuffle=True)
        nsyns = [conn[2] for conn in take_n(it, 100000)]  # sample pairs because it runs forever...
        if len(nsyns):
            df.loc[offset+i, "Model_mean"] = np.mean(nsyns)
            df.loc[offset+i, "Model_std"] = np.std(nsyns)
    return df


def plot_validation(df, fig_name):
    """Plots experimental vs. in silico nsyns/conn"""
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(1, 1, 1)
    sns.despine()

    ax.plot([0,50], [0,50], "--", color="gray", zorder=1)  # diag line
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
    df_ii = df[df["Type"] == "ii"]
    ax.plot(df_ii["Bio_mean"], df_ii["Model_mean"], ls="", marker="D", color="cyan", markerfacecoloralt="cyan",
            markersize=12, fillstyle="left", markeredgecolor=(0.3, 0.3, 0.3, 0.5), zorder=2, label="I-I")
    df_te = df[df["Type"] == "te"]
    ax.plot(df_te["Bio_mean"], df_te["Model_mean"], ls="", marker="D", color="yellow", markerfacecoloralt="magenta",
            markersize=12, fillstyle="left", markeredgecolor=(0.3, 0.3, 0.3, 0.5), zorder=2, label="T-E")

    ax.set_title("Validation of nsyns/conn")
    ax.set_xlabel("Nr. synapses/connection\nIn vitro", color="red")
    ax.set_xlim([0, 30])
    ax.spines["bottom"].set_color("red")
    ax.tick_params(axis="x", colors="red")
    ax.set_ylabel("In silico\nNr. synapses/connection", color="blue")
    ax.set_ylim([0, 30])
    ax.spines["left"].set_color("blue")
    ax.tick_params(axis="y", colors="blue")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:5], labels[:5], frameon=False)
    ax.text(25, 3, "r = %.2f\n(n = %i)" % (np.corrcoef(df["Bio_mean"], df["Model_mean"])[0, 1], len(df)),
            fontsize="small")
    fig.savefig(fig_name, dpi=100, bbox_inches="tight", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    bio_path = "/gpfs/bbp.cscs.ch/project/proj83/var/git/entities/recipe/connectome/nsyn_per_connection_20160509_full.tsv"
    df = pd.read_csv(bio_path, skiprows=1, names=["Pre", "Post", "Bio_mean", "Bio_std"],
                     usecols=[0, 1, 2, 3], delim_whitespace=True)
    circuitconfig = "/gpfs/bbp.cscs.ch/project/proj83/circuits/Bio_M/20200805/CircuitConfig_TC"
    circuit = Circuit(circuitconfig)

    df = get_synapse_count(circuit, df)
    df = get_TC_synapse_counts(circuit, df)
    # df.to_pickle("nsyns_conn.pkl")
    # df = pd.read_pickle("nsyns_conn.pkl")
    print("r=%.2f (n=%i)" % (np.corrcoef(df["Bio_mean"], df["Model_mean"])[0, 1], len(df)))
    plot_validation(df, os.path.join(FIGS_PATH, "nsyns_validation.png"))
