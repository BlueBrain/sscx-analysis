"""
After running s2f on the functional circuit to validate bouton density
this script loads in the xml into a DataFrame and counts total number of synapses
pathway-by-pathway in order to inspect number of synapses vs. bouton_reduction_factor
last modified: András Ecker 11.2020
"""

import os
from tqdm import tqdm
import xml.etree.ElementTree as et
import numpy as np
import pandas as pd
from libsonata import EdgePopulation
from bluepy import Circuit
from bluepy.enums import Cell
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="ticks", context="notebook")
FIGS_PATH = "/gpfs/bbp.cscs.ch/project/proj83/home/ecker/figures"


def _parse_conn_type(pre_mtype, post_mtype):
    """Connection type (eg. Exc->Exc.: 'ee') parser"""
    pre = "e" if "PC" in pre_mtype or pre_mtype == "L4_SSC" else "i"
    post = "e" if "PC" in post_mtype or post_mtype == "L4_SSC" else "i"
    return pre + post


def xml_to_df(xmlf_name):
    """Reads generated xml into pandas df"""
    df_cols = ["from", "to", "mean_syns_connection", "bouton_reduction_factor", "cv_syns_connection", "type"]
    tmp = {col:[] for col in df_cols}
    xtree = et.parse(xmlf_name)
    for node in xtree.getroot():
        tmp["from"].append(str(node.attrib.get("from")))
        tmp["to"].append(str(node.attrib.get("to")))
        tmp["mean_syns_connection"].append(float(node.attrib.get("mean_syns_connection")))
        tmp["bouton_reduction_factor"].append(float(node.attrib.get("bouton_reduction_factor")))
        tmp["cv_syns_connection"].append(float(node.attrib.get("cv_syns_connection")))
        tmp["type"].append(_parse_conn_type(node.attrib.get("from"), node.attrib.get("to")))
    return pd.DataFrame.from_dict(tmp)


def get_synapse_count(c, df):
    """Counts synpases in the circuit and adds them to the preloaded df"""
    gids = {}; afferent_gids = {}
    df["total_syns_connection"] = np.nan
    for row_id, row in tqdm(df.iterrows()):
        pre_mtype = row["from"]
        post_mtype = row["to"]
        if pre_mtype not in gids:
            pre_gids = c.cells.ids({"$target": "central_column_4_region_700um", Cell.MTYPE: pre_mtype})
            gids[pre_mtype] = pre_gids
        else:
            pre_gids = gids[pre_mtype]
        if post_mtype not in gids:
            post_gids = c.cells.ids({"$target": "central_column_700um", Cell.MTYPE: post_mtype})
            gids[pre_mtype] = post_gids
        else:
            post_gids = gids[post_mtype]

        #it = c.connectome.iter_connections(pre_gids, post_gids, return_synapse_count=True)
        #df.loc[row_id, "total_syns_connection"] = np.sum([conn[2] for conn in it])
        # pure libsonata is faster here than the above 2 bluepy lines
        if post_mtype not in afferent_gids:
            edges = libsonata.EdgePopulation(c.config["connectome"], "", "default")
            selection = edges.afferent_edges(post_gids - 1)  # +/-1 shift between bluepy and sonata
            afferents = np.asarray(edges.source_nodes(selection) + 1)  # +/-1 shift between bluepy and sonata
            afferent_gids[post_mtype] = afferents
        else:
            afferents = afferent_gids[post_mtype]
        df.loc[row_id, "total_syns_connection"] = np.count_nonzero(np.in1d(afferents, pre_gids))
    return df


def plot_validation(df, fig_name):
    """Plots experimental vs. in silico nsyns/conn"""
    # print biasing max value and remove from plots:
    print(df[df["bouton_reduction_factor"] == df["bouton_reduction_factor"].max()])
    df = df[df["bouton_reduction_factor"] != df["bouton_reduction_factor"].max()]

    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    sns.despine()
    df_ee = df[df["type"] == "ee"]
    ax.plot(df_ee["total_syns_connection"], df_ee["bouton_reduction_factor"], ls="None", marker="D", color="magenta",
               markerfacecoloralt="magenta", markersize=5, fillstyle="left", markeredgecolor=(0.3, 0.3, 0.3, 0.5), label="E-E")
    ax2.plot(df_ee["mean_syns_connection"], df_ee["bouton_reduction_factor"], ls="None", marker="D", color="magenta",
                markerfacecoloralt="magenta", markersize=5, fillstyle="left", markeredgecolor=(0.3, 0.3, 0.3, 0.5), label="E-E")
    df_ei = df[df["type"] == "ei"]
    ax.plot(df_ei["total_syns_connection"], df_ei["bouton_reduction_factor"], ls="None", marker="D", color="magenta",
               markerfacecoloralt="cyan", markersize=5, fillstyle="left", markeredgecolor=(0.3, 0.3, 0.3, 0.5), label="E-I")
    ax2.plot(df_ei["mean_syns_connection"], df_ei["bouton_reduction_factor"], ls="None", marker="D", color="magenta",
                markerfacecoloralt="cyan", markersize=5, fillstyle="left", markeredgecolor=(0.3, 0.3, 0.3, 0.5), label="E-I")
    df_ie = df[df["type"] == "ie"]
    ax.plot(df_ie["total_syns_connection"], df_ie["bouton_reduction_factor"], ls="None", marker="D", color="cyan",
               markerfacecoloralt="magenta", markersize=5, fillstyle="left", markeredgecolor=(0.3, 0.3, 0.3, 0.5), label="I-E")
    ax2.plot(df_ie["mean_syns_connection"], df_ie["bouton_reduction_factor"], ls="None", marker="D", color="cyan",
                markerfacecoloralt="magenta", markersize=5, fillstyle="left", markeredgecolor=(0.3, 0.3, 0.3, 0.5), label="I-E")
    df_ii = df[df["type"] == "ii"]
    ax.plot(df_ii["total_syns_connection"], df_ii["bouton_reduction_factor"], ls="None", marker="D", color="cyan",
               markerfacecoloralt="cyan", markersize=5, fillstyle="left", markeredgecolor=(0.3, 0.3, 0.3, 0.5), label="I-I")
    ax2.plot(df_ii["mean_syns_connection"], df_ii["bouton_reduction_factor"], ls="None", marker="D", color="cyan",
                markerfacecoloralt="cyan", markersize=5, fillstyle="left", markeredgecolor=(0.3, 0.3, 0.3, 0.5), label="I-I")
    ax.set_xlabel("Total nsyns")
    ax.set_xscale("log")
    ax.set_yticks([df["bouton_reduction_factor"].min(), 1., df["bouton_reduction_factor"].max()])
    ax.set_ylim([df["bouton_reduction_factor"].min(), df["bouton_reduction_factor"].max()])
    ax.set_ylabel("bouton reduction factor")
    ax2.set_xscale("linear")
    ax2.set_xlim([0, 25])
    ax2.set_xlabel("mean(nsyn/conn)")
    ax2.set_yticks([df["bouton_reduction_factor"].min(), 1., df["bouton_reduction_factor"].max()])
    ax2.set_ylim([df["bouton_reduction_factor"].min(), df["bouton_reduction_factor"].max()])
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles[:4], labels[:4], frameon=False)
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    xmlf_name = "/gpfs/bbp.cscs.ch/project/proj83/circuits/Bio_M/20200731/bioname/boutonDensityValidation.xml"
    c = Circuit("/gpfs/bbp.cscs.ch/project/proj83/circuits/Bio_M/20200805/CircuitConfig")
    df = get_synapse_count(c, xml_to_df(xmlf_name))
    # df.to_pickle("boutons.pkl")
    plot_validation(df, os.path.join(FIGS_PATH, "bouton_density_validation.png"))
