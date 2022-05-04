"""
Calibrate layer-wise shot noise (by matching layer-wise target firing rates)
author: András Ecker, last update: 05.2022
"""

import os
import numpy as np
import pandas as pd
from bluepy import Simulation
from bluepy.enums import Cell
from utils import load_sim_paths, get_spikes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="ticks", context="notebook")
SIMS_DIR = "/gpfs/bbp.cscs.ch/project/proj96/scratch/home/ecker/simulations"
FIGS_DIR = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/figures/sscx-analysis"


def _group_gids(sim_path):
    """Gets layer-wise E/I gids"""
    sim = Simulation(sim_path)
    c, target = sim.circuit, sim.target
    gid_dict = {}
    for layer_name in ["23", "4", "5", "6"]:
        layer = [2, 3] if layer_name == "23" else int(layer_name)
        for syn_class, class_name in zip(["EXC", "INH"], ["E", "I"]):
            gid_dict["L%s" % layer_name+class_name] = c.cells.ids({"$target": target, Cell.LAYER: layer,
                                                                   Cell.SYNAPSE_CLASS: syn_class})
    return gid_dict


def get_rates(sim_paths, t_start, t_end, norm="total"):
    """Builds DataFrame with layer-wise firing rates
    (it's easier for plotting to build a new one than extend the current one)"""
    assert norm in ["total", "spiking"]
    level_names = sim_paths.index.names
    df_columns = list(level_names)
    norm_t = (t_end - t_start) / 1e3
    gid_dict = _group_gids(sim_paths.iloc[0])
    df_columns.extend(["cell_type", "rate"])
    df = pd.DataFrame(columns=df_columns)

    for idx, sim_path in sim_paths.iteritems():
        df_row = {level_name: idx[i] for i, level_name in enumerate(level_names)}
        sim = Simulation(sim_path)
        _, spiking_gids = get_spikes(sim, t_start, t_end)
        for cell_type, gids in gid_dict.items():
            df_row["cell_type"] = cell_type
            if norm == "total":
                rate = np.isin(spiking_gids, gids).sum() / (len(gids) * norm_t)
            elif norm == "spiking":
                spikes = spiking_gids[np.isin(spiking_gids, gids)]
                rate = len(spikes) / (len(np.unique(spikes)) * norm_t)
            df_row["rate"] = rate
            df = df.append(df_row, ignore_index=True)
    return df


def plot_rates(df, fig_name):
    """Plot layer-wise firing rates (split by E-I types)"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(4, 2, 1)
    sns.barplot(x="shotn_mean_pct", y="rate", hue="shotn_sd_pct", data=df[df["cell_type"] == "L23E"],
                palette="OrRd", ax=ax)
    ax.axhline(0.24, color="red", ls="--", label="RP_2015")
    ax.axhline(0.32, color="gray", ls="--", label="dKS_2007")
    ax.set_title("Excitatory")
    ax.set_xlabel(""); ax.set_ylabel("L23 rate (Hz)")
    ax.legend(frameon=False)
    # for container in ax.containers:
    #     ax.bar_label(container)
    ax = fig.add_subplot(4, 2, 3)
    sns.barplot(x="shotn_mean_pct", y="rate", hue="shotn_sd_pct", data=df[df["cell_type"] == "L4E"],
                palette="OrRd", ax=ax)
    ax.axhline(0.44, color="red", ls="--")
    ax.axhline(0.58, color="gray", ls="--")
    ax.set_xlabel(""); ax.set_ylabel("L4 rate (Hz)")
    ax.legend([], [], frameon=False)
    ax = fig.add_subplot(4, 2, 5)
    sns.barplot(x="shotn_mean_pct", y="rate", hue="shotn_sd_pct", data=df[df["cell_type"] == "L5E"],
                palette="OrRd", ax=ax)
    ax.axhline(1.35, color="red", ls="--")
    ax.axhline(2.37, color="gray", ls="--")
    ax.legend([], [], frameon=False)
    ax.set_xlabel(""); ax.set_ylabel("L5 rate (Hz)")
    ax = fig.add_subplot(4, 2, 7)
    sns.barplot(x="shotn_mean_pct", y="rate", hue="shotn_sd_pct", data=df[df["cell_type"] == "L6E"],
                palette="OrRd", ax=ax)
    ax.axhline(0.47, color="gray", ls="--")
    ax.legend([], [], frameon=False)
    ax.set_ylabel("L6 rate (Hz)")
    ax = fig.add_subplot(4, 2, 2)
    sns.barplot(x="shotn_mean_pct", y="rate", hue="shotn_sd_pct", data=df[df["cell_type"] == "L23I"],
                palette="PuBu", ax=ax)
    ax.axhline(0.19, color="red", ls="--")
    ax.set_title("Inhibitory")
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.legend(title="shotn_sd_pct", frameon=False)
    ax = fig.add_subplot(4, 2, 4)
    sns.barplot(x="shotn_mean_pct", y="rate", hue="shotn_sd_pct", data=df[df["cell_type"] == "L4I"],
                palette="PuBu", ax=ax)
    ax.axhline(0.96, color="red", ls="--")
    ax.legend([], [], frameon=False)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax = fig.add_subplot(4, 2, 6)
    sns.barplot(x="shotn_mean_pct", y="rate", hue="shotn_sd_pct", data=df[df["cell_type"] == "L5I"],
                palette="PuBu", ax=ax)
    ax.axhline(1.34, color="red", ls="--")
    ax.legend([], [], frameon=False)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax = fig.add_subplot(4, 2, 8)
    sns.barplot(x="shotn_mean_pct", y="rate", hue="shotn_sd_pct", data=df[df["cell_type"] == "L6I"],
                palette="PuBu", ax=ax)
    ax.legend([], [], frameon=False)
    ax.set_ylabel("")
    sns.despine(bottom=True, trim=True)
    fig.tight_layout()
    fig.savefig(fig_name, bbox_inches="tight", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    project_name = "1f6af9a9-29c5-4459-ab07-1932f790b32d"
    t_start, t_end = 2000, 7000  # connected network, spontaneous activity
    sim_paths = load_sim_paths(project_name)

    rates = get_rates(sim_paths, t_start, t_end)
    plot_rates(rates, os.path.join(FIGS_DIR, project_name, "rates_Ca1p25.png"))



