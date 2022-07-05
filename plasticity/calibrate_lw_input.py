"""
Calibrate layer-wise input (shot- or OU noise) by matching layer-wise target firing rates
author: András Ecker, last update: 06.2022
"""

import os
import numpy as np
import pandas as pd
from bluepy import Simulation
from bluepy.enums import Cell
from utils import load_sim_paths, ensure_dir, get_spikes
from plots import plot_lw_rates


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


if __name__ == "__main__":
    project_name = "c5cd0f6a-fe2f-449b-9b56-21bb1cb4968a"
    t_start, t_end = 1500, 3500  # connected network, spontaneous activity

    sim_paths = load_sim_paths(os.path.join("/gpfs/bbp.cscs.ch/project/proj83/scratch/bbp_workflow/ji_cond_plast/",
                                            project_name, "analyses", "simulations.pkl"))
    ensure_dir(os.path.join(FIGS_DIR, project_name))
    rates = get_rates(sim_paths, t_start, t_end)
    rates.to_pickle("rates.pkl")

    rates = pd.read_pickle("rates.pkl")

    plot_lw_rates(rates.loc[rates["ca"] == 1.1], "fr_scale", "depol_stdev_mean_ratio",
                  os.path.join(FIGS_DIR, project_name, "lw_rates_Ca1p1.png"))
    plot_lw_rates(rates.loc[rates["ca"] == 1.15], "fr_scale", "depol_stdev_mean_ratio",
                  os.path.join(FIGS_DIR, project_name, "lw_rates_Ca1p15.png"))
    plot_lw_rates(rates.loc[rates["ca"] == 1.2], "fr_scale", "depol_stdev_mean_ratio",
                  os.path.join(FIGS_DIR, project_name, "lw_rates_Ca1p2.png"))



