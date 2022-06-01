"""
Calibrate layer-wise input (shot- or OU noise) by matching layer-wise target firing rates
author: András Ecker, last update: 06.2022
"""

import os
import numpy as np
import pandas as pd
from bluepy import Simulation
from bluepy.enums import Cell
from utils import load_sim_paths, get_spikes
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
    project_name = "0e1afde3-2cc3-480e-89e5-950bdb3ce9aa"
    t_start, t_end = 2000, 7000  # connected network, spontaneous activity
    sim_paths = load_sim_paths(project_name)
    rates = get_rates(sim_paths, t_start, t_end)

    plot_lw_rates(rates, "ou_mean_pct", "ou_sd_pct", os.path.join(FIGS_DIR, project_name, "lw_rates.png"))



