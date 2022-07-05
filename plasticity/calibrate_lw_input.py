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
from plots import plot_lw_rates, plot_lw_rates_pct


FIGS_DIR = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/figures/sscx-analysis"
# reference in vivo firing rates from Reyes-Puerta et al. 2015
RP_2015_RATES = {"L23E": 0.07, "L23I": 0.96, "L4E": 0.61, "L4I": 1.22, "L5E": 1.25, "L5I": 2.35}
# reference in vivo firing rates from deKock and Sakman 2007
dKS_2007_RATES = {"L23E": 0.32, "L4E": 0.58, "L5E": 2.37, "L6E": 0.47}


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
    df.drop("seed", axis=1, errors="ignore")  # just to make sure...
    return df


if __name__ == "__main__":
    project_name = "c5cd0f6a-fe2f-449b-9b56-21bb1cb4968a"
    t_start, t_end = 1500, 3500  # connected network, spontaneous activity
    mean_str, sd_str = "fr_scale", "depol_stdev_mean_ratio"

    sim_paths = load_sim_paths(os.path.join("/gpfs/bbp.cscs.ch/project/proj83/scratch/bbp_workflow/ji_cond_plast/",
                                            project_name, "analyses", "simulations.pkl"))
    ensure_dir(os.path.join(FIGS_DIR, project_name))
    rates = get_rates(sim_paths, t_start, t_end)
    plot_lw_rates(rates.loc[rates["ca"] == 1.1], mean_str, sd_str, RP_2015_RATES, dKS_2007_RATES,
                  os.path.join(FIGS_DIR, project_name, "lw_rates_Ca1p1.png"))
    plot_lw_rates(rates.loc[rates["ca"] == 1.15], mean_str, sd_str, RP_2015_RATES, dKS_2007_RATES,
                  os.path.join(FIGS_DIR, project_name, "lw_rates_Ca1p15.png"))
    plot_lw_rates(rates.loc[rates["ca"] == 1.2], mean_str, sd_str, RP_2015_RATES, dKS_2007_RATES,
                  os.path.join(FIGS_DIR, project_name, "lw_rates_Ca1p2.png"))

    # pct. of target firing rates
    rates["rate_pct"] = 0.0
    for cell_type, ref_rate in RP_2015_RATES.items():
        rates.loc[rates["cell_type"] ==  cell_type, "rate_pct"] = (rates["rate"] / ref_rate) * 100
    # aggregate across cell types
    agg_rates = rates.groupby(["ca", mean_str, sd_str])["rate_pct"].agg("mean")
    valid_rates = agg_rates.loc[agg_rates < 100].sort_values(ascending=False)
    print(valid_rates)
    plot_lw_rates_pct(rates.loc[rates["ca"] == 1.1], mean_str, sd_str,
                      os.path.join(FIGS_DIR, project_name, "lw_rates_pct_Ca1p1.png"))
    plot_lw_rates_pct(rates.loc[rates["ca"] == 1.15], mean_str, sd_str,
                      os.path.join(FIGS_DIR, project_name, "lw_rates_pct_Ca1p15.png"))
    plot_lw_rates_pct(rates.loc[rates["ca"] == 1.2], mean_str, sd_str,
                      os.path.join(FIGS_DIR, project_name, "lw_rates_pct_Ca1p2.png"))



