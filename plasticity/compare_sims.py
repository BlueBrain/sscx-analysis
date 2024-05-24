"""
Compare basic stats. of different sims
author: András Ecker, last update: 05.2024
"""

import os
import numpy as np
import pandas as pd
from bluepy import Simulation, Circuit
from bluepy.enums import Cell
import utils
from plots import plot_rate_comparison, plot_rho_comparison, plot_lw_comparison

FIGS_DIR = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/figures/sscx-analysis/sim_comparison"


def get_rates(c, sim_paths, t_start, t_end):
    """Gets populational and layer-wise single cell (EXC) firing rates"""
    gids = c.cells.ids({"$target": "hex_O1", Cell.SYNAPSE_CLASS: "EXC"})
    nrn = c.cells.get(gids, ["layer"])
    dfs = []
    for i, (sim_name, sim_path) in enumerate(sim_paths.items()):
        spike_times, spiking_gids = utils.get_spikes(Simulation(sim_path), t_start, t_end, gids=gids)
        if i == 0:
            rate = utils.calc_rate(spike_times, len(np.unique(spiking_gids)), t_start, t_end)
            rates = {sim_name: rate}
            t = np.linspace(t_start, t_end, len(rate))
        else:
            rates[sim_name] = utils.calc_rate(spike_times, len(np.unique(spiking_gids)), t_start, t_end)
        gids_, sc_rates = utils.calc_sc_rate(spiking_gids, t_start, t_end)
        df = pd.DataFrame(data=sc_rates, index=gids_, columns=["rate"])
        df["layer"] = nrn.loc[gids_]
        df["sim"] = sim_name
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    return t, rates, df


def _diffs_dict2pct_df(diffs, key_var):
    """Convert difference dict to DataFrame of depressed/potentiated percentages"""
    keys, data = [], np.zeros((len(diffs), 2), dtype=np.float32)
    for i, (key, val) in enumerate(diffs.items()):
        keys.append(key)
        val_nz = val[val != 0.]
        data[i, 0] = 100 * len(val_nz[val_nz > 0.]) / len(val)
        data[i, 1] = 100 * len(val_nz[val_nz < 0.]) / len(val)
    df = pd.DataFrame(data=data, columns=["pot.", "dep."])
    df[key_var] = keys
    return df


def get_diffs_by(c, data, split_by="layer"):
    """Splits total changes (the upreported ones as well) into categories"""
    split_data = utils.split_synapse_report(c, data, split_by)
    split_data = {key: var for key, var in split_data.items() if var.shape[1] != 0}
    split_data = utils.update_split_data(c, "rho", split_data, split_by)
    diffs = {key: val[-1] - val[0] for key, val in split_data.items()}
    return _diffs_dict2pct_df(diffs, split_by)


def compare_features(sim_paths, h5f_names, palette, t_start, t_end, compare_with="old (seed1)"):
    """Compare spike stats. and the evolution of rho values"""
    c = Circuit(sim_paths["old (seed1)"])
    t_rate, rates, rate_df = get_rates(c, sim_paths, t_start, t_end)

    data_compare = utils.load_synapse_report(h5f_names[compare_with], t_start=t_start, t_end=t_end, return_idx=True)
    data_compare.sort_index(axis=1, inplace=True)
    t_report = data_compare.index.to_numpy()
    df = get_diffs_by(c, data_compare)
    df["sim"] = compare_with
    dfs = [df]
    data_compare = data_compare.to_numpy()
    means = {compare_with: np.mean(data_compare, axis=1)}
    dists = {compare_with: np.linalg.norm(np.diff(data_compare, axis=0), axis=1)}

    dists_compare = {}
    for sim_name, h5f_name in h5f_names.items():
        if sim_name != compare_with:
            data = utils.load_synapse_report(h5f_name, t_start=t_start, t_end=t_end, return_idx=True)
            data.sort_index(axis=1, inplace=True)
            df = get_diffs_by(c, data)
            df["sim"] = sim_name
            dfs.append(df)
            data = data.to_numpy()
            means[sim_name] = np.mean(data, axis=1)
            dists[sim_name] = np.linalg.norm(np.diff(data, axis=0), axis=1)
            dists_compare[sim_name] = np.linalg.norm(data_compare - data, axis=1)
    diff_df = pd.concat(dfs, axis=0)

    plot_rate_comparison(t_rate, rates, t_report, dists, palette, os.path.join(FIGS_DIR, "rates.png"))
    plot_rho_comparison(t_report, means, dists_compare, palette, os.path.join(FIGS_DIR, "rhos.png"))
    plot_lw_comparison(rate_df, diff_df, palette, os.path.join(FIGS_DIR, "lw_changes.png"))


if __name__ == "__main__":
    sim_paths = {"old (seed1)": "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/simulations/3e3ef5bc-b474-408f-8a28-ea90ac446e24/0/BlueConfig",
                 "old (seed19)": "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/simulations/f4e2bde0-c83f-4f52-8a10-5ee114540c34/0/BlueConfig",
                 "old (seed31)": "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/simulations/d3b0ddac-3487-4e30-87f1-ee4343ef0f5e/0/BlueConfig",
                 "new (seed1)": "/gpfs/bbp.cscs.ch/project/proj96/scratch/home/ecker/simulations/6449d510-65d9-43d7-a366-84edf56cb8be/0/BlueConfig_1"}
    h5f_names = {"old (seed1)": "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/simulations/3e3ef5bc-b474-408f-8a28-ea90ac446e24/0/rho.h5",
                 "old (seed19)": "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/simulations/f4e2bde0-c83f-4f52-8a10-5ee114540c34/0/rho.h5",
                 "old (seed31)": "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/simulations/d3b0ddac-3487-4e30-87f1-ee4343ef0f5e/0/rho.h5",
                 "new (seed1)": "/gpfs/bbp.cscs.ch/project/proj96/scratch/home/ecker/simulations/6449d510-65d9-43d7-a366-84edf56cb8be/0/output-1/rho.h5"}
    palette = {"old (seed1)": "red", "old (seed19)": "lightcoral", "old (seed31)": "darkred", "new (seed1)": "green"}
    t_start, t_end = 87000, 172000  # 0, 86000

    compare_features(sim_paths, h5f_names, palette, t_start, t_end)






