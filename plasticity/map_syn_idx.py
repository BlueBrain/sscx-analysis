"""
Neurodamus reindexes global sonata synapse indices and reports local (starting from 0 for every gid) synapse IDs
This script is taking such a report a creates a mapping (as pandas.DataFrame) between global and local synapse IDs
authors: Sirio Bolaños-Puchet, András Ecker; last update: 11.2021
"""

from tqdm.contrib import tzip
import numpy as np
import pandas as pd
from libsonata import EdgeStorage, Selection
from bluepy import Simulation


def get_afferrent_global_syn_idx(edge_fname, gids):
    """Creates lookup for global/sonata synapse IDs: dict with (1-based) gids as keys
    and synapse ID Selections (that can be flatten to get all idx) as values"""
    edges = EdgeStorage(edge_fname)
    edge_pop = edges.open_population(list(edges.population_names)[0])
    # get (global) afferent synaps idx (from 0-based sonata nodes)
    return {gid: edge_pop.afferent_edges([gid - 1]) for gid in gids}


def _get_afferrent_gids(edge_fname, global_syn_idx):
    """Reads pre gids corresponding to syn_idx from sonata edge file"""
    edges = EdgeStorage(edge_fname)
    edge_pop = edges.open_population(list(edges.population_names)[0])
    # get afferent (0-based sonata) node idx (+1 := (1-based) pre gids)
    return edge_pop.source_nodes(Selection(global_syn_idx)) + 1


def local2global_syn_idx(syn_id_map, gid, local_syn_idx):
    """Maps local [gid][syn_id] synapse ID to global [sonata_syn_id] synapse ID"""
    flat_global_syn_idx = syn_id_map[gid].flatten()
    return flat_global_syn_idx[local_syn_idx]


def create_syn_idx_df(edge_fname, local_syn_idx_mi, pklf_name):
    """Creates pandas DataFrame with global [sonata_syn_id] synapse ID as index
    and local [gid][syn_id] synapse IDs as columns"""
    report_gids = local_syn_idx_mi.get_level_values(0).to_numpy()
    local_syn_idx = local_syn_idx_mi.get_level_values(1).to_numpy()
    sort_idx = np.argsort(report_gids)
    report_gids, local_syn_idx = report_gids[sort_idx], local_syn_idx[sort_idx]
    unique_gids, start_idx, counts = np.unique(report_gids, return_index=True, return_counts=True)

    global_syn_idx_dict = get_afferrent_global_syn_idx(edge_fname, unique_gids)
    global_syn_idx = np.zeros_like(local_syn_idx, dtype=np.int64)
    for gid, start_id, count in tzip(unique_gids, start_idx, counts, miniters=len(unique_gids)/100):
        end_id = start_id + count
        global_syn_idx[start_id:end_id] = local2global_syn_idx(global_syn_idx_dict, gid, local_syn_idx[start_id:end_id])

    sort_idx = np.argsort(global_syn_idx)
    global_syn_idx = global_syn_idx[sort_idx]
    pre_gids = _get_afferrent_gids(edge_fname, global_syn_idx)
    syn_df = pd.DataFrame({"pre_gid": pre_gids, "post_gid": report_gids[sort_idx],
                           "local_syn_idx": local_syn_idx[sort_idx]}, index=global_syn_idx)
    syn_df.index.name = "global_syn_idx"  # stupid pandas...
    syn_df.to_pickle(pklf_name)


if __name__ == "__main__":
    sim = Simulation("/gpfs/bbp.cscs.ch/project/proj96/scratch/home/ecker/simulations/"
                     "LayerWiseEShotNoise_PyramidPatterns/BlueConfig")
    report = sim.report("gmax_AMPA")
    # load the first time step from the report - which won't really be used... we just need the MultiIndex
    data = report.get(t_start=report.meta["start_time"], t_end=report.meta["start_time"] + report.meta["time_step"])

    create_syn_idx_df(sim.circuit.config["connectome"], data.columns,
                      "/gpfs/bbp.cscs.ch/project/proj96/circuits/plastic_v1/syn_idx.pkl")

