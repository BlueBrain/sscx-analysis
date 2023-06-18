"""
Concatenates time dependent weighted connectomes from `conntility` from multiple sims. into one HDF5 file
author: András Ecker, last update: 06.2023
"""

import os
import gc
import warnings
import h5py
import pandas as pd
from conntility.connectivity import TimeDependentMatrix
from utils import load_sim_paths


def load_td_matrix(sim_path, report_name, tag=""):
    h5f_name = os.path.join(os.path.split(sim_path)[0], "td_edges_%s%s.h5" % (report_name, tag))
    return TimeDependentMatrix.from_h5(h5f_name)


def extract_base_properties(conn_mat):
    """Get node properties, edge indicies, and weights at t=0 from `TimeDependentMatrix` object"""
    nodes = conn_mat.vertices.copy()
    nodes = nodes[["gid", "layer", "mtype", "ss_flat_x", "ss_flat_y", "depth"]]
    edge_idx = conn_mat._edge_indices
    conn_mat.at_time(0)
    edges = conn_mat.edges
    return nodes, edge_idx, edges


def edges_at_t(conn_mat, t, seed, pattern):
    """Get edges at time t (and adds some extra index levels)"""
    conn_mat.at_time(t)
    edges = conn_mat.edges
    edges.columns = pd.MultiIndex.from_product([[pattern], [seed], edges.columns], names=["pattern", "seed", "agg_fn"])
    return edges


def edges_to_h5(h5f_name, nodes, edge_idx, edges_at_0, edges, seeds, patterns):
    """Saves everything to one HDF5 file"""
    nodes.to_hdf(h5f_name, key="vertex_properties", format="table")
    edge_idx.to_hdf(h5f_name, key="edge_indices")
    edges_at_0.to_hdf(h5f_name, key="edges_at_0")
    edges.to_hdf(h5f_name, key="edges")
    with h5py.File(h5f_name, "a") as h5f:
        grp = h5f["edges"]
        grp.attrs["seeds"] = seeds
        grp.attrs["patterns"] = patterns


if __name__ == "__main__":
    t = 122000  # ms
    report_name = "rho"
    project_name, seed = "3e3ef5bc-b474-408f-8a28-ea90ac446e24", 1  # all 10 patterns
    conn_mat = load_td_matrix(load_sim_paths(project_name).loc[seed], report_name, "_122000")
    nodes, edge_idx, edges_at_0 = extract_base_properties(conn_mat)
    edges = [edges_at_t(conn_mat, t, seed, "all")]
    del conn_mat
    project_name, seed = "19c8b9d7-5b06-435d-b93d-8277f0c858fe", 1  # none of the edges aka. spontaneous
    conn_mat = load_td_matrix(load_sim_paths(project_name).loc[seed], report_name, "_122000")
    edges.append(edges_at_t(conn_mat, t, seed, "none"))
    del conn_mat
    gc.collect()

    project_name = "3b4f4df5-0f8d-4b5b-9fcf-5407bb9a7519"
    sim_paths = load_sim_paths(project_name)
    level_names = sim_paths.index.names
    for mi, sim_path in sim_paths.items():
        for i, level_name in enumerate(level_names):
            if level_name == "seed":
                seed = mi[i]
            elif "pattern" in level_name:
                patterns_level_name = level_name
                pattern = mi[i]
            else:
                warnings.warn("Unknown MultiIndex level name: %s" % level_name)
        conn_mat = load_td_matrix(sim_path, report_name)
        edges.append(edges_at_t(conn_mat, t, seed, pattern))
        del conn_mat
        gc.collect()
    edges = pd.concat(edges, axis=1, copy=False)

    h5f_name = os.path.join(os.path.split(os.path.split(sim_path)[0])[0], "grouped_td_edges_%s.h5" % report_name)
    edges_to_h5(h5f_name, nodes, edge_idx, edges_at_0, edges,
                seeds=sim_paths.index.get_level_values("seed").unique().to_numpy().tolist(),
                patterns=sim_paths.index.get_level_values(level_name).unique().to_numpy().tolist())


