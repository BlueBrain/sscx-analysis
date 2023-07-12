"""
Concatenates time dependent weighted connectomes from `conntility` from multiple sims. into one HDF5 file
author: András Ecker, last update: 07.2023
"""

import os
import gc
import warnings
import h5py
import numpy as np
from scipy.spatial.distance import squareform
from scipy.stats import wasserstein_distance
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


def _get_edge_dists(edges, distance_metric, n_bins=1000):
    """Gets Euclidean/Hamming/EM distance of last time steps of aggregated edges
    Using `pdist` would probably parallelize it, but this way we have better control of e.g. how Hamming is computed..."""
    assert distance_metric in ["Euclidean", "Hamming", "EMD"], "As the implementation doesn't use `pdist`" \
                                                               "you'll need to implement it yourself"
    columns = edges.columns.to_numpy()
    row_idx, col_idx = np.triu_indices(len(columns), k=1)
    dists = np.zeros_like(row_idx, dtype=np.float32)
    if distance_metric == "EMD":
        bin_edges = np.linspace(edges.min().min(), edges.max().max(), n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    for i, (row_id, col_id) in enumerate(zip(tqdm(row_idx, desc="Calculating distances"), col_idx)):
        a, b = edges[columns[row_id]].to_numpy(), edges[columns[col_id]].to_numpy()
        if distance_metric == "Euclidean":
            dists[i] = np.linalg.norm(a - b)
        elif distance_metric == "Hamming":
            dists[i] = len(np.where(np.abs(a - b) > np.finfo(np.float32).eps)[0])  # don't compare against 0 ...
        elif distance_metric == "EMD":
            a_hist, _ = np.histogram(a, bin_edges)
            b_hist, _ = np.histogram(b, bin_edges)
            dists[i] = wasserstein_distance(bin_centers, bin_centers, a_hist, b_hist)
    return dists, columns


def get_all_edge_dists(edges, npzf_name=None):
    """Gets all 3 distances for all edges"""
    if npzf_name is None or (npzf_name is not None and not os.path.isfile(npzf_name)):
        emds, columns = _get_edge_dists(edges, "EMD")
        hamming_dists, _ = _get_edge_dists(edges, "Hamming")
        euclidean_dists, _ = _get_edge_dists(edges, "Euclidean")
        if npzf_name is not None:
            np.savez(npzf_name, emds=emds, hamming_dists=hamming_dists, euclidean_dists=euclidean_dists, columns=columns)
    else:
        tmp = np.load(npzf_name, allow_pickle=True)
        columns = tmp["columns"]
        emds, hamming_dists, euclidean_dists = tmp["emds"], tmp["hamming_dists"], tmp["euclidean_dists"]
    return emds, hamming_dists, euclidean_dists, columns


def _get_ticks(columns):
    """Get ticks and ticklabels from MultiIndex columns (converted to a numpy array)
    The function assumes that repetitions of the same pattern are ordered and pattern name is the first index..."""
    pattern_names, idx = np.unique([col[0] for col in columns], return_inverse=True)
    ticks, ticklabels = [], []
    for pattern_name, pattern_id in zip(pattern_names, np.unique(idx)):
        ticklabels.append(pattern_name)
        tmp = np.where(idx == pattern_id)[0]
        tick = tmp[len(tmp) // 2] if len(tmp) % 2 != 0 else (tmp[int(len(tmp) / 2 - 1)] + tmp[int((len(tmp) / 2))]) / 2
        ticks.append(tick)
    return ticks, ticklabels


def _pair_dists(pattern_dists, pattern_names, dists, columns):
    """Repeats `pattern_dists` (in the same order `dists` are) and indexes out the ones with non-zero distance
    (As above it assumes that pattern name is the first in the index)"""
    assert len(dists) > len(pattern_dists)
    pattern_dists_mat = squareform(pattern_dists)
    pattern_names_dict = {pattern_name: id for id, pattern_name in enumerate(pattern_names)}
    pattern_dists_rep = np.zeros_like(dists)
    row_idx, col_idx = np.triu_indices(len(columns), k=1)
    for i, (row_id, col_id) in enumerate(zip(row_idx, col_idx)):
        pattern_dists_rep[i] = pattern_dists_mat[pattern_names_dict[columns[row_id][0]],
                                                 pattern_names_dict[columns[col_id][0]]]
    idx = np.where(pattern_dists_rep != 0.)[0]
    return pattern_dists_rep[idx], [dists[idx]]


def get_chaning_edges(edges_0, edges):
    """For each pattern, get number of times an edge is changing (across seeds)"""
    edge_idx, changing_edges = np.zeros_like(edges.index.to_numpy()), {}
    for column in edges.columns.to_numpy():
        pattern_name, seed = column[0], column[1]  # assemes fix order
        diffs = edges[column] - edges_0
        idx = diffs[diffs != 0.].index.to_numpy()
        if pattern_name not in changing_edges:
            changing_edges[pattern_name] = edge_idx.copy()
        changing_edges[pattern_name][idx] += 1
    return pd.DataFrame.from_dict(changing_edges)


def _only_max_changes(changing_edges):
    """Keeps only edges that change in all (actually max per pattern...) seeds"""
    changing_edges_bool = changing_edges.copy()
    for pattern_name, max_count in changing_edges.max().items():
        changing_edges_bool.loc[(0 < changing_edges[pattern_name]) & (changing_edges[pattern_name] < max_count),
        pattern_name] = 0
    return changing_edges_bool


def get_changing_nodes(nrn, edge_idx, changing_edges, only_max=True):
    """Gets gids that have edges changing (separated for pre and post population)
    As this usually results in almost the full set of gids some thresholding could be added..."""
    if only_max:  # keep only edges that change in all seeds (per pattern)
        changing_edges_bool = _only_max_changes(changing_edges)
    else:  # get rid of counts and keep everything that's changing
        changing_edges_bool = changing_edges.copy()
        changing_edges_bool[changing_edges >= 1] = 1

    changing_gids = {}
    for pattern_name in changing_edges_bool.columns.to_numpy():
        df = edge_idx.loc[changing_edges_bool.loc[changing_edges_bool[pattern_name] == 1, pattern_name].index]
        changing_gids[pattern_name] = {"pre": nrn.loc[df["row"].unique(), "gid"].to_numpy(),
                                       "post": np.sort(nrn.loc[df["col"].unique(), "gid"].to_numpy())}
    return changing_gids


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

    # TODO call the analysis functions...


