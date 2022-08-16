"""
Uses `conntility` to get the structural connectome,
then get (normalized, EXC) number of connections between layers - as a proxy of the circuit's recurrent connectivity
author: András Ecker, last update: 06.2022
"""

import os
import numpy as np
import pandas as pd
from conntility.connectivity import ConnectivityMatrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="ticks", context="notebook")
FIGS_PATH = "/gpfs/bbp.cscs.ch/project/proj83/home/ecker/figures"
MTYPES = {"L23": ["L2_IPC", "L2_TPC:A", "L2_TPC:B", "L3_TPC:A", "L3_TPC:C"],
          "L4": ["L4_SSC", "L4_TPC", "L4_UPC"], "L5": ["L5_TPC:A", "L5_TPC:B", "L5_TPC:C", "L5_UPC"],
          "L6": ["L6_BPC", "L6_HPC", "L6_IPC", "L6_TPC:A", "L6_TPC:C", "L6_UPC"]}


def get_conn_mat(circuit_cfg="/gpfs/bbp.cscs.ch/project/proj96/circuits/plastic_v1/BlueConfig",
                 target="hex_O1", h5f_name="conn_mat.h5"):
    """Calculates/loads structural connectivity matrix (of `target`)"""
    if not os.path.isfile(h5f_name):
        from bluepy import Circuit
        load_cfg = {"loading": {"base_target": target, "properties": ["mtype"]}}
        M = ConnectivityMatrix.from_bluepy(Circuit(circuit_cfg), load_cfg)
        M.to_h5(h5f_name)
    else:
        M = ConnectivityMatrix.from_h5(h5f_name)
    return M


def get_norm_nconns(M):
    """Counts EXC connections between layers, and normalizes them with the number of postsynaptic neurons"""
    nrn = M.vertices
    adj = M.matrix.tocsr()  # convert to CSR for fast row indexing
    del M
    cats = list(MTYPES.keys())
    nrn_idx = {cat: nrn.loc[nrn["mtype"].isin(MTYPES[cat])].index.to_numpy() for cat in cats}
    norm_nconns = np.zeros((len(cats), len(cats)), dtype=np.float32)
    for j, post_cat in enumerate(cats):
        n_post_cat = len(nrn_idx[post_cat])
        for i, pre_cat in enumerate(cats):
            nconns = adj[nrn_idx[pre_cat]][:, nrn_idx[post_cat]].count_nonzero()
            norm_nconns[i, j] = nconns / n_post_cat
    df = pd.DataFrame(data=norm_nconns, index=cats, columns=cats)
    return df


def plot_nconns_matrix(nconns_matrix, fig_name):
    """Plots heatmap of normalized number of connections matrix"""
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(1, 1, 1)
    sns.heatmap(data=nconns_matrix, annot=True, fmt=".1f", linewidths=0.2, ax=ax)
    ax.set_xlabel("to layer")
    ax.set_ylabel("from layer")
    ax.set_title("Layer-wise EXC connections")
    fig.savefig(fig_name, dpi=100, bbox_inches="tight", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    M = get_conn_mat()
    nconns_matrix = get_norm_nconns(M)
    plot_nconns_matrix(nconns_matrix, os.path.join(FIGS_PATH, "norm_nconns.png"))





