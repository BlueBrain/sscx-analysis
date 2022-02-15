"""
Loads synapse clusters saved by `assemblyfire` and checks their total changes in the synapse report
author: András Ecker, last update: 02.2022
"""

import os
from copy import deepcopy
import numpy as np
import pandas as pd
import utils
from plots import plot_2x2_cond_probs, plot_nx2_cond_probs, plot_late_assembly_diffs

FIGS_DIR = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/figures/v7_assemblies"


def _sort_keys(key_list):
    """Sort keys of assembly idx. If -1 is part of the list (standing for non-assembly) then that comes last"""
    if -1 not in key_list:
        return np.sort(key_list)
    else:
        keys = np.array(key_list)
        return np.concatenate((np.sort(keys[keys >= 0]), np.array([-1])))


def get_change_probs(syn_clusters, diffs):
    """Gets probabilities of potentiated, unchanged, and depressed synapses"""
    probs = {}
    for assembly_id, syn_cluster in syn_clusters.items():
        assembly_diffs = diffs.loc[syn_cluster.index]
        n_syns = len(assembly_diffs)
        potentiated, depressed = len(assembly_diffs[assembly_diffs > 0]), len(assembly_diffs[assembly_diffs < 0])
        assembly_probs = np.array([potentiated, 0., depressed]) / n_syns
        assembly_probs[1] = 1 - np.sum(assembly_probs)
        probs[assembly_id] = assembly_probs
    return probs


def group_diffs(syn_clusters, diffs):
    """
    Groups efficacy differences (rho at last time step - rho at first time step) in sample neurons from assemblies
    based on 2 criteria: 1) synapse is coming from assembly neuron vs. non-assembly neuron (fist key (-1 for non-assembly))
                         2) synapse is part of a synapse cluster (clustering done in `assemblyfire`) vs. not (second key)
    This seemed to be a good first implementation and `get_michelson_contrast()` below is based in this format...
    but `diffs2df()` below, which creates a DataFrame which is easier to understand and plot
    """
    grouped_diffs = {}
    for assembly_id, syn_cluster in syn_clusters.items():
        pre_asssembly_idx = [int(label.split("assembly")[1]) for label in syn_cluster.columns.to_list()
                             if label.split("assembly")[0] == '']
        assembly_diffs = {pre_assembly_id: {} for pre_assembly_id in pre_asssembly_idx + [-1]}  # -1 for non-assembly
        for pre_assembly_id in pre_asssembly_idx:
            assembly_syns = syn_cluster["assembly%i" % pre_assembly_id]
            assembly_diffs[pre_assembly_id][1] = diffs.loc[assembly_syns[assembly_syns >= 0].index]
            assembly_diffs[pre_assembly_id][0] = diffs.loc[assembly_syns[assembly_syns == -1].index]
        non_assembly_syns = syn_cluster["non_assembly"]
        assembly_diffs[-1] = {1: diffs.loc[non_assembly_syns[non_assembly_syns >= 0].index],
                              0: diffs.loc[non_assembly_syns[non_assembly_syns == -1].index]}
        grouped_diffs[assembly_id] = assembly_diffs
    return grouped_diffs


def get_fracs(syn_clusters):
    """Gets fraction of synapses coming from assemblies (and for non-assembly saved with key: -1)"""
    fracs = {}
    for assembly_id, syn_cluster in syn_clusters.items():
        pre_asssembly_idx = [int(label.split("assembly")[1]) for label in syn_cluster.columns.to_list()
                             if label.split("assembly")[0] == '']
        assembly_fracs = {pre_assembly_id: {} for pre_assembly_id in pre_asssembly_idx + [-1]}  # -1 for non-assembly
        for pre_assembly_id in pre_asssembly_idx:
            assembly_syns = syn_cluster["assembly%i" % pre_assembly_id]
            assembly_fracs[pre_assembly_id] = len(assembly_syns[assembly_syns >= -1]) / len(assembly_syns)
        non_assembly_syns = syn_cluster["non_assembly"]
        assembly_fracs[-1] = len(non_assembly_syns[non_assembly_syns >= -1]) / len(non_assembly_syns)
        fracs[assembly_id] = assembly_fracs
    return fracs


def get_grouped_diffs(seed, sim_path, report_name, late_assembly=False):
    """Wrapper of other functions that together load and pre-calculate/group stuff for statistic tests and plotting"""
    syn_clusters, gids = utils.load_synapse_clusters(seed, sim_path, late_assembly)
    diffs = utils.get_synapse_changes(sim_path, report_name, gids)
    probs = get_change_probs(syn_clusters, diffs)
    fracs = get_fracs(syn_clusters)
    grouped_diffs = group_diffs(syn_clusters, diffs)
    return probs, fracs, grouped_diffs


def get_michelson_contrast(probs, grouped_diffs):
    """Gets Michelson contrast (aka. visibility) defined as:
    P(pot/dep | condition) - P(pot/dep) / (P(pot/dep | condition) + P(pot/dep))"""
    pot_contrasts, dep_contrasts = {}, {}
    for assembly_id, uncond_probs in probs.items():
        p_pot, p_dep = uncond_probs[0], uncond_probs[2]  # unchanged is the 2nd element, which we won't use here
        pre_assembly_idx = _sort_keys(list(grouped_diffs[assembly_id].keys()))
        pot_contrast, dep_contrast = np.zeros((len(pre_assembly_idx), 2)), np.zeros((len(pre_assembly_idx), 2))
        for i, pre_assembly_id in enumerate(pre_assembly_idx):
            for j, clustered in enumerate([1, 0]):  # looks useless, but this way clustered comes first in the plots...
                df = grouped_diffs[assembly_id][pre_assembly_id][clustered]
                n_syns = len(df)
                if n_syns:
                    p_pot_cond = len(df[df > 0]) / n_syns
                    pot_contrast[i, j] = (p_pot_cond - p_pot) / (p_pot_cond + p_pot)
                    p_dep_cond = len(df[df < 0]) / n_syns
                    dep_contrast[i, j] = (p_dep_cond - p_dep) / (p_dep_cond + p_dep)
                else:
                    pot_contrast[i, j], dep_contrast[i, j] = np.nan, np.nan
        pot_contrasts[assembly_id], dep_contrasts[assembly_id] = pot_contrast, dep_contrast
    return pot_contrasts, dep_contrasts


def diffs2df(grouped_diffs):
    """Creates a DataFrame from `grouped_diffs` (easier to plot and merge with extra morph. features)"""
    pre_assembly_idx = _sort_keys(list(grouped_diffs.keys()))
    dfs = []
    for pre_assembly_id in pre_assembly_idx:
        df = grouped_diffs[pre_assembly_id][1].to_frame()
        df["pre_assembly"] = pre_assembly_id
        df["clustered"] = True
        dfs.append(df)
        df = grouped_diffs[pre_assembly_id][0].to_frame()
        df["pre_assembly"] = pre_assembly_id
        df["clustered"] = False
        dfs.append(df)
    return pd.concat(dfs).sort_index()


def main(project_name):
    report_name = "rho"
    sim_paths = utils.load_sim_paths(project_name)
    level_names = sim_paths.index.names
    assert len(level_names) == 1 and level_names[0] == "seed"
    utils.ensure_dir(os.path.join(FIGS_DIR, project_name))

    '''
    for seed, sim_path in sim_paths.iteritems():
        probs, _, grouped_diffs = get_grouped_diffs(seed, sim_path, report_name)
        pot_contrasts, dep_contrasts = get_michelson_contrast(probs, grouped_diffs)
        fig_name = os.path.join(FIGS_DIR, project_name, "syn_clust_plast_seed%i.png" % seed)
        plot_2x2_cond_probs(probs, pot_contrasts, dep_contrasts, fig_name)
    '''

    # morph_df = utils.load_extra_morph_features(["loc", "dist", "diam", "br_ord"])
    for seed, sim_path in sim_paths.iteritems():
        probs, fracs, grouped_diffs = get_grouped_diffs(seed, sim_path, report_name, late_assembly=True)
        pot_contrasts, dep_contrasts = get_michelson_contrast(probs, grouped_diffs)
        fig_name = os.path.join(FIGS_DIR, project_name, "late_assembly_cond_probs_seed%i.png" % seed)
        # late assembly_id is 0 (it just happens to be... and is hard coded in `assemblyfire` as well)
        # and as the rest of the functions create dicts, one has to select it by the `0` key here...
        plot_nx2_cond_probs(probs[0], fracs[0], pot_contrasts[0], dep_contrasts[0], fig_name)
        df = diffs2df(deepcopy(grouped_diffs[0]))
        fig_name = os.path.join(FIGS_DIR, project_name, "late_assembly_diff_stats_seed%i.png" % seed)
        plot_late_assembly_diffs(df.loc[df["pre_assembly"] >= 0], fig_name)
        # df = pd.concat([df, morph_df.loc[df.index]], axis=1)


if __name__ == "__main__":
    project_name = "e0fbb0c8-07a4-49e0-be7d-822b2b2148fb"
    main(project_name)
