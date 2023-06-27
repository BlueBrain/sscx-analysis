"""
Plot evolution of synaptic weights (in plasticity simulations) over time
author: András Ecker, last update: 05.2023
"""

import os
from tqdm import tqdm
from copy import deepcopy
import numpy as np
from bluepy import Circuit, Simulation
import utils
import plots

FIGS_DIR = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/figures/sscx-analysis"
ASSEMBLY_FIGS_DIR = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/figures/v7_assemblies"


def get_total_change_by(sim_path, report_name, split_by="layer", return_data=False):
    """Loads full report, splits it, updates it with non-reported data, and gets total change (last-first time step)"""
    c = Circuit(sim_path)
    h5f_name = os.path.join(os.path.split(sim_path)[0], "%s.h5" % report_name)
    data = utils.load_synapse_report(h5f_name, return_idx=True)
    split_data = utils.split_synapse_report(c, data, split_by)
    del data
    split_data = utils.update_split_data(c, report_name, split_data, split_by)
    diffs = {key: val[-1]-val[0] for key, val in split_data.items()}
    if not return_data:
        return diffs
    else:
        data = np.hstack([val for _, val in split_data.items()])
        return data, diffs


def get_transition_matrix(data, bins):
    """Calculates transition matrix from 2 time steps of data (after binning)"""
    assert data.shape[0] == 2, "Transition matrix can only be calculated from 2 (ideally consecutive) time steps"
    bin_edges = np.linspace(0, 1, bins+1) if isinstance(bins, int) else bins
    bin_edges[-1] += 1e-5
    idx = np.digitize(data, bin_edges) - 1
    # work only with values that change
    diff_idx = idx[:, np.where(np.diff(idx, axis=0) != 0)[1]]
    unique_transitions, counts = np.unique(diff_idx, return_counts=True, axis=1)
    transition_matrix = np.zeros((len(bin_edges)-1, len(bin_edges)-1))
    for i in range(unique_transitions.shape[1]):
        transition_matrix[unique_transitions[0, i], unique_transitions[1, i]] = counts[i]
    # normalize with (total) count (in the 1st state) to get probabilities
    unique_states, counts = np.unique(idx[0, :], return_counts=True)
    transition_matrix = np.divide(transition_matrix[unique_states], counts.reshape(-1, 1))
    # fill in missing diagonal (non-changing) values by getting rows sum to 1
    diags = 1 - np.sum(transition_matrix, axis=1)
    for i, diag in enumerate(diags):
        transition_matrix[i, i] = diag
    bin_edges[-1] -= 1e-5
    return transition_matrix, bin_edges


def get_all_synapses_tend(sim_path, report_name):
    """Loads last time step of report, reindexes it, updates it with non-reported data,
    and loads morph. features (for advanced grouping and plotting)"""
    t, data = utils.get_all_synapses_at_t(sim_path, report_name, t=-1)
    # load extra morph. features and add the above data as extra column
    df = utils.load_extra_morph_features(["pre_mtype", "post_mtype", "loc"])
    df[report_name] = data.to_numpy()
    return t, df


def get_all_synapse_diffs(sim_path, report_name):
    """Loads first and last time step of report, takes diff., reindexes it, updates it with non-reported data,
    and loads pre-post gids (for being able to index out assembly neurons)"""
    df = utils.load_extra_morph_features(["pre_gid", "post_gid"])
    diffs = utils.get_all_synapse_changes(sim_path, report_name)
    # df = df.merge(deltas, left_index=True, right_index=True)  # no time for this...
    df[diffs.name] = diffs.to_numpy()
    return df


def get_assembly_change_probs(assembly_grp, df):
    """Calculates depression/potentiation probabilities (of synapses) between assemblies"""
    var = np.setdiff1d(df.columns.to_numpy(), np.array(["pre_gid", "post_gid"]), assume_unique=True)[0]
    dep_probs = np.zeros((len(assembly_grp), len(assembly_grp)), dtype=np.float32)
    pot_probs = np.zeros_like(dep_probs)
    for i, pre_assemly in enumerate(assembly_grp.assemblies):
        df_pre = df.loc[df["pre_gid"].isin(pre_assembly.gids)]
        for j, post_assembly in enumerate(assembly_grp.assemblies):
            df_pre_post = df_pre.loc[df_pre["post_gid"].isin(post_assembly.gids)]
            dep_probs[i, j] = len(df_pre_post.loc[df_pre_post[var] < 0.]) / len(df_pre_post)
            pot_probs[i, j] = len(df_pre_post.loc[df_pre_post[var] > 0.]) / len(df_pre_post)
            del df_pre_post
        del df_pre
    return dep_probs, pot_probs


def get_mean_rho_matrix(df):
    """Gets pathway specific mean rho matrix based on the output of `get_all_synapses_tend()` above"""
    mtypes = np.sort(df["post_mtype"].unique())
    mean_rhos, sum_rhos = np.zeros((len(mtypes), len(mtypes))), np.zeros((len(mtypes), len(mtypes)))
    for i, mtype in enumerate(mtypes):  # could be done in one groupby but whatever...
        df_mtype = df.loc[df["pre_mtype"] == mtype]
        mean_rhos[i, :] = df_mtype.groupby("post_mtype").mean("rho").to_numpy().reshape(-1)
        sum_rhos[i, :] = df_mtype.groupby("post_mtype").sum("rho").to_numpy().reshape(-1)
    mean_rhos[sum_rhos < 5000] = np.nan  # the threshold of at least 5000 synapses is pretty arbitrary
    return mtypes, mean_rhos


def get_td_edge_dists(td_df):
    """Gets L2 norm of differences between edges in consecutive time bins"""
    ts = td_df.columns.get_level_values(0).to_numpy()
    edges = td_df.to_numpy()
    dists = np.linalg.norm(np.diff(edges, axis=1), axis=0)
    return ts[1:], dists


def corr_pw_rate2change(sim, gids, conn_idx, agg_data):
    """Correlate (builds DataFrame with 2 columns...) mean pairwise firing rates
    and the total change of mean (per connection) values"""
    assert agg_data.columns.name == "time", "Aggregated data is not in the expected format (columns should be `time`)"
    t = agg_data.columns.get_level_values(0).to_numpy()
    df = (agg_data.iloc[:, -1] - agg_data.iloc[:, 0]).to_frame("delta")
    del agg_data
    pw_rates = utils.get_gids_pairwise_avg_rates(sim, gids, t[0], t[-1])
    df["pw_rate"] = pw_rates[conn_idx["row"].to_numpy(), conn_idx["col"].to_numpy()]
    return df


def main(project_name):
    sim_paths = utils.load_sim_paths(project_name)
    level_names = sim_paths.index.names
    utils.ensure_dir(os.path.join(FIGS_DIR, project_name))

    for idx, sim_path in tqdm(sim_paths.items()):
        report_name = "gmax_AMPA"
        data, diffs = get_total_change_by(sim_path, report_name, return_data=True)
        fig_name = os.path.join(FIGS_DIR, project_name, "%sgmax_AMPA_delta_pies.png" % utils.midx2str(idx, level_names))
        plots.plot_gmax_change_pie(diffs, fig_name)
        fig_name = os.path.join(FIGS_DIR, project_name, "%sgmax_AMPA_delta_hists.png" % utils.midx2str(idx, level_names))
        plots.plot_gmax_change_hist(data, fig_name)
        fig_name = os.path.join(FIGS_DIR, project_name, "%sgmax_AMPA_hists.png" % utils.midx2str(idx, level_names))
        plots.plot_gmax_dists(data, fig_name)

        report_name = "rho"
        h5f_name = os.path.join(os.path.split(sim_path)[0], "%s.h5" % report_name)
        # general plots about synapses changing in time
        bins, t, hist_data = utils.get_synapse_report_hist(h5f_name)
        hist_data = utils.update_hist_data(report_name, hist_data, bins)
        fig_name = os.path.join(FIGS_DIR, project_name, "%srho_stack.png" % utils.midx2str(idx, level_names))
        plots.plot_rho_stack(bins, t.copy(), hist_data, fig_name, split=False)
        fig_name = os.path.join(FIGS_DIR, project_name, "%srho_stack_split.png" % utils.midx2str(idx, level_names))
        plots.plot_rho_stack(bins, t.copy(), hist_data, fig_name)
        t_idx = int(len(t) / 2)
        _, middle_data = utils.load_synapse_report(h5f_name, t_start=t[t_idx], t_end=t[t_idx + 1])
        transition_matrix, _ = get_transition_matrix(middle_data, bins)
        fig_name = os.path.join(FIGS_DIR, project_name, "%srho_transition.png" % utils.midx2str(idx, level_names))
        plots.plot_transition_matrix(transition_matrix.copy(), bins, fig_name)
        # bit more detailed plots with pre-post mtype pairs at the end of the sim.
        last_t, last_df = get_all_synapses_tend(sim_path, report_name)
        fig_name = os.path.join(FIGS_DIR, project_name, "%srho_hist.png" % utils.midx2str(idx, level_names))
        plots.plot_rho_hist(deepcopy(last_t), last_df, fig_name)
        mtypes, last_rho_matrix = get_mean_rho_matrix(last_df)
        fig_name = os.path.join(FIGS_DIR, project_name, "%srho_matrix.png" % utils.midx2str(idx, level_names))
        plots.plot_mean_rho_matrix(deepcopy(last_t), mtypes, last_rho_matrix, fig_name)
        # plots with edges (aka. aggregated synapses, which have to be computed beforehand)
        gids, conn_idx, agg_data = utils.load_td_edges(sim_path, report_name, "mean")
        t_change, dists = get_td_edge_dists(agg_data)
        fig_name = os.path.join(FIGS_DIR, project_name, "%std_edges_dist.png" % utils.midx2str(idx, level_names))
        plots.plot_agg_edge_dists(t_change.copy(), dists, fig_name)
        rate_change_df = corr_pw_rate2change(Simulation(sim_path), gids, conn_idx, agg_data)
        fig_name = os.path.join(FIGS_DIR, project_name, "%srate_vs_rho.png" % utils.midx2str(idx, level_names))
        plots.plot_rate_vs_change(rate_change_df, fig_name)
        # plot change probs. between assemblies (if they exist)
        h5f_name = os.path.join(os.path.split(os.path.split(sim_path)[0])[0], "assemblies.h5")
        if len(level_names) == 1 and level_names[0] == "seed" and os.path.isfile(h5f_name):
            from assemblyfire.utils import load_assemblies_from_h5
            assembly_grp_dict, _ = load_assemblies_from_h5(h5f_name)
            assembly_grp = assembly_grp_dict["seed%i" % idx]
            df = get_all_synapse_diffs(sim_path, report_name)
            dep_probs, pot_probs = get_assembly_change_probs(assembly_grp, df)
            fig_name = os.path.join(ASSEMBLY_FIGS_DIR, project_name, "cross_assembly_plast_probs_seed%i.png" % idx)
            plots.plot_change_probs(dep_probs, pot_probs, fig_name)


if __name__ == "__main__":
    project_name = "3e3ef5bc-b474-408f-8a28-ea90ac446e24"
    main(project_name)
