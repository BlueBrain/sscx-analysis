"""
Loads synapse clusters saved by `assemblyfire` and checks their total changes in the synapse report
author: András Ecker, last update: 11.2022
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import utils
from plots import plot_2x2_cond_probs, plot_nx2_cond_probs, plot_diffs_stats

pd.set_option('mode.chained_assignment', None)
FIGS_DIR = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/figures/v7_assemblies"
CRITICAL_NSAMPLES = 10  # min sample size from one the 4 categories to start significance test


def _sort_keys(key_list):
    """Sort keys of assembly idx. If -1 is part of the list (standing for non-assembly) then that comes last"""
    if -1 not in key_list:
        return np.sort(key_list)
    else:
        keys = np.array(key_list)
        return np.concatenate((np.sort(keys[keys >= 0]), np.array([-1])))


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


def _prepare2statmodels(df, nsamples, seed=12345):
    """Prepares DataFrame for significance testing in `statmodels` (renaming stuff and balancing dataset)"""
    # get rid of assembly IDs, and replace pre_assembly IDs with just True/False (+rename the column)
    df.loc[df["pre_assembly"] != -1, "assembly"] = True
    df.loc[df["pre_assembly"] == -1, "assembly"] = False
    df.drop("pre_assembly", axis=1, inplace=True)
    # balance dataset
    ncat = df.groupby(["assembly", "clustered"]).size()
    if len(ncat) == 4:
        nmax = ncat.min()  # get the min number of synapses in the 4 categories
        nsamples = nmax if nsamples > nmax else nsamples
        if nsamples > CRITICAL_NSAMPLES:
            dfs = [df.loc[(df["assembly"] == True) & (df["clustered"] == True)].sample(nsamples, random_state=seed)]
            dfs.append(df.loc[(df["assembly"] == True) & (df["clustered"] == False)].sample(nsamples, random_state=seed))
            dfs.append(df.loc[(df["assembly"] == False) & (df["clustered"] == True)].sample(nsamples, random_state=seed))
            dfs.append(df.loc[(df["assembly"] == False) & (df["clustered"] == False)].sample(nsamples, random_state=seed))
            balanced_df = pd.concat(dfs).sort_index()
            # add extra column with single group name for plotting with `seaborn` and Tukey's test
            balanced_df["groups"] = "a-c"
            balanced_df.loc[(balanced_df["assembly"] == True) & (balanced_df["clustered"] == False), "groups"] = "a-nc"
            balanced_df.loc[(balanced_df["assembly"] == False) & (balanced_df["clustered"] == True), "groups"] = "na-c"
            balanced_df.loc[(balanced_df["assembly"] == False) & (balanced_df["clustered"] == False), "groups"] = "na-nc"
        else:
            balanced_df = None
    else:
        balanced_df = None
    if balanced_df is None:
        warnings.warn("Not enough sample (most likely from clustered assembly synapses) for statistical testing!")
    return balanced_df


def _prepare2statannotations(tukey_df):
    """Prepares "pairs" for plotting with `statannotations` (with custom p-values)"""
    tukey_df.loc[tukey_df["reject"] == True]
    pairs, p_vals = [], []
    for _, row in tukey_df.loc[tukey_df["reject"] == True].iterrows():
        pairs.append((row["group1"], row["group2"]))
        p_vals.append(row["p-adj"])
    return pairs, p_vals


def test_significance(df, nsamples=500, sign_th=0.05):
    """Performs 2-way ANOVA and post-hoc Tukey's test using `statmodels`"""
    balanced_df = _prepare2statmodels(df, nsamples=nsamples)
    if balanced_df is not None:
        model = ols('delta_rho ~ C(assembly) + C(clustered) + C(assembly):C(clustered)', data=balanced_df).fit()
        anova_df = sm.stats.anova_lm(model, typ=2)
        if np.nanmin(anova_df["PR(>F)"].to_numpy()) < sign_th:
            tukey = pairwise_tukeyhsd(endog=balanced_df["delta_rho"], groups=balanced_df["groups"], alpha=sign_th)
            tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
            if tukey_df["reject"].any():
                pairs, p_vals = _prepare2statannotations(tukey_df)
            else:
                pairs, p_vals = None, None
        else:
            tukey_df, pairs, p_vals = None, None, None
        return balanced_df, anova_df, tukey_df, pairs, p_vals
    else:
        return None, None, None, None, None


def main(project_name, dir_tag):
    report_name = "rho"
    sim_paths = utils.load_sim_paths(project_name)
    level_names = sim_paths.index.names
    assert len(level_names) == 1 and level_names[0] == "seed"
    figs_dir = os.path.join(FIGS_DIR, project_name + dir_tag)
    utils.ensure_dir(figs_dir)
    # morph_df = utils.load_extra_morph_features(["loc", "dist", "diam", "br_ord"])

    for seed, sim_path in sim_paths.items():
        # probability of any change (given the assembly/non-assembly and clustered/non-clustered conditions)
        _, probs, grouped_diffs, df = utils.get_grouped_syn_diffs(project_name, seed, sim_path, report_name, dir_tag)
        pot_contrasts, dep_contrasts = get_michelson_contrast(probs, grouped_diffs)
        fig_name = os.path.join(figs_dir, "syn_clust_plast_seed%i.png" % seed)
        plot_2x2_cond_probs(probs, pot_contrasts, dep_contrasts, fig_name)
        # checking if the amount of any change is significant
        # df = pd.concat([df, morph_df.loc[df.index]], axis=1)
        pot_df, _, _, pot_pairs, pot_p_vals = test_significance(df.loc[df["delta_rho"] > 0])
        dep_df, _, _, dep_pairs, dep_p_vals = test_significance(df.loc[df["delta_rho"] < 0])
        fig_name = os.path.join(figs_dir, "assembly_diff_stats_seed%i.png" % seed)
        if pot_df is not None and dep_df is not None:
            plot_diffs_stats(pot_df, dep_df, pot_pairs, dep_pairs, pot_p_vals, dep_p_vals, fig_name)

        # same as above, but for cross assemblies
        # probability of any change (given the assembly/non-assembly and clustered/non-clustered conditions)
        fracs, probs, grouped_diffs, df = utils.get_grouped_syn_diffs(project_name, seed, sim_path, report_name,
                                                                      dir_tag, cross_assembly=True)
        pot_contrasts, dep_contrasts = get_michelson_contrast(probs, grouped_diffs)
        for post_assembly, pot_contrast in pot_contrasts.items():
            fig_name = os.path.join(figs_dir, "cross_assembly%i_cond_probs_seed%i.png" % (post_assembly, seed))
            plot_nx2_cond_probs(probs[post_assembly], fracs[post_assembly],
                                pot_contrast, dep_contrasts[post_assembly], post_assembly, fig_name)
        # checking if the amount of any change is significant
        pot_df, _, _, pot_pairs, pot_p_vals = test_significance(df.loc[df["delta_rho"] > 0])
        dep_df, _, _, dep_pairs, dep_p_vals = test_significance(df.loc[df["delta_rho"] < 0])
        fig_name = os.path.join(figs_dir, "cross_assembly_diff_stats_seed%i.png" % seed)
        if pot_df is not None and dep_df is not None:
            plot_diffs_stats(pot_df, dep_df, pot_pairs, dep_pairs, pot_p_vals, dep_p_vals, fig_name)


if __name__ == "__main__":
    dir_tag = ""
    project_name = "LayerWiseOUNoise_Ca1p05_PyramidPatterns"
    main(project_name, dir_tag)
