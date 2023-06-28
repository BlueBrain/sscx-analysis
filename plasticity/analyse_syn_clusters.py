"""
Loads synapse clusters saved by `assemblyfire` and checks their total changes in the synapse report
author: András Ecker, last update: 06.2023
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


def get_michelson_contrast(df):
    """Gets Michelson contrast (aka. visibility) defined as:
    P(pot/dep | condition) - P(pot/dep) / (P(pot/dep | condition) + P(pot/dep))"""

    probs, pot_contrasts, dep_contrasts = {}, {}, {}
    for assembly in df["post_assembly"].unique():
        probs[assembly] = {}
        df_assembly = df.loc[df["post_assembly"] == assembly]
        p_pot = len(df_assembly.loc[df_assembly["delta_rho"] > 0]) / len(df_assembly)
        p_dep = len(df_assembly.loc[df_assembly["delta_rho"] < 0]) / len(df_assembly)
        probs[assembly]["pot"], probs[assembly]["dep"] = p_pot, p_dep

        pre_assemblies = utils.sort_assembly_keys(df_assembly["pre_assembly"].unique())
        pot_contrast, dep_contrast = np.zeros((len(pre_assemblies), 2)), np.zeros((len(pre_assemblies), 2))
        for i, pre_assembly in enumerate(pre_assemblies):
            df_tmp = df_assembly.loc[df_assembly["pre_assembly"] == pre_assembly]
            for j, clustered in enumerate([1, -1]):
                diffs = df_tmp.loc[df_tmp["clustered"] == clustered, "delta_rho"]
                n_syns = len(diffs)
                if n_syns:
                    p_pot_cond = len(diffs[diffs > 0]) / n_syns
                    pot_contrast[i, j] = (p_pot_cond - p_pot) / (p_pot_cond + p_pot)
                    p_dep_cond = len(diffs[diffs < 0]) / n_syns
                    dep_contrast[i, j] = (p_dep_cond - p_dep) / (p_dep_cond + p_dep)
                else:
                    pot_contrast[i, j], dep_contrast[i, j] = np.nan, np.nan
        del df_assembly
        pot_contrasts[assembly], dep_contrasts[assembly] = pot_contrast, dep_contrast

    return probs, pot_contrasts, dep_contrasts


def _prepare2statmodels(df, nsamples, seed=12345):
    """Prepares DataFrame for significance testing in `statmodels` (renaming stuff and balancing dataset)"""
    # keep only boolean assembly & clustered
    df["assembly"] = False
    df.loc[df["pre_assembly"] != -1, "assembly"] = True
    df.replace({"clustered": {1: True, -1: False}}, inplace=True)
    df = df[["assembly", "clustered", "delta_rho"]]
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
    pairs, p_vals = [], []
    for _, row in tukey_df.loc[tukey_df["reject"] == True].iterrows():
        pairs.append((row["group1"], row["group2"]))
        p_vals.append(row["p-adj"])
    return pairs, p_vals


def test_significance(df, nsamples=1000, sign_th=0.05):
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

    for seed, sim_path in sim_paths.items():
        # probability of any change (given the assembly/non-assembly and clustered/non-clustered conditions)
        df = utils.get_assembly_clustered_syn_diffs(project_name, seed, sim_path, report_name, dir_tag)
        probs, pot_contrasts, dep_contrasts = get_michelson_contrast(df)
        fig_name = os.path.join(figs_dir, "syn_clust_plast_seed%i.png" % seed)
        plot_2x2_cond_probs(probs, pot_contrasts, dep_contrasts, fig_name)
        # checking if the amount of any change is significant
        pot_df, _, _, pot_pairs, pot_p_vals = test_significance(df.loc[df["delta_rho"] > 0])
        dep_df, _, _, dep_pairs, dep_p_vals = test_significance(df.loc[df["delta_rho"] < 0])
        fig_name = os.path.join(figs_dir, "assembly_diff_stats_seed%i.png" % seed)
        if pot_df is not None and dep_df is not None:
            plot_diffs_stats(pot_df, dep_df, pot_pairs, dep_pairs, pot_p_vals, dep_p_vals, fig_name)

        # same as above, but for cross assemblies
        # probability of any change (given the assembly/non-assembly and clustered/non-clustered conditions)
        df = utils.get_assembly_clustered_syn_diffs(project_name, seed, sim_path, report_name,
                                                    dir_tag, cross_assembly=True)
        probs, pot_contrasts, dep_contrasts = get_michelson_contrast(df)
        for assembly, pot_contrast in pot_contrasts.items():
            fracs = df.loc[df["post_assembly"] == assembly, "pre_assembly"].value_counts(normalize=True).to_dict()
            fig_name = os.path.join(figs_dir, "cross_assembly%i_cond_probs_seed%i.png" % (assembly, seed))
            plot_nx2_cond_probs(probs[assembly], fracs, pot_contrast, dep_contrasts[assembly], assembly, fig_name)
        # checking if the amount of any change is significant
        pot_df, _, _, pot_pairs, pot_p_vals = test_significance(df.loc[df["delta_rho"] > 0])
        dep_df, _, _, dep_pairs, dep_p_vals = test_significance(df.loc[df["delta_rho"] < 0])
        fig_name = os.path.join(figs_dir, "cross_assembly_diff_stats_seed%i.png" % seed)
        if pot_df is not None and dep_df is not None:
            plot_diffs_stats(pot_df, dep_df, pot_pairs, dep_pairs, pot_p_vals, dep_p_vals, fig_name)


if __name__ == "__main__":
    dir_tag = ""
    project_name = "3e3ef5bc-b474-408f-8a28-ea90ac446e24"
    main(project_name, dir_tag)
