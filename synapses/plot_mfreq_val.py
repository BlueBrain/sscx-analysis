"""
Compares minis freqs in vitro vs. (fake) in silico
(just to have the same plot style as for the other syn. related stuff)
last modified: András Ecker 01.2021
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="ticks", context="notebook", font_scale=1.5)
FIGS_PATH = "/gpfs/bbp.cscs.ch/project/proj83/home/ecker/figures"


def _parse_conn_type(pathway):
    """Connection type (eg. Exc->Exc.: 'ee') parser"""
    syn_type = "e" if "Exc" in pathway else "i"
    return "%se" % syn_type


def update_df(df):
    """Adds fake Model_mean value (same as Bio_mean) to df, but no way of getting std..."""
    df["Model_mean"] = np.nan; df["Type"] = ""
    for id_ in df.index:
        pathway = df.loc[id_, "Pathway"]
        df.loc[id_, "Type"] = _parse_conn_type(pathway)
        df.loc[id_, "Model_mean"] = df.loc[id_, "Bio_mean"]
    return df


def plot_validation(df, fig_name):
    """Plots experimental vs. in silico minis freqs"""
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(1, 1, 1)
    sns.despine()

    ax.plot([0,30], [0,30], "--", color="gray", zorder=1)  # diag line
    ax.errorbar(df["Bio_mean"], df["Model_mean"], xerr=df["Bio_std"], color="red", fmt="o", capsize=5, capthick=2, zorder=1)
    df_ee = df[df["Type"] == "ee"]
    ax.plot(df_ee["Bio_mean"], df_ee["Model_mean"], ls="", marker="D", color="magenta", markerfacecoloralt="magenta",
            markersize=12, fillstyle="left", markeredgecolor=(0.3, 0.3, 0.3, 0.5), zorder=2, label="E-E")
    df_ie = df[df["Type"] == "ie"]
    ax.plot(df_ie["Bio_mean"], df_ie["Model_mean"], ls="", marker="D", color="cyan", markerfacecoloralt="magenta",
            markersize=12, fillstyle="left", markeredgecolor=(0.3, 0.3, 0.3, 0.5), zorder=2, label="I-E")

    ax.set_title("Validation of mPSC frequencies")
    ax.set_xlabel("mPSC frequency (Hz)\nIn vitro", color="red")
    ax.set_xlim([0, 30])
    ax.spines["bottom"].set_color("red")
    ax.tick_params(axis="x", colors="red")
    ax.set_ylabel("In silico\nmPSC frequency (Hz)", color="blue")
    ax.set_ylim([0, 30])
    ax.spines["left"].set_color("blue")
    ax.tick_params(axis="y", colors="blue")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], frameon=False)
    fig.savefig(fig_name, dpi=100, bbox_inches="tight", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    f_name = "/gpfs/bbp.cscs.ch/project/proj83/scratch/home/bolanos/circuits/Bio_M/20200805/minis/results.tsv"
    df = pd.read_csv(f_name, sep='\t', skiprows=1, names=["Pathway", "Bio_mean", "Bio_std"], usecols=[0, 1, 2])
    df = update_df(df)
    print(df)
    print("r=%.2f (n=%i)" % (np.corrcoef(df["Bio_mean"], df["Model_mean"])[0, 1], len(df)))
    plot_validation(df, os.path.join(FIGS_PATH, "minis_validation.png"))
