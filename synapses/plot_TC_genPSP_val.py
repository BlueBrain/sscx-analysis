"""
Plots PSP from projections and compare them to ratios from Sermet et al. 2019
author: András Ecker, last update 04.2021
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = "/gpfs/bbp.cscs.ch/project/proj83/home/ecker/psp-validation/out"
BIO_DIR = "/gpfs/bbp.cscs.ch/project/proj83/home/ecker/psp-validation/usecases/sscx"
FIGS_PATH = "/gpfs/bbp.cscs.ch/project/proj83/home/ecker/figures"
sns.set(style="white", context="notebook")


def load_data(proj):
    """Loads in amplitudes from summary files produced by the psp-validation framework"""
    layers = [1, 2, 3, 4, 5, 6]
    types = ["EXC", "PV", "Sst", "5HT3aR"]
    df = pd.DataFrame(index=layers, columns=types, dtype=np.float32)
    df.index.name = "Layer"
    for i, layer in enumerate(layers):
        for j, type in enumerate(types):
            if type != "PV":
                f_name = os.path.join(OUT_DIR, "%s-L%i_%s.summary.yaml" % (proj, layer, type))
                if os.path.isfile(f_name):
                    with open(f_name, "r") as f:
                        data = yaml.load(f, Loader=yaml.SafeLoader)
                    df.iloc[i, j] = data["model"]["mean"]
            else:  # filter out EPSPs above 10 mV
                f_name = os.path.join(OUT_DIR, "%s-L%i_PV.amplitudes.txt" % (proj, layer))
                if os.path.isfile(f_name):
                    data = np.genfromtxt(f_name)
                    df.iloc[i, j] = np.mean(data[data < 10.])
    return df


def plot_psp_ratios(df, df_bio, fig_name):
    """Plots heatmap with PSP values and difference from ref. PSP ratios"""
    norm_layer = 4 if "VPM" in fig_name else 5
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(2, 2, 1)
    sns.heatmap(df, cmap="viridis", annot=True, fmt=".2f", cbar=True, square=False,
                cbar_kws={"label": "mean EPSP (mV) from 50 pairs with 35 repetitions", "orientation": "horizontal"},
                linewidths=.1, ax=ax)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax2 = fig.add_subplot(2, 2, 2)
    df_norm = df / df.loc[norm_layer, "EXC"]  # normalize ratios as Sermet et al. 2019
    sns.heatmap(df_norm, cmap="viridis", annot=True, fmt=".2f", cbar=True, square=False,
                cbar_kws={"label": "EPSP ratios (normed to L%i EXC)" % norm_layer, "orientation": "horizontal"},
                linewidths=.1, ax=ax2)
    ax2.xaxis.tick_top(); ax2.tick_params(length=0)
    ax2.set_ylabel(""); ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
    ax3 = fig.add_subplot(2, 2, 3)
    sns.heatmap(df_bio, cmap="viridis", annot=True, fmt=".2f", cbar=True, square=False,
                cbar_kws={"label": "Sermet et al. 2019 EPSP ratios", "orientation": "horizontal"},
                linewidths=.1, ax=ax3)
    ax3.xaxis.tick_top(); ax3.tick_params(length=0)
    ax3.set_yticklabels(ax3.get_yticklabels(), rotation=0)
    ax4 = fig.add_subplot(2, 2, 4)
    df_diff = df_norm.drop(1, axis=0) - df_bio  # biological ref. doesn't have L1 data
    max_diff = np.max(np.abs(df_diff.to_numpy()))
    sns.heatmap(df_diff, cmap="coolwarm", annot=True, fmt=".2f", cbar=True, square=False,
                cbar_kws={"label": "Difference from Sermet et al. 2019 ratios", "orientation": "horizontal"},
                vmin=-1*max_diff, vmax=max_diff, linewidths=.1, ax=ax4)
    ax4.xaxis.tick_top(); ax4.tick_params(length=0)
    ax4.set_ylabel(""); ax4.set_yticklabels(ax3.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(fig_name, dpi=100, bbox_inches="tight", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    for proj in ["VPM", "POM"]:
        df_bio = pd.read_csv(os.path.join(BIO_DIR, "%s_ratios.csv" % proj), index_col=0)
        df = load_data(proj)
        plot_psp_ratios(df, df_bio, os.path.join(FIGS_PATH, "%s_epsp_ratios.png" % proj))


