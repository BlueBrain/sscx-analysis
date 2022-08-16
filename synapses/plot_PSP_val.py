"""
Plots results from PSP validation (experimental vs. model PSP amplitudes)
author: András Ecker, last update 05.2020
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="ticks", context="notebook", font_scale=1.5)
FIGS_PATH = "/gpfs/bbp.cscs.ch/project/proj83/home/ecker/figures"


def load_data(pathways, out_path):
    """Loads in amlitudes from summary files produced by the psp-validation framework"""
    psps = {"Pathway": [], "Bio_mean": [], "Bio_std": [], "Model_mean": [], "Model_std": [], "Type": []}
    for phway in pathways:
        psps["Pathway"].append(phway[0])
        psps["Type"].append(phway[1])
        f_name = os.path.join(out_path, "%s.summary.yaml" % phway[0])
        with open(f_name, "r") as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
            psps["Bio_mean"].append(data["reference"]["mean"]); psps["Bio_std"].append(data["reference"]["std"])
            psps["Model_mean"].append(data["model"]["mean"]); psps["Model_std"].append(data["model"]["std"])
    df = pd.DataFrame(psps, columns=["Pathway", "Bio_mean", "Bio_std", "Model_mean", "Model_std", "Type"])
    df["Bio_std"].replace("None", 0.0, inplace=True)
    return df


def df2tex(df):
    """Converts df to table (rows) used in Latex"""
    for _, row in df.iterrows():
        phway = row["Pathway"].split('-')
        print(r"%s & %s & %.2f$\pm$%.2f & %.2f$\pm$%.2f & \cite{} \\" % (phway[0], phway[1],
              row["Bio_mean"], row["Bio_std"], row["Model_mean"], row["Model_std"]))


def plot_validation(df, fig_name):
    """Plots experimental vs. in silico PSPs"""
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(1, 1, 1)
    sns.despine()
    ax.plot([0, 5], [0, 5], "--", color="gray", zorder=1)  # diag line
    ax.errorbar(df["Bio_mean"], df["Model_mean"], yerr=df["Model_std"], color="blue", fmt="o", capsize=5, capthick=2, zorder=1)
    ax.errorbar(df["Bio_mean"], df["Model_mean"], xerr=df["Bio_std"], color="red", fmt="o", capsize=5, capthick=2, zorder=1)

    df_ee = df[df["Type"] == "ee"]
    ax.plot(df_ee["Bio_mean"], df_ee["Model_mean"], ls="", marker="D", color="magenta", markerfacecoloralt="magenta",
            markersize=12, fillstyle="left", markeredgecolor=(0.3, 0.3, 0.3, 0.5), zorder=2, label="E-E")
    df_ei = df[df["Type"] == "ei"]
    ax.plot(df_ei["Bio_mean"], df_ei["Model_mean"], ls="", marker="D", color="magenta", markerfacecoloralt="cyan",
            markersize=12, fillstyle="left", markeredgecolor=(0.3, 0.3, 0.3, 0.5), zorder=2, label="E-I")
    df_ie = df[df["Type"] == "ie"]
    ax.plot(df_ie["Bio_mean"], df_ie["Model_mean"], ls="", marker="D", color="cyan", markerfacecoloralt="magenta",
            markersize=12, fillstyle="left", markeredgecolor=(0.3, 0.3, 0.3, 0.5), zorder=2, label="I-E")
    #df_ii = df[df["Type"] == "ii"]
    #ax.plot(df_ii["Bio_mean"], df_ii["Model_mean"], ls="", marker="D", color="cyan", markerfacecoloralt="cyan",
    #        markersize=12, fillstyle="left", markeredgecolor=(0.3, 0.3, 0.3, 0.5), zorder=2, label="I-I")
    df_te = df[df["Type"] == "te"]
    ax.plot(df_te["Bio_mean"], df_te["Model_mean"], ls="", marker="D", color="yellow", markerfacecoloralt="magenta",
            markersize=12, fillstyle="left", markeredgecolor=(0.3, 0.3, 0.3, 0.5), zorder=2, label="T-E")
    df_ti = df[df["Type"] == "ti"]
    ax.plot(df_ti["Bio_mean"], df_ti["Model_mean"], ls="", marker="D", color="yellow", markerfacecoloralt="cyan",
            markersize=12, fillstyle="left", markeredgecolor=(0.3, 0.3, 0.3, 0.5), zorder=2, label="T-I")
    ax.set_title("Validation of PSP amplitudes")
    ax.set_xlabel("PSP amplitude (mV)\nIn vitro", color="red")
    ax.set_xlim([0, 3])  # 4.5])
    ax.spines["bottom"].set_color("red")
    ax.tick_params(axis="x", colors="red")
    ax.set_ylabel("In silico\nPSP amplitude (mV)", color="blue")
    ax.set_ylim([0, 3])  # 4.5])
    ax.spines["left"].set_color("blue")
    ax.tick_params(axis="y", colors="blue")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:5], labels[:5], frameon=False)
    ax.text(2.6, 0.3, "r = %.2f\n(n = %i)" % (np.corrcoef(df["Bio_mean"], df["Model_mean"])[0, 1], len(df)),
            fontsize="small")
    fig.savefig(fig_name, dpi=100, bbox_inches="tight", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    out_path = "/gpfs/bbp.cscs.ch/project/proj83/home/ecker/psp-validation/out"
    pathways = [("L1_NGC-L23_PC", "ie"), ("L1_nonNGC-L23_PC", "ie"),
                ("L4_EXC-L4_EXC", "ee"), ("L4_EXC-L4_FS", "ei"), ("L4_FS-L4_EXC", "ie"),
                ("L4_SS-L23_PC", "ee"), ("L4_SS-L5A_PC", "ee"), ("L4_SS-L6_PC", "ee"),
                ("L5_MC-L5_TTPC", "ie"), ("L5_TTPC-L5_MC", "ei"), ("L5_TTPC-L5_TTPC", "ee"), ("L5A_PC-L5A_PC", "ee"),
                ("L6_BPC-L6_TPC", "ee"), ("L6_IPC-L6_BC", "ei"), ("L6_IPC-L6_BPC", "ee"), ("L6_IPC-L6_IPC", "ee"),
                ("L6_NPC-L6_BC", "ei"), ("L6_NPC-L6_IPC", "ee"), ("L6_NPC-L6_NPC", "ee"),
                ("L6_PC-L6_MC", "ei"), ("L6_TPC-L6_BC", "ei"), ("L6_TPC-L6_BPC", "ee"),
                ("L6_TPC-L6_NPC", "ee"), ("L6_TPC-L6_TPC", "ee"), ("L23_PC-L1_nonNGC", "ei"),
                ("L23_PC-L23_PC", "ee"), ("L23_PC-L5_TTPC", "ee"), ("TC-L4_EXC", "te"), ("TC-L4_FS", "ti")]
                # ("L5_TTPC-L5_FS", "ei"), ("L234_PC-L234_NBC", "ei")  # not calibrated
    df = load_data(pathways, out_path)
    print(df)
    print("r=%.2f (n=%i)" % (np.corrcoef(df["Bio_mean"], df["Model_mean"])[0, 1], len(df)))
    plot_validation(df, os.path.join(FIGS_PATH, "PSP_validation.png"))
    # df2tex(df)
