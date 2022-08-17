"""
SSCx analysis related plots
author: András Ecker, last update: 08.2022
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="ticks", context="notebook")


def plot_vm_dist_spect(v, mean, std, spiking, f, pxx, coeffs, freq_window, fig_name):
    """Plots V_m's distribution and power spectrum
    (`mean`, `std`, and `spiking` could be easily computed)"""
    fig = plt.figure(figsize=(10, 6.5))
    ax = fig.add_subplot(1, 2, 1)
    col = "red" if spiking else "blue"
    ax.hist(v[v < -55], bins=30, color=col, label="%.2f+/-%.2f" % (mean, std))  # -55 is a rather arbitrary threshold
    # ax.set_xlim([-80, -50])
    ax.set_xlabel("V_m (mV)")
    ax.legend(frameon=False)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(f, pxx, color="black")
    if not spiking:
        idx = np.where((freq_window[0] < f) & (f < freq_window[1]))[0]
        fit = np.polyval(coeffs, np.log10(f[idx]))
        ax2.plot(f[idx], 10**fit, color="red", label="alpha=%.2f" % np.abs(coeffs[0]))
        ax2.legend(frameon=False)
    plt.xscale("log")
    plt.yscale("log")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("PSD (V^2/Hz)")
    sns.despine()
    fig.tight_layout()
    fig.savefig(fig_name, bbox_inches="tight", dpi=100)
    plt.close(fig)




