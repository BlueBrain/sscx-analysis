"""
SSCx sims related utility functions
author: András Ecker, last update: 08.2022
"""

import numpy as np
import pandas as pd


def parse_stim_blocks(config):
    """Creates pandas DataFrame from stimulus blocks in a bluepy.Simulation.config object"""
    stim_dict = {key: [] for key in ["pattern", "mode", "t_start", "t_end", "mean", "std", "tau", "amp_cv"]}
    stims = config.typed_sections("Stimulus")
    for stim in stims:
        pattern = stim.Pattern
        if pattern in ["AbsoluteShotNoise", "RelativeShotNoise", "OrnsteinUhlenbeck", "RelativeOrnsteinUhlenbeck"]:
            stim_dict["pattern"].append(pattern)
            stim_dict["mode"].append(stim.Mode)
            t_start = float(stim.Delay)
            stim_dict["t_start"].append(t_start)
            stim_dict["t_end"].append(t_start + float(stim.Duration))
            mean = float(stim.Mean) if pattern in ["AbsoluteShotNoise", "OrnsteinUhlenbeck"] else float(stim.MeanPercent)
            stim_dict["mean"].append(mean)
            std = float(stim.Sigma) if pattern in ["AbsoluteShotNoise", "OrnsteinUhlenbeck"] else float(stim.SDPercent)
            stim_dict["std"].append(std)
            tau = float(stim.DecayTime) if pattern in ["AbsoluteShotNoise", "RelativeShotNoise"] else float(stim.Tau)
            stim_dict["tau"].append(tau)
            stim_dict["amp_cv"] = float(stim.AmpCV) if pattern in ["AbsoluteShotNoise", "RelativeShotNoise"] else np.nan
    return pd.DataFrame.from_dict(stim_dict)


def stim2str(stim):
    pattern = stim["pattern"]
    str = "%s_%s_" % (pattern, stim["mode"])
    str += "Mean%.2f_Std%.2f_Tau%.1f" % (stim["mean"], stim["std"], stim["tau"])
    if pattern in ["AbsoluteShotNoise", "RelativeShotNoise"]:
        str += "_AmpCV%.2f" % stim["amp_cv"]
    return str






