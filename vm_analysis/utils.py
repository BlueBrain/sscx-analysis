"""
SSCx sims related utility functions
author: András Ecker, last update: 08.2022
"""

import json
import numpy as np
import pandas as pd


def parse_stim_blocks(config):
    """Creates pandas DataFrame from stimulus blocks in a bluepy.Simulation.config object"""
    stim_dict = {key: [] for key in ["pattern", "mode", "t_start", "t_end", "mean", "std", "tau"]}  #, "amp_cv"]}
    injected_stims = [stim_inj.Stimulus for stim_inj in config.typed_sections("StimulusInject")]
    for stim in config.typed_sections("Stimulus"):
        if stim.name in injected_stims:
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
                # stim_dict["amp_cv"] = float(stim.AmpCV) if pattern in ["AbsoluteShotNoise", "RelativeShotNoise"] else -1
    return pd.DataFrame.from_dict(stim_dict)


def repeat_stim_block(stim_dict, nreps):
    """Creates pandas DataFrame with fixed stimulus block values
    (used to create empty stim blocks and then merge with the rest)"""
    return pd.DataFrame.from_dict({key: [val for _ in range(nreps)] for key, val in stim_dict.items()})


def parse_replay_config(jsonf_name):
    """Creates pandas DataFrame from saved JSON config with info about spike replay"""
    with open(jsonf_name, "r") as f:
        data = json.load(f)
    replay_dict = {key: [] for key in ["t_start", "t_end", "pre_frac"]}
    ca, sources, rates, fracs = data["calcium"], data["replay_sources"], data["pre_rate_exc"], data["pre_frac_exc"]
    for i, stim_times in enumerate(data["stim_windows"]):
        replay_dict["t_start"].append(float(stim_times["tmin"]))
        replay_dict["t_end"].append(float(stim_times["tmax"]))
        replay_dict["pre_frac"].append(fracs[i])
    replays = pd.DataFrame.from_dict(replay_dict)
    for source, rate in zip(sources, rates):
        replays["%s_rate" % source] = float(rate)
    replays["ca"] = ca
    return replays


def stim2str(stim):
    pattern = stim["pattern"]
    str = "%s_%s_" % (pattern, stim["mode"])
    str += "Mean%.2f_Std%.2f_Tau%.1f" % (stim["mean"], stim["std"], stim["tau"])
    # if pattern in ["AbsoluteShotNoise", "RelativeShotNoise"]:
    #     str += "_AmpCV%.2f" % stim["amp_cv"]
    return str






