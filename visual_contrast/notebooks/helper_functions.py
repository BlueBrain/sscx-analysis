# Helper functions for grating and opto stimulus analysis
# Author: C. Pokorny
# Last modified: 01/2022

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bluepy import Cell, Synapse, Circuit, Simulation
import os
import json
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


### MOVED TO psth_peak_stats.py ###
# def detect_rate_peaks(t_rate, rates, peak_th=5.0, peak_width=20.0, peak_distance=200.0, t_range=None):
#     """
#     Peak detection (first & second peak) of firing rates (<#GIDs x #time_steps>),
#     using peak_th (Hz), peak_width (ms), peak_distance (ms), within given time
#     range and selecting the two highest peaks
#     """
#     t_res = np.median(np.diff(t_rate))
#     peak_idx = [find_peaks(r, height=peak_th, width=peak_width / t_res, distance=peak_distance / t_res)[0] for r in rates]

#     # Remove out-of-range peaks
#     if t_range is not None:
#         for idx, pidx in enumerate(peak_idx):
#             peak_idx[idx] = pidx[np.logical_and(t_rate[pidx] >= t_range[0], t_rate[pidx] < t_range[1])]

#     # In case of more than two (remaining) peaks: keep only two highest
#     for idx, pidx in enumerate(peak_idx):
#         r = rates[idx][pidx]
#         sel = np.argsort(r)[:-3:-1] # Find two highest
#         peak_idx[idx] = pidx[np.sort(sel)] # Select two highest (keeping order)

#     peak_rate = [rates[idx][pidx] for idx, pidx in enumerate(peak_idx)]
#     peak_t = [t_rate[pidx] for pidx in peak_idx]

#     t1 = [t[0] if len(t) > 0 else np.nan for t in peak_t] # First peak time
#     t2 = [t[1] if len(t) > 1 else np.nan for t in peak_t] # Second peak time
#     r1 = [r[0] if len(r) > 0 else np.nan for r in peak_rate] # First peak rate
#     r2 = [r[1] if len(r) > 1 else np.nan for r in peak_rate] # Second peak rate

#     peak_ratio = np.array([(_r1 - _r2) / (_r1 + _r2) for (_r1, _r2) in zip(r1, r2)])

#     return peak_idx, t1, t2, r1, r2, peak_ratio


### ALTERNATIVE IMPLEMENTATION ###
# def detect_rate_peaks(t_rate, rates, peak_th=None):
#     """
#     Peak detection (first & second peak) of firing rates (<#GIDs x #time_steps>),
#     separated by population rate minima, above specified peak_th (Hz)
#     """
#     # Find population rate minima, to separate first from second peak interval
#     min_idx = find_peaks(-np.mean(rates, 0))[0][:2]
#     assert len(min_idx) == 2, 'ERROR: Population minima could not be determined!'

#     # Find first and second peak between population minima
#     peak_idx = []
#     for r in rates:
#         pidx1 = np.argmax(r[:min_idx[0]])
#         if peak_th is not None and r[pidx1] <= peak_th:
#             pidx1 = -1
#         pidx2 = min_idx[0] + np.argmax(r[min_idx[0]:min_idx[1]])
#         if peak_th is not None and r[pidx2] <= peak_th:
#             pidx2 = -1
#         peak_idx.append(np.array([pidx1, pidx2]))

#     peak_rate = [np.array([rates[idx][p] if p >= 0 else np.nan for p in pidx]) for idx, pidx in enumerate(peak_idx)]
#     peak_t = [np.array([t_rate[p] if p >= 0 else np.nan for p in pidx]) for pidx in peak_idx]

#     t1 = [t[0] for t in peak_t] # First peak time
#     t2 = [t[1] for t in peak_t] # Second peak time
#     r1 = [r[0] for r in peak_rate] # First peak rate
#     r2 = [r[1] for r in peak_rate] # Second peak rate

#     peak_ratio = np.array([(_r1 - _r2) / (_r1 + _r2) for (_r1, _r2) in zip(r1, r2)])

#     return peak_idx, t1, t2, r1, r2, peak_ratio


### MOVED TO single_cell_psths.py ###
# def get_single_cell_psths(blue_config, target_spec, psth_interval=None, t_res=20.0, t_smooth=None):
#     """Extract single-cell spikes per pattern, re-aligned to stimulus onsets"""

#     # Load simulation, circuit, and stim/opto configs
#     sim = Simulation(blue_config)
#     c = sim.circuit

#     stim_file = os.path.abspath(os.path.join(os.path.split(blue_config)[0], sim.config['Stimulus_spikeReplay']['SpikeFile']))
#     stim_file = os.path.splitext(stim_file)[0] + '.json'
#     assert os.path.exists(stim_file), 'ERROR: Stim config file not found!'
#     with open(stim_file, 'r') as f:
#         stim_cfg = json.load(f)

#     opto_file = os.path.join(os.path.split(blue_config)[0], 'opto_stim.json')
#     if os.path.exists(opto_file):
#         with open(opto_file, 'r') as f:
#             opto_cfg = json.load(f)
#     else:
#         # print('INFO: No opto config found!')
#         opto_cfg = None

#     # Select cells and spikes
#     if isinstance(target_spec, dict):
#         target_spec = target_spec.copy()
#         cell_target = target_spec.pop('target', None)
#         gids = c.cells.ids(target_spec)
#         if cell_target is not None:
#             gids = np.intersect1d(gids, c.cells.ids(cell_target))
#     else:
#         gids = c.cells.ids(target_spec)

#     spikes = sim.spikes.get(gids)

#     # Find overlapping opto stimuli
#     time_windows = stim_cfg['props']['time_windows']
#     stim_train = stim_cfg['props']['stim_train']
#     num_patterns = max(stim_train) + 1
#     stim_int = [[t, t + stim_cfg['cfg']['duration_stim']] for t in np.array(time_windows[:-1])]
#     if opto_cfg is None:
#         stim_opto_overlap = np.zeros(len(stim_int))
#     else:
#         opto_int = [[opto_cfg['props']['opto_t'][i], opto_cfg['props']['opto_t'][i] + opto_cfg['props']['opto_dur'][i]] for i in range(len(opto_cfg['props']['opto_t']))]
#         stim_opto_overlap = np.array([np.max([(max(0, min(max(stim_int[i]), max(oint)) - max(min(stim_int[i]), min(oint)))) / np.diff(stim_int[i]) for oint in opto_int]) for i in range(len(stim_int))])
#     opto_stim_train = (stim_opto_overlap > 0.0) * num_patterns + stim_train # Reindex stimuli (0..N-1: stims w/o opto, N..2N-1: stims with opto)

#     # Extract single-cell PSTHs
#     if psth_interval is None:
#         psth_interval = [0, np.max(np.diff(time_windows))]
#     spike_trains = [{gid: [] for gid in gids} for p in range(num_patterns * 2)] # Spikes per pattern and gid, re-aligned to stimulus onset
#     for sidx, pidx in enumerate(opto_stim_train):
#         spk = spikes[np.logical_and(spikes.index - time_windows[sidx] >= psth_interval[0],
#                                     spikes.index - time_windows[sidx] < psth_interval[-1])]
#         spk.index = spk.index - time_windows[sidx] # Re-align to stimulus onset
#         for gid in gids:
#             spike_trains[pidx][gid].append(spk[spk.values == gid].index.to_list())

#     # Single-cell average spike rates over trials and PSTH interval
#     avg_cell_rates = [[1e3 * np.mean([len(st) for st in spike_trains[p][gid]]) / np.diff(psth_interval)[0] if len(spike_trains[p][gid]) > 0 else np.nan for gid in gids] for p in range(num_patterns * 2)]

#     # Instantaneous spike rates averaged over trials
#     rates = []
#     for p in range(num_patterns * 2):
#         pattern_rates = []
#         for gid in gids:
#             t_rate, rate = instant_firing_rate(spike_trains[p][gid], t_res, t_min=psth_interval[0], t_max=psth_interval[-1], t_smooth=t_smooth)
#             pattern_rates.append(rate)
#         rates.append(np.array(pattern_rates))

#     return t_rate, rates, spike_trains, avg_cell_rates, gids, stim_cfg, opto_cfg


### MOVED TO single_cell_psths.py ###
# def instant_firing_rate(spikes, res_ms, t_min=None, t_max=None, t_smooth=None):
#     """ Estimation of instantaneous firing rates of given spike trains
#         over multiple trials (list of lists of spike times), optionally
#         using Gaussian smoothing
#     """
#     assert res_ms > 0.0, 'ERROR: Resolution must be larger than zero ms!'
#     num_trials = len(spikes)

#     if t_min is None:
#         t_min = np.min(spikes)
#     if t_max is None:
#         t_max = np.max(spikes)

#     bins = np.linspace(t_min, t_max, np.round((t_max - t_min) / res_ms).astype(int) + 1)
#     hist_count = np.zeros(len(bins) - 1)
#     for trial in range(num_trials):
#         hist_count += np.histogram(spikes[trial], bins=bins)[0]
#     if num_trials == 0:
#         rate = np.full_like(hist_count, np.nan)
#     else:
#         rate = hist_count / (num_trials * res_ms * 1e-3)
#     t = bins[:-1] + 0.5 * res_ms

#     if t_smooth is not None: # Allpy Gaussian smoothing
#         assert t_smooth > 0.0, 'ERROR: Smoothing time constant must be larger than zero ms!'
#         rate = gaussian_filter1d(rate, sigma=t_smooth / res_ms)

#     return t, rate


def load_sim_results(sims):
    """Load simulation results into dataframes"""

    blank_rates_table = sims.to_frame()
    blank_rates_table.drop(columns=blank_rates_table.keys(), inplace=True)
    stim_rates_table = sims.to_frame()
    stim_rates_table.drop(columns=stim_rates_table.keys(), inplace=True)
    opto_rates_table = sims.to_frame()
    opto_rates_table.drop(columns=opto_rates_table.keys(), inplace=True)
    blue_configs = []
    stim_configs = []
    opto_configs = []
    for sidx, blue_config in enumerate(sims):
        activity_blank, activity_stim, activity_opto, stim_cfg, opto_cfg = get_activity(blue_config)
        blue_configs.append(blue_config)
        stim_configs.append(stim_cfg)
        opto_configs.append(opto_cfg)
        num_stim = len(activity_stim)
        for k in activity_blank['rates'].keys():
            if k not in blank_rates_table.keys():
                blank_rates_table[k] = np.full(blank_rates_table.shape[0], np.nan) # Column init
                for s in range(num_stim):
                    if num_stim == 1:
                        k_stim = k
                    else:
                        k_stim = k + f'_STIM{s}'
                    stim_rates_table[k_stim] = np.full(stim_rates_table.shape[0], np.nan) # Column init
                    opto_rates_table[k_stim] = np.full(opto_rates_table.shape[0], np.nan) # Column init
            blank_rates_table.iloc[sidx, np.where(blank_rates_table.columns == k)[0]] = activity_blank['rates'][k]
            for s in range(num_stim):
                if num_stim == 1:
                    k_stim = k
                else:
                    k_stim = k + f'_STIM{s}'
                stim_rates_table.iloc[sidx, np.where(stim_rates_table.columns == k_stim)[0]] = activity_stim[s]['rates'][k]
                opto_rates_table.iloc[sidx, np.where(opto_rates_table.columns == k_stim)[0]] = activity_opto[s]['rates'][k]

    return blank_rates_table, stim_rates_table, opto_rates_table, blue_configs, stim_configs, opto_configs


def get_activity(blue_config):
    """ Returns baseline (blank), stim and opto-stim activity for opto targets of given simulation"""

    # Load simulation, circuit, and opto/stim configs
    sim = Simulation(blue_config)
    c = sim.circuit

    opto_file = os.path.join(os.path.split(blue_config)[0], 'opto_stim.json')
    assert os.path.exists(opto_file), 'ERROR: Opto config file not found!'
    with open(opto_file, 'r') as f:
        opto_cfg = json.load(f)

    stim_file = os.path.abspath(os.path.join(os.path.split(blue_config)[0], sim.config['Stimulus_spikeReplay']['SpikeFile']))
    stim_file = os.path.splitext(stim_file)[0] + '.json'
    assert os.path.exists(stim_file), 'ERROR: Stim config file not found!'
    with open(stim_file, 'r') as f:
        stim_cfg = json.load(f)

    # Determine stimulus, blank, and opto intervals
    num_patterns = max(stim_cfg['props']['stim_train']) + 1
    blank_int = [[t + stim_cfg['cfg']['duration_stim'], t + stim_cfg['cfg']['duration_stim'] + stim_cfg['cfg']['duration_blank']] for t in stim_cfg['props']['time_windows'][:-1]]
    stim_int = [[[t, t + stim_cfg['cfg']['duration_stim']] for t in np.array(stim_cfg['props']['time_windows'][:-1])[np.array(stim_cfg['props']['stim_train']) == stim]] for stim in range(num_patterns)]
    opto_int = [[opto_cfg['props']['opto_t'][i], opto_cfg['props']['opto_t'][i] + opto_cfg['props']['opto_dur'][i]] for i in range(len(opto_cfg['props']['opto_t']))]

    # Determine rel. stimulus overlap with optogenetic stimulation
    stim_opto_overlap = [np.array([np.max([(max(0, min(max(stim_int[stim][i]), max(oint)) - max(min(stim_int[stim][i]), min(oint)))) / np.diff(stim_int[stim][i]) for oint in opto_int]) for i in range(len(stim_int[stim]))]) for stim in range(num_patterns)]
    blank_opto_overlap = np.array([np.max([(max(0, min(max(blank_int[i]), max(oint)) - max(min(blank_int[i]), min(oint)))) / np.diff(blank_int[i]) for oint in opto_int]) for i in range(len(blank_int))])

    # Select (non-)overlapping intervals
    baseline_sel = [blank_int[i] for i in np.where(blank_opto_overlap == 0.0)[0]] # Blank intervals w/o opto stim (0% overlap)
    stim_sel = [[stim_int[stim][i] for i in np.where(stim_opto_overlap[stim] == 0.0)[0]] for stim in range(num_patterns)] # Stim intervals w/o opto stim (0% overlap)
    opto_sel = [[stim_int[stim][i] for i in np.where(stim_opto_overlap[stim] == 1.0)[0]] for stim in range(num_patterns)] # Stim intervals with opto stim (100% overlap)
    baseline_count = len(baseline_sel)
    stim_count = [len(stim_sel[stim]) for stim in range(num_patterns)]
    opto_count = [len(opto_sel[stim]) for stim in range(num_patterns)]

    # Population activity per opto target
    if 'layer' in opto_cfg['cfg']['opto_target'].keys():
        layer_dicts = []
    else: # Add per-layer targets
        num_layers = 6
        layer_dicts = [opto_cfg['cfg']['opto_target'].copy() for i in range(num_layers)]
        for i in range(num_layers):
            layer_dicts[i].update({'layer': i + 1})
    targets = [opto_cfg['cfg']['opto_target']] + layer_dicts + opto_cfg['props']['inject_target_names']
    blank_dict = {'count': baseline_count, 'rates': {}}
    stim_dict = [{'count': stim_count[stim], 'rates': {}} for stim in range(num_patterns)]
    opto_dict = [{'count': opto_count[stim], 'rates': {}} for stim in range(num_patterns)]
    for opto_tgt in targets:
        if isinstance(opto_tgt, dict):
            target_spec = opto_tgt.copy()
            cell_target = target_spec.pop('target', None)
            tgt_gids = c.cells.ids(target_spec)
            if cell_target is not None:
                tgt_gids = np.intersect1d(tgt_gids, c.cells.ids(cell_target))
            if 'layer' in opto_tgt:
                opto_tgt = f'FullTargetL{opto_tgt["layer"]}'
            else:
                opto_tgt = 'FullTarget'
        else:
            tgt_gids = c.cells.ids(opto_tgt)

        # Baseline activity (w/o opto)
        if baseline_count == 0:
            blank_dict['rates'][opto_tgt] = np.nan
        else:
            baseline_spikes = pd.concat([sim.spikes.get(tgt_gids, t_start=t_sel[0], t_end=t_sel[-1]) for t_sel in baseline_sel])
            blank_dict['rates'][opto_tgt] = baseline_spikes.count() / (1e-3 * np.sum([np.diff(t_sel) for t_sel in baseline_sel]) * len(tgt_gids)) # Avg. firing rate (Hz) of a target neuron

        for stim in range(num_patterns):
            # Stimulus activity (w/o opto)
            if stim_count[stim] == 0:
                stim_dict[stim]['rates'][opto_tgt] = np.nan
            else:
                stim_spikes = pd.concat([sim.spikes.get(tgt_gids, t_start=t_sel[0], t_end=t_sel[-1]) for t_sel in stim_sel[stim]])
                stim_dict[stim]['rates'][opto_tgt] = stim_spikes.count() / (1e-3 * np.sum([np.diff(t_sel) for t_sel in stim_sel[stim]]) * len(tgt_gids)) # Avg. firing rate (Hz) of a target neuron

            # Stimulus + opto activity
            if opto_count[stim] == 0:
                opto_dict[stim]['rates'][opto_tgt] = np.nan
            else:
                opto_spikes = pd.concat([sim.spikes.get(tgt_gids, t_start=t_sel[0], t_end=t_sel[-1]) for t_sel in opto_sel[stim]])
                opto_dict[stim]['rates'][opto_tgt] = opto_spikes.count() / (1e-3 * np.sum([np.diff(t_sel) for t_sel in opto_int]) * len(tgt_gids)) # Avg. firing rate (Hz) of a target neuron

    return blank_dict, stim_dict, opto_dict, stim_cfg, opto_cfg


# Population response PSTHs (per layer and Exc/INH)
def plot_PSTH(blue_config, psth_bin_size=10, psth_interval=None, psth_target=None, layers=None, syn_classes=None, label=None, match_scale=True, save_fn=None):
    sim = Simulation(blue_config)
    c = sim.circuit
    if layers is None:
        layers = np.unique(c.cells.get(properties=Cell.LAYER))
    else:
        assert isinstance(layers, list) and np.all(np.isin(layers, np.unique(c.cells.get(properties=Cell.LAYER)))), 'ERROR: Layer selection error!'
    layer_colors = plt.cm.jet(np.linspace(0, 1, len(layers)))
    if syn_classes is None:
        syn_classes = np.unique(c.cells.get(properties=Cell.SYNAPSE_CLASS))
    else:
        assert isinstance(syn_classes, list) and np.all(np.isin(syn_classes, np.unique(c.cells.get(properties=Cell.SYNAPSE_CLASS)))), 'ERROR: Synapse class selection error!'
    syn_ls = ['-', '--'] # Linestyles

    stim_spike_file = os.path.abspath(os.path.join(os.path.split(blue_config)[0], sim.config['Stimulus_spikeReplay']['SpikeFile']))
    stim_cfg_file = os.path.splitext(stim_spike_file)[0] + '.json'
    with open(stim_cfg_file, 'r') as f:
        stim_cfg = json.load(f)

    num_patterns = max(stim_cfg['props']['stim_train']) + 1
    stim_train = stim_cfg['props']['stim_train']
    time_windows = stim_cfg['props']['time_windows']

    assert psth_bin_size > 0, 'ERROR: PSTH bin size (ms) must be larger than 0 ms!'
    if psth_interval is None:
        psth_interval = [0, np.max(np.diff(time_windows))]

    t_len = np.diff(psth_interval)[0]
    num_bins = np.round(t_len / psth_bin_size).astype(int)
    bins = np.arange(num_bins + 1) * psth_bin_size + psth_interval[0]

    fig, ax = plt.subplots(1, num_patterns, figsize=(3 * 3.5 * num_patterns, 3.5), dpi=300)
    if not hasattr(ax, '__iter__'):
        ax = [ax]
    ax[0].set_ylabel(f'Firing rates (Hz)')
    mean_PSTHs = np.zeros((len(syn_classes), num_patterns, num_bins))
    for lidx, lay in enumerate(layers):
        for sclidx, sclass in enumerate(syn_classes):
            gids_sel = np.intersect1d(sim.target_gids, c.cells.ids({Cell.LAYER: lay, Cell.SYNAPSE_CLASS: sclass}))
            if psth_target is not None:
                gids_sel = np.intersect1d(gids_sel, c.cells.ids(psth_target))
            stim_spikes = sim.spikes.get(gids=gids_sel)
            num_cells = len(gids_sel)

            pattern_spikes = [[]] * num_patterns
            for sidx, pidx in enumerate(stim_train):
                spikes = stim_spikes[np.logical_and(stim_spikes.index - time_windows[sidx] >= psth_interval[0],
                                                    stim_spikes.index - time_windows[sidx] < psth_interval[-1])]
                spikes.index = spikes.index - time_windows[sidx] # Re-align to stimulus onset
                pattern_spikes[pidx] = pattern_spikes[pidx] + [spikes]

            num_stim_per_pattern = [len(p) for p in pattern_spikes]
            pattern_spikes = [pd.concat(p) for p in pattern_spikes]

            for pidx in range(num_patterns):
                if num_cells > 0:
                    pattern_PSTH = 1e3 * np.histogram(pattern_spikes[pidx].index, bins=bins)[0] / (psth_bin_size * num_cells * num_stim_per_pattern[pidx])
                    mean_PSTHs[sclidx, pidx, :] = mean_PSTHs[sclidx, pidx, :] + pattern_PSTH / len(layers)
                    ax[pidx].plot(bins[:-1] + 0.5 * psth_bin_size, pattern_PSTH, color=layer_colors[lidx], linestyle=syn_ls[sclidx], label=f'L{lay}-{sclass}' if pidx == num_patterns - 1 else None)
                ax[pidx].set_xlim((min(bins), max(bins)))
                ax[pidx].set_xlabel('Time (ms)')
                ax[pidx].set_title(f'Pattern {pidx} (N={num_stim_per_pattern[pidx]})', fontweight='bold')
    for sclidx, sclass in enumerate(syn_classes):
        for pidx in range(num_patterns):
            ax[pidx].plot(bins[:-1] + 0.5 * psth_bin_size, mean_PSTHs[sclidx, pidx, :], color='k', linestyle=syn_ls[sclidx], linewidth=2, label=f'Mean-{sclass}' if pidx == num_patterns - 1 else None)
    max_lim = max([max(ax[pidx].get_ylim()) for pidx in range(num_patterns)])
    for pidx in range(num_patterns):
        if match_scale:
            ax[pidx].set_ylim((0, max_lim))
        else:
            ax[pidx].set_ylim((0, max(ax[pidx].get_ylim())))
        ax[pidx].plot(np.zeros(2), ax[pidx].get_ylim(), '-', color='darkgrey', linewidth=2, zorder=0)
    if psth_target is None:
        tgt_name = 'Population'
    else:
        tgt_name = f'"{psth_target}"'
    if label is None:
        lbl_str = ''
    else:
        lbl_str = f' ({label})'
    fig.suptitle(f'{tgt_name} PSTHs{lbl_str}', fontweight='bold')
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.93), ncol=(len(layers) + 1) * len(syn_classes), fontsize=6)
    fig.tight_layout()

    if save_fn is not None:
        plt.savefig(save_fn, dpi=300)
