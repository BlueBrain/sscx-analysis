# Description: Analysis to be used with BBP-WORKFLOW analysis launcher for computing
#              single-cell PSTHs of visual contrast stimulus responses
# Author: C. Pokorny
# Last modified: 14/01/2022

import sys
import json
import os
import pickle
import numpy as np
import pandas as pd
from bluepy import Simulation
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


def get_single_cell_psths(blue_config, target_spec, psth_interval=None, t_res=20.0, t_smooth=None):
    """Extract single-cell spikes per pattern, re-aligned to stimulus onsets"""

    # Load simulation, circuit, and stim/opto configs
    sim = Simulation(blue_config)
    c = sim.circuit

    stim_file = os.path.abspath(os.path.join(os.path.split(blue_config)[0], sim.config['Stimulus_spikeReplay']['SpikeFile']))
    stim_file = os.path.splitext(stim_file)[0] + '.json'
    assert os.path.exists(stim_file), 'ERROR: Stim config file not found!'
    with open(stim_file, 'r') as f:
        stim_cfg = json.load(f)

    opto_file = os.path.join(os.path.split(blue_config)[0], 'opto_stim.json')
    if os.path.exists(opto_file):
        with open(opto_file, 'r') as f:
            opto_cfg = json.load(f)
    else:
        # print('INFO: No opto config found!')
        opto_cfg = None

    # Select cells and spikes
    if isinstance(target_spec, dict):
        target_spec = target_spec.copy()
        cell_target = target_spec.pop('target', None)
        gids = c.cells.ids(target_spec)
        if cell_target is not None:
            gids = np.intersect1d(gids, c.cells.ids(cell_target))
    else:
        gids = c.cells.ids(target_spec)

    spikes = sim.spikes.get(gids)

    # Find overlapping opto stimuli
    time_windows = stim_cfg['props']['time_windows']
    stim_train = stim_cfg['props']['stim_train']
    num_patterns = max(stim_train) + 1
    stim_int = [[t, t + stim_cfg['cfg']['duration_stim']] for t in np.array(time_windows[:-1])]
    if opto_cfg is None:
        stim_opto_overlap = np.zeros(len(stim_int))
    else:
        opto_int = [[opto_cfg['props']['opto_t'][i], opto_cfg['props']['opto_t'][i] + opto_cfg['props']['opto_dur'][i]] for i in range(len(opto_cfg['props']['opto_t']))]
        stim_opto_overlap = np.array([np.max([(max(0, min(max(stim_int[i]), max(oint)) - max(min(stim_int[i]), min(oint)))) / np.diff(stim_int[i]) for oint in opto_int]) for i in range(len(stim_int))])
    opto_stim_train = (stim_opto_overlap > 0.0) * num_patterns + stim_train # Reindex stimuli (0..N-1: stims w/o opto, N..2N-1: stims with opto)

    # Extract single-cell PSTHs
    if psth_interval is None:
        psth_interval = [0, np.max(np.diff(time_windows))]
    spike_trains = [{gid: [] for gid in gids} for p in range(num_patterns * 2)] # Spikes per pattern and gid, re-aligned to stimulus onset
    for sidx, pidx in enumerate(opto_stim_train):
        spk = spikes[np.logical_and(spikes.index - time_windows[sidx] >= psth_interval[0],
                                    spikes.index - time_windows[sidx] < psth_interval[-1])]
        spk.index = spk.index - time_windows[sidx] # Re-align to stimulus onset
        for gid in gids:
            spike_trains[pidx][gid].append(spk[spk.values == gid].index.to_list())

    # Single-cell average spike rates over trials and PSTH interval
    avg_cell_rates = [[1e3 * np.mean([len(st) for st in spike_trains[p][gid]]) / np.diff(psth_interval)[0] if len(spike_trains[p][gid]) > 0 else np.nan for gid in gids] for p in range(num_patterns * 2)]

    # Instantaneous spike rates averaged over trials
    rates = []
    for p in range(num_patterns * 2):
        pattern_rates = []
        for gid in gids:
            t_rate, rate = instant_firing_rate(spike_trains[p][gid], t_res, t_min=psth_interval[0], t_max=psth_interval[-1], t_smooth=t_smooth)
            pattern_rates.append(rate)
        rates.append(np.array(pattern_rates))

    return t_rate, rates, spike_trains, avg_cell_rates, gids, stim_cfg, opto_cfg


def instant_firing_rate(spikes, res_ms, t_min=None, t_max=None, t_smooth=None):
    """ Estimation of instantaneous firing rates of given spike trains
        over multiple trials (list of lists of spike times), optionally
        using Gaussian smoothing
    """
    assert res_ms > 0.0, 'ERROR: Resolution must be larger than zero ms!'
    num_trials = len(spikes)

    if t_min is None:
        t_min = np.min(spikes)
    if t_max is None:
        t_max = np.max(spikes)

    bins = np.linspace(t_min, t_max, np.round((t_max - t_min) / res_ms).astype(int) + 1)
    hist_count = np.zeros(len(bins) - 1)
    for trial in range(num_trials):
        hist_count += np.histogram(spikes[trial], bins=bins)[0]
    if num_trials == 0:
        rate = np.full_like(hist_count, np.nan)
    else:
        rate = hist_count / (num_trials * res_ms * 1e-3)
    t = bins[:-1] + 0.5 * res_ms

    if t_smooth is not None: # Allpy Gaussian smoothing
        assert t_smooth > 0.0, 'ERROR: Smoothing time constant must be larger than zero ms!'
        rate = gaussian_filter1d(rate, sigma=t_smooth / res_ms)

    return t, rate


def plot_psth_maps(t_rate, rates, save_path, save_spec=None):
    """ Plots PSTH overview maps (all cells) for all stimulus conditions. """

    if save_spec is None:
        save_spec = ''
    if not isinstance(save_spec, str):
        save_spec = str(save_spec)
    if len(save_spec) > 0:
        save_spec = '_' + save_spec

    # Plot figure
    num_patterns = len(rates)
    plt.figure(figsize=(5 * num_patterns, 5))
    t_res = np.median(np.diff(t_rate))
    for p in range(num_patterns):
        plt.subplot(1, num_patterns, p + 1)
        plt.imshow(rates[p][sort_idx, :], extent=(t_rate[0] - 0.5 * t_res, t_rate[-1] + 0.5 * t_res, rates[p].shape[0] - 0.5, -0.5), aspect='auto', interpolation='nearest')
        plt.colorbar(label='Firing rate (Hz)')
        plt.xlabel('Time (ms)')
        plt.ylabel('Cells (sorted)')
        if p < num_patterns >> 1: # Patterns assumed to be twice the actual grating patterns, first w/o and second with OPTO stimulation on top
            plt.title(f'Pattern {p}')
        else:
            plt.title(f'Pattern {p - (num_patterns >> 1)}-OPTO')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f'psth_overview{save_spec}.png'), dpi=300)
    plt.show()


def plot_psths_spikes(t_rate, rates, spike_trains, avg_cell_rates, gids, N_to_plot, save_path, save_spec=None):
    """ Plots single-trial spikes and PSTHs for N selected GIDs.
        The N most responding GIDs are selected based on their
        average firing rate over all stimulus conditions."""

    if N_to_plot is None or N_to_plot <= 0:
        return # Nothing to plot

    if save_spec is None:
        save_spec = ''
    if not isinstance(save_spec, str):
        save_spec = str(save_spec)
    if len(save_spec) > 0:
        save_spec = '_' + save_spec

    # Filter & sort GIDs by increasing average firing rates over all patterns
    avg_rates_sel = np.nanmean(avg_cell_rates, 0)
    sort_idx = np.argsort(avg_rates_sel)[::-1] # Ordered by decreasing rate
    sel_idx = sort_idx[:N_to_plot] # Selecting N gids
    gids_sel = gids[sel_idx]

    # Define y axis scalings
    y_scale_range = 0.9
    y_rate_scale = y_scale_range / np.nanmax(rates)
    y_trial_scale = y_scale_range / (np.max([[len(spike_trains[p][gid]) for gid in spike_trains[p].keys()] for p in range(len(spike_trains))]) - 1)

    # Plot figure
    num_patterns = len(rates)
    plt.figure(figsize=(5 * num_patterns, 5))
    for p in range(num_patterns):
        rates_sel = rates[p][sel_idx, :]
        plt.subplot(1, num_patterns, p + 1)
        for gidx, gid in enumerate(gids_sel):
            trials = spike_trains[p][gid]
            y_offset = len(gids_sel) - 1 - gidx - y_scale_range / 2
            for trial, st in enumerate(trials):
                plt.plot(st, np.full(len(st), trial * y_trial_scale + y_offset), '.k', markersize=2.0, markeredgecolor='none')
            plt.plot(t_rate, rates_sel[gidx, :] * y_rate_scale + y_offset, 'r-')
        plt.yticks(np.arange(len(gids_sel)), gids_sel[::-1])
        plt.ylim((-1, len(gids_sel)))
        plt.xlabel('Time (ms)')
        plt.ylabel('GID')
        if p < num_patterns >> 1: # Patterns assumed to be twice the actual grating patterns, first w/o and second with OPTO stimulation on top
            plt.title(f'Pattern {p}')
        else:
            plt.title(f'Pattern {p - (num_patterns >> 1)}-OPTO')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f'psth_spikes{save_spec}.png'), dpi=300)
    plt.show()


def main():

    # Parse inputs
    args = sys.argv[1:]
    if len(args) < 2:
        print(f'Usage: {__file__} simulations.pkl config_file.json')
        sys.exit(2)

    # Load simulation table
    sims = pd.read_pickle(args[0])

    # Load analysis parameters
    with open(args[1], 'r') as f:
        params = json.load(f)

    # Get params
    output_root = params.get('output_root')
    assert output_root is not None, 'ERROR: Output root folder not specified!'

    cell_target = params.get('cell_target') # Cell target (str), e.g. 'hex0'
    cell_filter = params.get('cell_filter') # Cell filter (dict), e.g. {'synapse_class': 'EXC', 'layer': 4}
    psth_res = params.get('psth_res') # PSTH time resolution (ms), e.g. 1.0
    psth_smooth = params.get('psth_smooth') # PSTH Gaussian smoothing time constant (ms), e.g. 20.0
    N_to_plot = params.get('N_cells_to_plot') # Number of most responding cells to plot

    # Label for file names
    spec_label_base = '-'.join([f'{k}{v}' for k, v in cell_filter.items()])
    spec_label_base = spec_label_base.replace('synapse_class', '')
    spec_label_base = spec_label_base.replace('layer', 'L')
    spec_label_base = '_'.join([cell_target, spec_label_base])

    # Run analysis (single cell PSTHs)
    cond_names = sims.index.names
    for cond, cfg_path in sims.iteritems():
        cond_dict = dict(zip(cond_names, cond))

        sim_id = os.path.split(os.path.split(sims.iloc[0])[0])[-1] # Subfolder name (i.e., 000, 001, ...)
        sim_spec = '__'.join([f'{k}_{v}' for k, v in cond_dict.items()]) # Sim conditions (e.g., sparsity_1.0__rate_bk_0.2__rate_max_10.0)

        # Compute PSTHs
        t_rate, rates, spike_trains, avg_cell_rates, gids, stim_cfg, opto_cfg = get_single_cell_psths(cfg_path, {'target': cell_target, **cell_filter}, t_res=psth_res, t_smooth=psth_smooth)
        res_dict = {'t_rate': t_rate, 'rates': rates, 'spike_trains': spike_trains, 'avg_cell_rates': avg_cell_rates, 'gids': gids, 'stim_cfg': stim_cfg, 'opto_cfg': opto_cfg}
        res_dict.update({'sim_id': sim_id, 'cond_dict': cond_dict})

        # Write to pickled files
        res_file = os.path.join(output_root, f'single_cell_psths__SIM{sim_id}__{sim_spec}.pickle')
        with open(res_file, 'wb') as f:
            pickle.dump(res_dict, f)
        print(f'INFO: Single-cell PSTH data written to {res_file}')

        # Do some plotting
        plot_psth_maps(t_rate, rates, output_root, f'_SIM{sim_id}__{sim_spec}')
        plot_psths_spikes(t_rate, rates, spike_trains, avg_cell_rates, gids, N_to_plot, output_root, f'_SIM{sim_id}__{sim_spec}')


if __name__ == "__main__":
    main()
