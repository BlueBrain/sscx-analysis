# Description: Analysis to be used with BBP-WORKFLOW analysis launcher for computing
#              peak statistics of PSTHs of visual contrast stimulus responses
#              => Single cell PSTHs need to be computed beforehand and exist as .pkl files!
# Author: C. Pokorny
# Last modified: 01/2022

import sys
import json
import os
import pickle
import numpy as np
import pandas as pd
from bluepy import Simulation
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


def detect_rate_peaks(t_rate, rates, peak_th=5.0, peak_width=20.0, peak_distance=200.0, t_range=None):
    """
    Peak detection (first & second peak) of firing rates (<#GIDs x #time_steps>),
    using peak_th (Hz), peak_width (ms), peak_distance (ms), within given time
    range and selecting the two highest peaks
    """
    t_res = np.median(np.diff(t_rate))
    peak_idx = [find_peaks(r, height=peak_th, width=peak_width / t_res, distance=peak_distance / t_res)[0] for r in rates]

    # Remove out-of-range peaks
    if t_range is not None:
        for idx, pidx in enumerate(peak_idx):
            peak_idx[idx] = pidx[np.logical_and(t_rate[pidx] >= t_range[0], t_rate[pidx] < t_range[1])]

    # In case of more than two (remaining) peaks: keep only two highest
    for idx, pidx in enumerate(peak_idx):
        r = rates[idx][pidx]
        sel = np.argsort(r)[:-3:-1] # Find two highest
        peak_idx[idx] = pidx[np.sort(sel)] # Select two highest (keeping order)

    peak_rate = [rates[idx][pidx] for idx, pidx in enumerate(peak_idx)]
    peak_t = [t_rate[pidx] for pidx in peak_idx]

    t1 = [t[0] if len(t) > 0 else np.nan for t in peak_t] # First peak time
    t2 = [t[1] if len(t) > 1 else np.nan for t in peak_t] # Second peak time
    r1 = [r[0] if len(r) > 0 else np.nan for r in peak_rate] # First peak rate
    r2 = [r[1] if len(r) > 1 else np.nan for r in peak_rate] # Second peak rate

    peak_ratio = np.array([(_r1 - _r2) / (_r1 + _r2) for (_r1, _r2) in zip(r1, r2)])

    return peak_idx, t1, t2, r1, r2, peak_ratio


def plot_peak_overview(t_rate, rates, t1, t2, r1, r2, peak_ratio, save_path, save_spec=None):
    """ Plot instantaneous firing rates (<#GIDs x #time_steps>)
        inkl. peaks (all GIDs for which peak_ratio is defined) """

    if save_spec is None:
        save_spec = ''
    if not isinstance(save_spec, str):
        save_spec = str(save_spec)
    if len(save_spec) > 0:
        save_spec = '_' + save_spec

    gid_sel = np.isfinite(peak_ratio)
    plt.figure(figsize=(8, 3))
    plt.plot(t_rate, rates.T, 'k', alpha=0.25)
    for idx in range(np.sum(gid_sel)):
        plt.plot(np.array(t1)[gid_sel][idx], np.array(r1)[gid_sel][idx], 'x', color='tab:blue', alpha=1.0, label='First peak' if idx == 0 else None)
        plt.plot(np.array(t2)[gid_sel][idx], np.array(r2)[gid_sel][idx], 'x', color='tab:orange', alpha=1.0, label='Second peak' if idx == 0 else None)
    plt.grid()
    plt.xlabel('Time (ms)')
    plt.ylabel('Firing rate (Hz)')
    plt.title(f'PSTH peak overview ({np.sum(gid_sel)} of {len(gid_sel)} cells)', fontweight='bold')
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f'psth_peak_overview{save_spec}.png'), dpi=300)
    plt.show()


def plot_peak_statistics(t1, t2, r1, r2, peak_ratio, save_path, save_spec=None, num_bins=None):
    """ Plot peak statistics of first vs. second peak """

    if save_spec is None:
        save_spec = ''
    if not isinstance(save_spec, str):
        save_spec = str(save_spec)
    if len(save_spec) > 0:
        save_spec = '_' + save_spec

    if num_bins is None:
        num_bins = [50, 25, 25] # Default for (i) peak time, (ii) peak rate, (iii) peak ratio histograms
    else:
        if np.isscalar(num_bins):
            num_bins = [num_bins] * 3
        else:
            assert len(num_bins) == 3, 'ERROR: Number of bins must be a scalar or a list with 3 entries!'

    gid_sel = np.isfinite(peak_ratio)
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 4, 1)
    nbins = num_bins[0]
    plt.hist(np.array(t1)[gid_sel], bins=np.linspace(0, np.max([np.array(t1)[gid_sel], np.array(t2)[gid_sel]]), nbins + 1), width=1.0 * np.max([np.array(t1)[gid_sel], np.array(t2)[gid_sel]]) / nbins, label='First')
    plt.hist(np.array(t2)[gid_sel], bins=np.linspace(0, np.max([np.array(t1)[gid_sel], np.array(t2)[gid_sel]]), nbins + 1), width=1.0 * np.max([np.array(t1)[gid_sel], np.array(t2)[gid_sel]]) / nbins, label='Second')
    plt.grid()
    plt.xlabel('Peak time (ms)')
    plt.ylabel('Count')
    plt.title('Peak time histograms')
    plt.legend()

    plt.subplot(1, 4, 2)
    nbins = num_bins[1]
    plt.hist([np.array(r1)[gid_sel], np.array(r2)[gid_sel]], bins=nbins, label=['First', 'Second'])
    plt.grid()
    plt.xlabel('Peak rate (Hz)')
    plt.ylabel('Count')
    plt.title('Peak rate histograms')
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.plot(r1, r2, '.', color='tab:purple')
    plt.xlim([0, np.max([np.array(r1)[gid_sel], np.array(r2)[gid_sel]])])
    plt.ylim([0, np.max([np.array(r1)[gid_sel], np.array(r2)[gid_sel]])])
    plt.plot([min(plt.xlim()), max(plt.xlim())], [min(plt.ylim()), max(plt.ylim())], '--k', zorder=0)
    plt.grid()
    plt.xlabel('First peak rate (Hz)')
    plt.ylabel('Second peak rate (Hz)')
    plt.title('First vs. second peak rate')

    plt.subplot(1, 4, 4)
    nbins = num_bins[2]
    plt.hist(peak_ratio, bins=nbins, color='tab:purple')
    plt.ylim(plt.ylim())
    plt.plot(np.zeros(2), plt.ylim(), '--k')
    plt.plot(np.full(2, np.nanmean(peak_ratio)), plt.ylim(), '-', color='tab:green', linewidth=3)
    plt.text(np.nanmean(peak_ratio), 0.99 * max(plt.ylim()), f'  Mean: {np.nanmean(peak_ratio):.3f}', color='tab:green', ha='left', va='top')
    plt.grid()
    plt.xlabel('Norm. peak ratio')
    plt.ylabel('Count')
    plt.title('First vs. second peak ratio')

    plt.suptitle(f'PSTH peak statistics ({np.sum(gid_sel)} of {len(gid_sel)} cells)', fontweight='bold')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f'psth_peak_stats{save_spec}.png'), dpi=300)
    plt.show()


def plot_peak_examples(gid_idx_to_plot, t_rate, rates, spike_trains, gids, t1, t2, r1, r2, peak_ratio, save_path, save_spec=None):
    """ Plot spikes, firing rate PSTH, and detected peaks
        for selected exemplary cell indices """

    if save_spec is None:
        save_spec = ''
    if not isinstance(save_spec, str):
        save_spec = str(save_spec)
    if len(save_spec) > 0:
        save_spec = '_' + save_spec

    if gid_idx_to_plot is None or len(gid_idx_to_plot) == 0:
        return # Nothing to plot
    assert np.all(gid_idx_to_plot >= 0) and np.all(gid_idx_to_plot < len(gids)), 'ERROR: Cell plot indices out of range!'

    for gidx in gid_idx_to_plot:
        gid = gids[gidx]
        plt.figure(figsize=(10, 1))
        plt.plot(t_rate, rates[gidx, :], 'k')
        plt.plot(t1[gidx], r1[gidx], 'x', color='tab:blue', markersize=10, clip_on=False, label='First peak')
        plt.plot(t2[gidx], r2[gidx], 'x', color='tab:orange', markersize=10, clip_on=False, label='Second peak')
        plt.ylim(plt.ylim())
        trials = spike_trains[gid]
        y_scale = np.diff(plt.ylim()) / len(trials)
        for trial, st in enumerate(trials):
            plt.plot(st, np.full(len(st), (trial - 0.5 * (len(trials) - 1)) * y_scale + np.mean(plt.ylim())), '|g', markersize=5.0)
        plt.grid()
        plt.xlabel('Time (ms)')
        plt.ylabel('Firing rate (Hz)')
        plt.title(f'GID {gid} (peak_ratio={peak_ratio[gidx]:.3})', fontweight='bold')
        plt.legend(loc='upper left', bbox_to_anchor=[1.0, 1.0])
        if save_path is not None:
            plt.savefig(os.path.join(save_path, f'psth_rate_peaks{save_spec}_GID{gid}.png'), dpi=300)
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

    psth_name = params.get('psth_name') # PSTH folder/filename to load PSTH data from (will be appended by sim condition specifier)
    psth_path = os.path.join(os.path.normpath(os.path.join(output_root, '..')), psth_name)
    pattern_idx = params.get('pattern_idx') # Grating pattern (contrast) index to compute peak statistics from (N patterns: idx 0..N-1 w/o opto, idx N..2N-1 with opto)
    peak_th = params.get('peak_th') # Peak detection threshold (Hz)
    peak_width = params.get('peak_width') # Min. peak width (ms)
    peak_distance = params.get('peak_distance') # Min. distance between peaks (ms)
    peak_range = params.get('peak_range') # Time range to detect peaks, e.g. [0.0, 1000.0]
    do_plot = bool(params.get('do_plot'))
    num_bins = params.get('num_bins') # Number of bins for (i) peak time, (ii) peak rate, (iii) peak ratio histograms (scalar of 3 items list)
    gids_to_plot = params.get('gids_to_plot') # List of exemplary GIDs to plot in detail OR
    cell_idx_to_plot = params.get('cell_idx_to_plot') # List of (sorted) cell indices to plot in detail

    if do_plot:
        figs_path = os.path.join(output_root, 'figs')
        if not os.path.exists(figs_path):
            os.makedirs(figs_path)

        if gids_to_plot is not None:
            assert cell_idx_to_plot is None, 'ERROR: Either "cell_idx_to_plot" or "gids_to_plot" can be specified!'
        if cell_idx_to_plot is not None:
            assert gids_to_plot is None, 'ERROR: Either "cell_idx_to_plot" or "gids_to_plot" can be specified!'

    # Check input data existence first & run analysis (peak statistics)
    cond_names = sims.index.names
    for check_only in [True, False]:
        for cond, cfg_path in sims.iteritems():

            cond_dict = dict(zip(cond_names, cond))
            sim_id = os.path.split(os.path.split(cfg_path)[0])[-1] # Subfolder name (i.e., 000, 001, ...)
            sim_spec = '__'.join([f'{k}_{v}' for k, v in cond_dict.items()]) # Sim conditions (e.g., sparsity_1.0__rate_bk_0.2__rate_max_10.0)

            # Check PSTH data
            psth_file = os.path.join(psth_path, f'{psth_name}__SIM{sim_id}__{sim_spec}.pickle')
            assert os.path.exists(psth_file), f'ERROR: Required PSTH data file {psth_file} not found! Please run "{psth_name}" analysis first!'
            if check_only:
                continue

            # Load PSTH data
            with open(psth_file, 'rb') as f:
                psth_data = pickle.load(f)
            assert sim_id == psth_data['sim_id'], 'ERROR: Sim ID mismatch!'
            assert cond_dict == psth_data['cond_dict'], 'ERROR: Sim conditions mismatch!'
            t_rate = psth_data['t_rate']
            rates = psth_data['rates']
            avg_cell_rates = psth_data['avg_cell_rates']
            spike_trains = psth_data['spike_trains']
            gids = psth_data['gids']

            # Compute peak statistics
            peak_idx, t1, t2, r1, r2, peak_ratio = detect_rate_peaks(t_rate, rates[pattern_idx], peak_th=peak_th, peak_width=peak_width, peak_distance=peak_distance, t_range=peak_range)
            res_dict = {'peak_idx': peak_idx, 't1': t1, 't2': t2, 'r1': r1, 'r2': r2, 'peak_ratio': peak_ratio}
            res_dict.update({'sim_id': sim_id, 'cond_dict': cond_dict, 'pattern_idx': pattern_idx, 'peak_th': peak_th, 'peak_width': peak_width, 'peak_distance': peak_distance, 'peak_range': peak_range})

            # Write to pickled file
            res_file = os.path.join(output_root, f'psth_peak_stats__SIM{sim_id}__{sim_spec}.pickle')
            with open(res_file, 'wb') as f:
                pickle.dump(res_dict, f)
            print(f'INFO: PSTH peak statistics written to {res_file}')

            # Do some plotting
            if do_plot:                
                plot_peak_overview(t_rate, rates[pattern_idx], t1, t2, r1, r2, peak_ratio, figs_path, f'_SIM{sim_id}__{sim_spec}')
                plot_peak_statistics(t1, t2, r1, r2, peak_ratio, figs_path, f'_SIM{sim_id}__{sim_spec}', num_bins)
                
                if gids_to_plot is not None:
                    gid_idx_to_plot = [np.where(gids == gid)[0][0] for gid in gids_to_plot]
                elif cell_idx_to_plot is not None:
                    avg_rates_sel = np.nanmean(avg_cell_rates, 0)
                    sort_idx = np.argsort(avg_rates_sel)[::-1] # Ordered by decreasing rate
                    gid_idx_to_plot = sort_idx[cell_idx_to_plot]
                else:
                    gid_idx_to_plot = None
                plot_peak_examples(gid_idx_to_plot, t_rate, rates[pattern_idx], spike_trains[pattern_idx], gids, t1, t2, r1, r2, peak_ratio, figs_path, f'_SIM{sim_id}__{sim_spec}')


if __name__ == "__main__":
    main()
