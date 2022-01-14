# Description: Analysis to be used with BBP-WORKFLOW analysis launcher for computing
#              peak statistics of PSTHs of visual contrast stimulus responses
#              => Single cell PSTHs need to be computed beforehand and exist as .pkl files!
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
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


def detect_rate_peaks():
    #TODO: Insert from helper_functions incl. required imports


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

    psth_base_fn = params.get('psth_base_fn') # Base filename to load PSTH data from (will be appended by sim condition specifier)
    pattern_idx = params.get('pattern_idx') # Grating pattern (contrast) index to compute peak statistics from (N patterns: idx 0..N-1 w/o opto, idx N..2N-1 with opto)
    peak_th = params.get('peak_th') # Peak detection threshold (Hz)
    peak_width = params.get('peak_width') # Min. peak width (ms)
    peak_distance = params.get('peak_distance') # Min. distance between peaks (ms)
    peak_range = params.get('peak_range') # Time range to detect peaks, e.g. [0.0, 1000.0]
    do_plot = bool(params.get('do_plot'))

    # Check input data existence first & run analysis (peak statistics)
    for check_only in [True, False]:
        for cond, cfg_path in sims.iteritems():

            cond_dict = dict(zip(cond_names, cond))
            sim_id = os.path.split(os.path.split(cfg_path)[0])[-1] # Subfolder name (i.e., 000, 001, ...)
            sim_spec = '__'.join([f'{k}_{v}' for k, v in cond_dict.items()]) # Sim conditions (e.g., sparsity_1.0__rate_bk_0.2__rate_max_10.0)

            # Check PSTH data
            psth_file = os.path.join(output_root, f'{psth_base_fn}__SIM{sim_id}__{sim_spec}.pickle')
            assert os.path.exists(psth_file), f'ERROR: PSTH data file {psth_file} not found! Run "single_cell_psths" first!'
            if check_only:
                continue

            # Load PSTH data
            with open(psth_file, 'rb') as f:
                psth_data = pickle.load(f)

            # Compute peak statistics
            peak_idx, t1, t2, r1, r2, peak_ratio = detect_rate_peaks(t_rate, rates[patt_sel], peak_th=peak_th, peak_width=peak_width, peak_distance=peak_distance, t_range=peak_range)
            res_dict = {'peak_idx': peak_idx, 't1': t1, 't2': t2, 'r1': r1, 'r2': r2, 'peak_ratio': peak_ratio}
            res_dict.update({'sim_id': sim_id, 'cond_dict': cond_dict, 'pattern_idx': pattern_idx, 'peak_th': peak_th, 'peak_width': peak_width, 'peak_distance': peak_distance, 'peak_range': peak_range})

            # Write to pickled files
            res_file = os.path.join(output_root, f'psth_peak_stats__SIM{sim_id}__{sim_spec}.pickle')
            with open(res_file, 'wb') as f:
                pickle.dump(res_dict, f)
            print(f'INFO: PSTH peak statistics written to {res_file}')

            # Do some plotting
            if do_plot:
                pass


if __name__ == "__main__":
    main()
