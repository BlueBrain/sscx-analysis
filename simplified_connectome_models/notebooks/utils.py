"""
Utilities functions for accessing and analysing SONATA simulation campaigns.
Author: C. Pokorny
Date: 10/2023
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tqdm
from bluepysnap import Circuit, Simulation
from scipy.stats import wilcoxon

COLOR_EXC = 'tab:red'
COLOR_INH = 'tab:blue'

RATE_COLOR_DICT = {'EXC': 'darkred',
                   'INH': 'royalblue',
                   'INP': 'black',
                   'Input': 'black'}


def _get_population_spikes(sim):
    """Returns spikes of non-virtual nodes population."""
    c = sim.circuit
    popul_name = [_p for _p in c.nodes.population_names if c.nodes[_p].type != 'virtual']
    assert len(popul_name) == 1, "ERROR: No or multiple non-virtual nodes populations found!"

    nodes = c.nodes[popul_name[0]]
    spikes = sim.spikes[popul_name[0]]

    return nodes, spikes


def extract_spikes(circuit_names, sim_configs, save_name, node_set='hex0', save_path=None, load_if_existing=True):
    """Extracts spikes per layer and EXC/INH population."""
    if save_path is None:
        save_path = ''
    fn = os.path.join(save_path, save_name + f'_{node_set}.npz')
    if os.path.exists(fn) and load_if_existing:
        data_dict = dict(np.load(fn, allow_pickle=True))
        print(f'INFO: Spike data loaded from "{fn}"')
    else:
        spk_exc = []
        spk_inh = []

        spk_exc_per_layer = []
        spk_inh_per_layer = []

        for ci, cn in enumerate(tqdm.tqdm(circuit_names)):
            sim = Simulation(sim_configs[cn])
            nodes, spikes = _get_population_spikes(sim)

            # EXC/INH rates
            nids_exc = np.intersect1d(nodes.ids(node_set), nodes.ids('Excitatory'))
            nids_inh = np.intersect1d(nodes.ids(node_set), nodes.ids('Inhibitory'))
            st_e = spikes.get(nids_exc)
            st_i = spikes.get(nids_inh)
            spk_exc.append(st_e)
            spk_inh.append(st_i)

            #EXC/INH rates per layer
            layers = np.unique(nodes.get(properties='layer'))
            spk_exc_lay = []
            spk_inh_lay = []
            for lay in layers:
                nids_exc_lay = np.intersect1d(nids_exc, nodes.ids({'layer': lay}))
                nids_inh_lay = np.intersect1d(nids_inh, nodes.ids({'layer': lay}))
                st_e = spikes.get(nids_exc_lay)
                st_i = spikes.get(nids_inh_lay)
                spk_exc_lay.append(st_e)
                spk_inh_lay.append(st_i)
            spk_exc_per_layer.append(spk_exc_lay)
            spk_inh_per_layer.append(spk_inh_lay)

        # Save to disc
        data_dict = dict(spk_exc=np.array(spk_exc + [pd.DataFrame([])], dtype=object)[:-1],  # [Weird conversion to np.array (incl. dummy) required for storing as .npz file and keeping data frames]
                         spk_inh=np.array(spk_inh + [pd.DataFrame([])], dtype=object)[:-1],  # [Weird conversion to np.array (incl. dummy) required for storing as .npz file and keeping data frames]
                         spk_exc_per_layer=np.array(spk_exc_per_layer + [pd.DataFrame([])], dtype=object)[:-1],  # [Weird conversion to np.array (incl. dummy) required for storing as .npz file and keeping data frames]
                         spk_inh_per_layer=np.array(spk_inh_per_layer + [pd.DataFrame([])], dtype=object)[:-1])  # [Weird conversion to np.array (incl. dummy) required for storing as .npz file and keeping data frames]
        np.savez(fn, **data_dict)
        print(f'INFO: Spike data saved to "{fn}"')
    return data_dict


def extract_rates(circuit_names, sim_configs, save_name, node_set='hex0', t_start=None, t_end=None, save_path=None, load_if_existing=True):
    """Extracts average firing rates per layer and EXC/INH population."""
    if save_path is None:
        save_path = ''
    fn = os.path.join(save_path, save_name + f'_{node_set}.npz')
    if os.path.exists(fn) and load_if_existing:
        data_dict = dict(np.load(fn))
        print(f'INFO: Rate data loaded from "{fn}"')
    else:
        num_exc = []
        num_inh = []
        rates_exc = []
        rates_inh = []
        rates_exc_spiking = []
        rates_inh_spiking = []
        pct_exc_spiking = []
        pct_inh_spiking = []

        num_exc_per_layer = []
        num_inh_per_layer = []
        rates_exc_per_layer = []
        rates_inh_per_layer = []
        rates_exc_spiking_per_layer = []
        rates_inh_spiking_per_layer = []
        pct_exc_spiking_per_layer = []
        pct_inh_spiking_per_layer = []

        for ci, cn in enumerate(tqdm.tqdm(circuit_names)):
            sim = Simulation(sim_configs[cn])
            nodes, spikes = _get_population_spikes(sim)
            if t_start is None:
                t_start = spikes.spike_report.time_start
            if t_end is None:
                t_end = spikes.spike_report.time_stop

            # EXC/INH rates
            nids_exc = np.intersect1d(nodes.ids(node_set), nodes.ids('Excitatory'))
            nids_inh = np.intersect1d(nodes.ids(node_set), nodes.ids('Inhibitory'))
            n_e = len(nids_exc)
            n_i = len(nids_inh)
            st_e = spikes.get(nids_exc, t_start=t_start, t_stop=t_end)
            st_i = spikes.get(nids_inh, t_start=t_start, t_stop=t_end)
            r_e = len(st_e) / (n_e * 1e-3 * (t_end - t_start)) # EXC population rate
            r_i = len(st_i) / (n_i * 1e-3 * (t_end - t_start)) # INH population rate
            r_e_spk = len(st_e) / (len(np.unique(st_e.values)) * 1e-3 * (t_end - t_start)) # Spiking EXC population rate
            r_i_spk = len(st_i) / (len(np.unique(st_i.values)) * 1e-3 * (t_end - t_start)) # Spiking INH population rate
            p_e_spk = 100.0 * len(np.unique(st_e.values)) / n_e
            p_i_spk = 100.0 * len(np.unique(st_i.values)) / n_i
            num_exc.append(n_e)
            num_inh.append(n_i)
            rates_exc.append(r_e)
            rates_inh.append(r_i)
            rates_exc_spiking.append(r_e_spk)
            rates_inh_spiking.append(r_i_spk)
            pct_exc_spiking.append(p_e_spk)
            pct_inh_spiking.append(p_i_spk)

            #EXC/INH rates per layer
            layers = np.unique(nodes.get(properties='layer'))
            num_exc_lay = []
            num_inh_lay = []
            rates_exc_lay = []
            rates_inh_lay = []
            rates_exc_spiking_lay = []
            rates_inh_spiking_lay = []
            pct_exc_spiking_lay = []
            pct_inh_spiking_lay = []
            for lay in layers:
                nids_exc_lay = np.intersect1d(nids_exc, nodes.ids({'layer': lay}))
                nids_inh_lay = np.intersect1d(nids_inh, nodes.ids({'layer': lay}))
                n_e = len(nids_exc_lay)
                n_i = len(nids_inh_lay)
                st_e = spikes.get(nids_exc_lay, t_start=t_start, t_stop=t_end)
                st_i = spikes.get(nids_inh_lay, t_start=t_start, t_stop=t_end)
                n_e_spk = len(np.unique(st_e.values))
                n_i_spk = len(np.unique(st_i.values))
                if n_e == 0:
                    r_e = np.nan
                    p_e_spk = np.nan
                else:
                    r_e = np.double(len(st_e)) / (n_e * 1e-3 * (t_end - t_start)) # EXC population rate
                    p_e_spk = 100.0 * np.double(n_e_spk) / n_e
                if n_e_spk == 0:
                    r_e_spk = np.nan
                else:
                    r_e_spk = np.double(len(st_e)) / (n_e_spk * 1e-3 * (t_end - t_start)) # Spiking EXC population rate
                if n_i == 0:
                    r_i = np.nan
                    p_i_spk = np.nan
                else:
                    r_i = np.double(len(st_i)) / (n_i * 1e-3 * (t_end - t_start)) # INH population rate
                    p_i_spk = 100.0 * np.double(n_i_spk) / n_i
                if n_i_spk == 0:
                    r_i_spk = np.nan
                else:
                    r_i_spk = np.double(len(st_i)) / (n_i_spk * 1e-3 * (t_end - t_start)) # Spiking INH population rate
                num_exc_lay.append(n_e)
                num_inh_lay.append(n_i)
                rates_exc_lay.append(r_e)
                rates_inh_lay.append(r_i)
                rates_exc_spiking_lay.append(r_e_spk)
                rates_inh_spiking_lay.append(r_i_spk)
                pct_exc_spiking_lay.append(p_e_spk)
                pct_inh_spiking_lay.append(p_i_spk)
            num_exc_per_layer.append(num_exc_lay)
            num_inh_per_layer.append(num_inh_lay)
            rates_exc_per_layer.append(rates_exc_lay)
            rates_inh_per_layer.append(rates_inh_lay)
            rates_exc_spiking_per_layer.append(rates_exc_spiking_lay)
            rates_inh_spiking_per_layer.append(rates_inh_spiking_lay)
            pct_exc_spiking_per_layer.append(pct_exc_spiking_lay)
            pct_inh_spiking_per_layer.append(pct_inh_spiking_lay)

        # Save to disc
        data_dict = dict(t_start=t_start,
                         t_end=t_end,
                         num_exc=num_exc,
                         num_inh=num_inh,
                         rates_exc=rates_exc,
                         rates_inh=rates_inh,
                         rates_exc_spiking=rates_exc_spiking,
                         rates_inh_spiking=rates_inh_spiking,
                         pct_exc_spiking=pct_exc_spiking,
                         pct_inh_spiking=pct_inh_spiking,
                         num_exc_per_layer=num_exc_per_layer,
                         num_inh_per_layer=num_inh_per_layer,
                         rates_exc_per_layer=rates_exc_per_layer,
                         rates_inh_per_layer=rates_inh_per_layer,
                         rates_exc_spiking_per_layer=rates_exc_spiking_per_layer,
                         rates_inh_spiking_per_layer=rates_inh_spiking_per_layer,
                         pct_exc_spiking_per_layer=pct_exc_spiking_per_layer,
                         pct_inh_spiking_per_layer=pct_inh_spiking_per_layer)
        np.savez(fn, **data_dict)
        print(f'INFO: Rate data saved to "{fn}"')
    return data_dict


def extract_single_cell_rates(circuit_names, sim_configs, save_name, node_set='hex0', t_start=None, t_end=None, save_path=None, load_if_existing=True):
    """Extracts single-cell firing rates."""
    if save_path is None:
        save_path = ''
    fn = os.path.join(save_path, save_name + f'_{node_set}.npz')
    if os.path.exists(fn) and load_if_existing:
        data_dict = dict(np.load(fn))
        print(f'INFO: Cell rate data loaded from "{fn}"')
    else:
        sim = Simulation(sim_configs[circuit_names[0]])
        nodes, _ = _get_population_spikes(sim)

        nids = nodes.ids(node_set)
        nid_mapping = np.full(max(nids) + 1, -1)
        nid_mapping[nids] = np.arange(len(nids))

        spk_rates = []
        for ci, cn in enumerate(tqdm.tqdm(circuit_names)):
            sim = Simulation(sim_configs[cn])
            _, spikes = _get_population_spikes(sim)
            if t_start is None:
                t_start = spikes.spike_report.time_start
            if t_end is None:
                t_end = spikes.spike_report.time_stop

            spk = spikes.get(node_set, t_start=t_start, t_stop=t_end)
            n, cnt = np.unique(spk.values, return_counts=True)

            spk_counts = np.zeros(len(nids), dtype=int)
            spk_counts[nid_mapping[n]] += cnt
            spk_rates.append(spk_counts / (1e-3 * (t_end - t_start)))

        # Save to disc
        data_dict = dict(spk_rates=spk_rates,
                         nids=nids,
                         nid_mapping=nid_mapping,
                         t_start=t_start,
                         t_end=t_end,
                         circuit_names=circuit_names)
        np.savez(fn, **data_dict)
        print(f'INFO: Cell rate data saved to "{fn}"')
    return data_dict


def compute_cell_rate_significance(circuit_names, sim_configs, cell_rates, node_set, save_name, save_path=None, load_if_existing=True):
    if save_path is None:
        save_path = ''
    fn = os.path.join(save_path, save_name + f'_{node_set}.npz')
    if os.path.exists(fn) and load_if_existing:
        data_dict = dict(np.load(fn))
        print(f'INFO: Cell rate p-values loaded from "{fn}"')
    else:
        sim = Simulation(sim_configs[circuit_names[0]])
        nodes, _ = _get_population_spikes(sim)
        nrn_info = nodes.get(node_set, properties=['layer', 'synapse_class'])
        layers = np.unique(nrn_info['layer']).tolist()

        # Compute mean rates [just for consistency check]
        mean_exc_rates_per_layer = np.array([[np.mean(cell_rates[sidx][np.logical_and(nrn_info['layer'] == lay, nrn_info['synapse_class'] == 'EXC')]) for lay in layers] for sidx in range(len(cell_rates))])
        mean_inh_rates_per_layer = np.array([[np.mean(cell_rates[sidx][np.logical_and(nrn_info['layer'] == lay, nrn_info['synapse_class'] == 'INH')]) for lay in layers] for sidx in range(len(cell_rates))])

        # Test 1st..5th order vs. original
        p_rates_exc = []
        p_rates_inh = []
        r_exc_orig = [cell_rates[0][np.logical_and(nrn_info['layer'] == lay, nrn_info['synapse_class'] == 'EXC')] for lay in layers]
        r_inh_orig = [cell_rates[0][np.logical_and(nrn_info['layer'] == lay, nrn_info['synapse_class'] == 'INH')] for lay in layers]
        for cidx in range(1, len(circuit_names)):
            r_exc_manip = [cell_rates[cidx][np.logical_and(nrn_info['layer'] == lay, nrn_info['synapse_class'] == 'EXC')] for lay in layers]
            r_inh_manip = [cell_rates[cidx][np.logical_and(nrn_info['layer'] == lay, nrn_info['synapse_class'] == 'INH')] for lay in layers]
            p_rates_exc.append([wilcoxon(r_exc_orig[lidx], r_exc_manip[lidx]).pvalue if len(r_exc_orig[lidx]) > 0 else np.nan for lidx in range(len(layers))])
            p_rates_inh.append([wilcoxon(r_inh_orig[lidx], r_inh_manip[lidx]).pvalue if len(r_inh_orig[lidx]) > 0 else np.nan for lidx in range(len(layers))])
        p_rates_exc = np.array(p_rates_exc)
        p_rates_inh = np.array(p_rates_inh)

        # Save to disc
        data_dict = dict(p_rates_exc=p_rates_exc,
                         p_rates_inh=p_rates_inh,
                         mean_exc_rates_per_layer=mean_exc_rates_per_layer,
                         mean_inh_rates_per_layer=mean_inh_rates_per_layer,
                         circuit_names=circuit_names)
        np.savez(fn, **data_dict)
        print(f'INFO: Cell rate p-values saved to "{fn}"')
    return data_dict


def extract_overall_psths(spk_all, t_stim, t_psth, bin_size):
    """ PSTHs of firing (!) cells. """
    t_max = np.max(np.diff(t_psth))
    num_bins = np.round(t_max / bin_size).astype(int)
    bins = np.arange(num_bins + 1) * bin_size + t_psth[0]

    psths = []
    for ci in range(len(spk_all)):
        # Extract stimulus spikes
        stim_spikes = []
        for t in t_stim:
            spk = spk_all[ci][np.logical_and(spk_all[ci].index >= t + t_psth[0], spk_all[ci].index < t + t_psth[1])]
            spk.index = spk.index - t # Re-align to stimulus onset
            stim_spikes.append(spk)
        stim_spikes = pd.concat(stim_spikes)

        # Compute PSTHs
        n_all = len(np.unique(stim_spikes.values)) # Number of neurons or fibers
        psth = 1e3 * np.histogram(stim_spikes.index, bins=bins)[0] / (bin_size * n_all * len(t_stim))
        psths.append(psth)

    return psths, bins


def extract_psths(spk_exc, spk_inh, t_stim, t_psth, bin_size):
    """ PSTHs of firing (!) EXC/INH cells. """
    psths_exc, bins = extract_overall_psths(spk_exc, t_stim, t_psth, bin_size)
    psths_inh, _ = extract_overall_psths(spk_inh, t_stim, t_psth, bin_size)

    return psths_exc, psths_inh, bins


def extract_psths_per_layer(spk_exc_per_layer, spk_inh_per_layer, t_stim, t_psth, bin_size):
    psths_exc_per_layer = []
    psths_inh_per_layer = []
    for ci in range(len(spk_exc_per_layer)):
        psth_e_lay = []
        psth_i_lay = []
        num_layers = len(spk_exc_per_layer[ci])
        for lidx in range(num_layers):
            psths_exc, psths_inh, bins = extract_psths([spk_exc_per_layer[ci][lidx]], [spk_inh_per_layer[ci][lidx]], t_stim, t_psth, bin_size)
            psth_e_lay.append(psths_exc[0])
            psth_i_lay.append(psths_inh[0])
        psths_exc_per_layer.append(psth_e_lay)
        psths_inh_per_layer.append(psth_i_lay)

    return psths_exc_per_layer, psths_inh_per_layer, bins


def extract_psths_per_pattern(spk_exc, spk_inh, t_stim, stim_train, t_psth, bin_size):
    """ PSTHs of firing cells per pattern. """
    num_patterns = np.max(stim_train) + 1
    pattern_psths_exc = []
    pattern_psths_inh = []
    for p in range(num_patterns):
        psths_exc, psths_inh, bins = extract_psths(spk_exc, spk_inh, t_stim[stim_train == p], t_psth, bin_size)
        pattern_psths_exc.append(psths_exc)
        pattern_psths_inh.append(psths_inh)
    return pattern_psths_exc, pattern_psths_inh, bins


def _plot_spikes(st, nid_offset, t_start, alpha=1.0, color=None):
    if len(st) == 0:
        return np.nan, 0
    unique_nids = np.unique(st.values)
    nid_mapping = np.full(np.max(st.values) + 1, -1) # Map node IDs to continuous range for plotting
    nid_mapping[unique_nids] = np.arange(len(unique_nids))
    plt.plot(st.index - t_start, nid_mapping[st.values] + nid_offset, '.', markersize=1, markeredgewidth=0, color=color, alpha=alpha)
    y_pos = np.mean(nid_mapping[st.values] + nid_offset)

    return y_pos, len(unique_nids)


def plot_spikes_per_layer(plot_names, spk_exc_per_layer, spk_inh_per_layer, t_start, t_end, t_zero, figsize=(10, 3), save_path=None, alpha=0.5, t_stim=None, stim_train=None, psth_dict={}, no_title=False, dpi=300):
    if t_stim is None or len(t_stim) == 0:
        sim_type = 'Spontaneous'
    else:
        sim_type = 'Stimulus-evoked'
    if stim_train is not None:
        assert t_stim is not None and len(t_stim) == len(stim_train), 'ERROR: Stimulus train mismatch!'
        num_patterns = np.max(stim_train) + 1
        pat_colors = plt.cm.jet(np.linspace(0, 1, num_patterns))

    for ci, cn in enumerate(plot_names):
        nid_offset = 0
        y_ticks = []
        y_lbls = []
        y_cols = []
        plt.figure(figsize=figsize)

        layers = np.arange(1, len(spk_exc_per_layer[ci]) + 1)
        for lidx, lay in enumerate(layers):
            st_e = spk_exc_per_layer[ci][lidx]
            st_i = spk_inh_per_layer[ci][lidx]

            y, n = _plot_spikes(st_e, nid_offset, t_zero, color=COLOR_EXC, alpha=alpha)
            if np.isfinite(y):
                y_ticks.append(y)
                y_lbls.append(f'L{lay}-EXC')
                y_cols.append(COLOR_EXC)
            nid_offset += n

            y, n = _plot_spikes(st_i, nid_offset, t_zero, color=COLOR_INH, alpha=alpha)
            if np.isfinite(y):
                y_ticks.append(y)
                y_lbls.append(f'L{lay}-INH')
                y_cols.append(COLOR_INH)
            nid_offset += n

        plt.yticks(y_ticks, y_lbls, fontsize=4)
        for i, lbl in enumerate(plt.gca().get_yticklabels()):
             lbl.set_color(y_cols[i])
        plt.gca().invert_yaxis()
        fig_title = f'{sim_type} activity ({plot_names[ci]})'
        if not no_title:
            plt.title(fig_title, fontweight='bold')
        plt.xlim([t_start, t_end])
        plt.xlabel('Time (ms)')
        if t_stim is not None and len(t_stim) > 0:
            plt.ylim(plt.ylim())
            if stim_train is None:
                plt.vlines(t_stim - t_zero, *plt.ylim(), 'k', alpha=0.75, label='Stimulus')
            else:
                for p in range(num_patterns):
                    plt.vlines(t_stim[stim_train == p] - t_zero, *plt.ylim(), color=pat_colors[p, :], alpha=0.75, label=f'P{p}')
            plt.legend(loc='lower left', bbox_to_anchor=[0.0, 1.0], fontsize=6, ncol=1 if stim_train is None else num_patterns, frameon=False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        # Plot PSTHs, if provided
        if len(psth_dict) > 0:
            plt.twinx()
            plt.ylabel('Rate (Hz)')
            bin_size = 20  # (ms)
            for _name, _spk in psth_dict.items():
                psth, bins = extract_overall_psths([_spk[ci]], [0], [0, t_end + t_zero], bin_size)
                bin_centers = bins[:-1] + 0.5 * np.mean(np.diff(bins)) - t_zero
                if _name in RATE_COLOR_DICT:
                    col = RATE_COLOR_DICT[_name]
                else:
                    col = None
                plt.plot(bin_centers, psth[0], color=col, alpha=0.75, lw=1, label=_name, zorder=0)
            plt.gca().spines['top'].set_visible(False)
            plt.legend(loc='lower right', bbox_to_anchor=[1.0, 1.0], fontsize=6, ncol=len(psth_dict), frameon=False)

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(os.path.join(save_path, f'plot_spikes_per_layer__{fig_title.replace(" ", "_").replace("(", "").replace(")", "")}__{t_end:.0f}ms.png'), dpi=dpi)


def plot_per_layer(inp_per_layer, plot_names, ylabel, title, pvals=None, log_y=False, figsize=(10, 3), show_legend=True, lgd_props={'loc': 'lower left'}, save_path=None, dpi=300):
    num_layers = len(inp_per_layer[0])
    circ_colors = plt.cm.jet(np.linspace(0, 1, len(plot_names)))

    if pvals is not None:
        num_tests = np.sum(np.isfinite(pvals))
        alpha_levels = np.power(10.0, [-2, -3, -4]) # Corresponding to *, **, and ***
        alpha_levels_corr = alpha_levels / num_tests # Bonferroni correction for multiple comparisons
        print("ALPHA LEVELS: ", end='')
        for _idx, _alph in enumerate(alpha_levels):
            print(f'{"*" * (_idx + 1)}...{_alph:g} ', end='')

    w = 0.75
    plt.figure(figsize=figsize)
    for ci, cn in enumerate(plot_names):
        plt.bar(np.arange(num_layers) + w * ci / len(plot_names), inp_per_layer[ci], width=w / len(plot_names), color=circ_colors[ci], label=cn)
        if pvals is not None and ci > 0:
            for lidx in range(num_layers):
                if np.isfinite(pvals[ci - 1][lidx]):
                    if np.any(pvals[ci - 1][lidx] < alpha_levels_corr):
                        plt.text(lidx + w * ci / len(plot_names), inp_per_layer[ci][lidx], '*' * np.sum(pvals[ci - 1][lidx] < alpha_levels_corr) + '\n', ha='center', va='center', fontsize=7, fontweight='bold')
                    else:
                        plt.text(lidx + w * ci / len(plot_names), inp_per_layer[ci][lidx], 'n.s.\n', ha='center', va='center', fontsize=7)
    plt.xticks(np.arange(num_layers) + np.mean(w * np.arange(len(plot_names)) / len(plot_names)), [f'L{lidx + 1}' for lidx in range(num_layers)], fontweight='bold')
    plt.xlim([-0.5, num_layers])
    plt.ylabel(ylabel)
    if log_y:
        plt.yscale('log')
    plt.title(title, fontweight='bold')
    if show_legend:
        plt.legend(**lgd_props)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f'plot_per_layer__{title.replace(" ", "_")}__{ylabel.split(" (")[0]}.png'), dpi=dpi)
    plt.show()


def plot_cell_rate_histograms(cell_rates, plot_names, sim_type, figsize=(4, 4), bins=None, show_legend=True, save_path=None, dpi=300):
    circ_colors = plt.cm.jet(np.linspace(0, 1, len(plot_names)))
    if bins is None:
        bins = np.arange(0, np.ceil(np.max(cell_rates)) + 1, 0.5)
    plt.figure(figsize=figsize)
    for ci, cn in enumerate(plot_names):
        cnt, bins = np.histogram(cell_rates[ci], bins=bins)
    #     plt.step(bins, np.hstack((cnt[0], cnt)), where='pre', label=cn, color=circ_colors[ci, :], alpha=0.9, zorder=len(plot_names) - ci)
        plt.plot([np.mean(bins[i : i + 2]) for i in range(len(bins) - 1)], cnt, '-', label=cn, color=circ_colors[ci, :], alpha=0.9, zorder=len(plot_names) - ci)
    plt.yscale('log')
    plt.xlabel('Firing rate (Hz)')
    plt.ylabel('Cell count')
    plt.title(f'{sim_type} rates', fontweight='bold')
    if show_legend:
        plt.legend(loc='upper right', frameon=False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f'single_cell_rate_distributions__{sim_type}.png'), dpi=dpi)
    plt.show()


def plot_psths(psths, bins, plot_names, syn_type, figsize=(4, 4), show_legend=True, lgd_props={'loc': 'upper right'}, save_path=None, dpi=300):
    circ_colors = plt.cm.jet(np.linspace(0, 1, len(plot_names)))
    plt.figure(figsize=figsize)
    for ci, cn in enumerate(plot_names):
        if np.all(np.isnan(psths[ci])):
            _lbl = None
        else:
            _lbl = cn
        # plt.step(bins, np.hstack((psths[ci][0], psths[ci])), where='pre', label=_lbl, color=circ_colors[ci, :], alpha=0.9, zorder=len(plot_names) - ci)
        plt.plot([np.mean(bins[i : i + 2]) for i in range(len(bins) - 1)], psths[ci], '-', label=_lbl, color=circ_colors[ci, :], alpha=0.9, zorder=len(plot_names) - ci)
    plt.xlabel('Time (ms)')
    plt.ylabel('Firing rate (Hz)')
    fig_title = f'{syn_type} PSTHs'
    plt.title(fig_title, fontweight='bold')
    if show_legend:
        plt.legend(**lgd_props)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f'psths__{fig_title.replace(" ", "_").replace("(", "").replace(")", "")}'), dpi=dpi)
    plt.show()


def plot_psths_per_layer(psths_per_layer, bins, plot_names, syn_type, figsize=(4, 4), show_legend=True, lgd_props={'loc': 'upper right'}, save_path=None, dpi=300):
    if isinstance(show_legend, list):
        assert len(show_legend) == len(plot_names), "ERROR: Show legend error!"
    else:
        show_legend = [show_legend] * len(plot_names)
    for ci, cn in enumerate(plot_names):
        plt.figure(figsize=figsize)
        num_layers = len(psths_per_layer[ci])
        for lidx in range(num_layers):
            if np.all(np.isnan(psths_per_layer[ci][lidx])):
                _lbl = None
            else:
                _lbl = f'L{lidx + 1}'
            # plt.step(bins, np.hstack((psths_per_layer[ci][lidx][0], psths_per_layer[ci][lidx])), where='pre', alpha=0.9, label=_lbl)
            plt.plot([np.mean(bins[i : i + 2]) for i in range(len(bins) - 1)], psths_per_layer[ci][lidx], '-', label=_lbl, alpha=0.9)
        plt.xlabel('Time (ms)')
        plt.ylabel('Firing rate (Hz)')
        fig_title = f'{syn_type} PSTHs ({cn})'
        plt.title(fig_title, fontweight='bold')
        if show_legend[ci]:
            plt.legend(**lgd_props)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(os.path.join(save_path, f'psths_per_layer__{fig_title.replace(" ", "_").replace("(", "").replace(")", "")}'), dpi=dpi)
        plt.show()


def plot_psths_per_pattern(psths, bins, plot_names, syn_type, stim_train, figsize=(12, 4), show_legend=True, lgd_props={'loc': 'upper right'}, save_path=None, dpi=300):
    num_patterns = len(psths)
    circ_colors = plt.cm.jet(np.linspace(0, 1, len(plot_names)))
    plt.figure(figsize=figsize)
    for p in range(num_patterns):
        plt.subplot(1, num_patterns, p + 1)
        for ci, cn in enumerate(plot_names):
            if np.all(np.isnan(psths[p][ci])):
                _lbl = None
            else:
                _lbl = cn
            plt.plot([np.mean(bins[i : i + 2]) for i in range(len(bins) - 1)], psths[p][ci], '-', label=_lbl, color=circ_colors[ci, :], alpha=0.9, zorder=len(plot_names) - ci)
        plt.xlabel('Time (ms)')
        plt.ylabel('Firing rate (Hz)')
        plt.title(f'Pattern {p} (N={np.sum(np.array(stim_train) == p)})')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
    fig_title = f'{syn_type} PSTHs'
    plt.suptitle(fig_title, fontweight='bold')
    if show_legend:
        plt.legend(**lgd_props)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f'pattern_psths__{fig_title.replace(" ", "_").replace("(", "").replace(")", "")}'), dpi=dpi)
    plt.show()
