#!/usr/bin/env python


import os

import bluepy
import simProjectAnalysis as spa
import pandas
import numpy
import tqdm

from scipy import sparse


circuit_rel_entries = ['cells', 'morphologies', 'emodels', 'connectome', 'atlas', 'projections']


def parse_arguments():
    import sys
    import json
    args = sys.argv[1:]
    if len(args) < 2:
        print("""Usage:
        {0} simulations.pkl config_file.json
        For details, see included README.txt
        """.format(__file__))
        sys.exit(2)
    with open(args[1], "r") as fid:
        cfg = json.load(fid)
    sims = pandas.read_pickle(args[0])
    return sims, cfg


def initial_setup(sims):
    import hashlib
    conditions = sims.index.names
    out = []
    circuit_dict = {}
    for cond, path in sims.iteritems():
        cond_dict = dict(zip(conditions, cond))
        value = bluepy.Simulation(path)
        circ_cfg = dict([(k, value.circuit.config.get(k, "__NONE__")) for k in circuit_rel_entries])
        circ_hash = hashlib.md5(str(circ_cfg).encode("UTF-8")).hexdigest()
        if circ_hash not in circuit_dict:
            circuit_dict[circ_hash] = value.circuit
        out.append(spa.ResultsWithConditions(value, circuit_hash=circ_hash,
                                             **cond_dict))
    return spa.ConditionCollection(out), circuit_dict


def pool_spikes_factory(t_win):

    def pool_spikes(lst_sims):
        offset = 0.0
        out = []
        for sim in lst_sims:
            if t_win is None:
                t_win_use = [sim.t_start, sim.t_end]
            else:
                t_win_use = t_win
            win_dur = t_win_use[1] - t_win_use[0]
            raw_spikes = sim.spikes.get()
            valid = (raw_spikes.index >= t_win_use[0]) & (raw_spikes.index < t_win_use[1])
            raw_spikes = raw_spikes.loc[valid]
            out.append(numpy.vstack([raw_spikes.index.values - t_win_use[0] + offset,
                                     raw_spikes.values]).transpose())
            offset += win_dur
        return numpy.vstack(out), [0.0, offset]
    return pool_spikes


def create_gid_dict(circ_dict, type_definitions, base_target, max_per_population=None):
    out = {}
    properties = numpy.unique(numpy.hstack([list(x.keys()) for x in
                                            type_definitions.values()]))

    for circ_hash, circ in circ_dict.items():
        cells = circ.cells.get(group=base_target, properties=properties)

        def evaluate_a_property(prop_name, valid_vals):
            valid = numpy.zeros(len(cells))
            for v in valid_vals:
                valid = valid | (cells[prop_name] == v)
            return valid

        type_gids = {}
        for type_lbl, type_fltrs in type_definitions.items():
            valid = numpy.ones(len(cells))
            for prop, valid_vals in type_fltrs.items():
                valid = valid & evaluate_a_property(prop, valid_vals)
            gids = cells.index[valid].values
            if max_per_population is not None:
                if len(gids) > max_per_population:
                    gids = numpy.random.choice(gids, int(max_per_population), replace=False)
            gids.sort()
            type_gids[type_lbl] = gids
        out[circ_hash] = type_gids
    return out


def split_spikes_factory(target_gid_dicts):
    def split_spikes(circ_fns, spks_in):
        for circ, spks in zip(circ_fns, spks_in):
            spks, spk_win = spks
            gid_dict = target_gid_dicts[circ]
            for label, gids in tqdm.tqdm(gid_dict.items()):
                valid = numpy.in1d(spks[:, 1], gids)
                yield (spks[valid], spk_win, gids), {"circuit_config": circ, "neuron_type": label}
    return split_spikes


def test_time_windows_consistent(lst_spk_data):
    _, t_wins, _ = zip(*lst_spk_data)
    t_wins = list(t_wins)
    for t_win in t_wins[1:]:
        if t_win != t_wins[0]:
            raise ValueError("Inconsistent time windows")


def spikes_to_binned_matrix(spk_data, t_step, shuffle=False, randomize_t=False, randomize_gid=False):
    spks, spk_win, gids = spk_data
    t_bins = numpy.arange(spk_win[0], spk_win[1] + t_step, t_step)
    gid_bins = numpy.hstack([gids, gids[-1] + 1])

    t_id = numpy.digitize(spks[:, 0], bins=t_bins) - 1
    cell_id = numpy.digitize(spks[:, 1], bins=gid_bins) - 1

    if shuffle:
        t_id = numpy.random.permutation(t_id)
    if randomize_t:
        t_id = numpy.random.randint(len(t_bins) - 1, size=len(t_id))
    if randomize_gid:
        cell_id = numpy.random.randint(len(gids), size=len(cell_id))

    M = sparse.coo_matrix((numpy.ones_like(cell_id, dtype=float), # TODO: Move todense into the next function?
                           (cell_id, t_id)),
                          shape=(len(gids), len(t_bins) - 1)).todense()
    M = numpy.array(M - M.mean(axis=1)) # cell x time
    return M


def pair_correlation_payload(m1, m2, nrmlz1, nrmlz2, corr_bins, disable_diagonal=False):
    raw = numpy.dot(m1, m2.transpose())
    nrmlz = numpy.sqrt(numpy.dot(nrmlz1, nrmlz2.transpose())) * m1.shape[1]
    ret = raw / nrmlz
    if disable_diagonal:
        h = numpy.histogram(ret[numpy.triu_indices_from(ret, 1)], bins=corr_bins, density=True)[0]
    else:
        h = numpy.histogram(ret.flat, bins=corr_bins, density=True)[0]
    return h


def pair_correlation_factory(t_step=20.0):
    corr_bins = numpy.linspace(-1, 1, 201) # TODO: Configurable


    def pair_correlations(lst_nrn_types, lst_spk_data):
        test_time_windows_consistent(lst_spk_data)
        print("Calculating binned spiking matrices...")
        lst_M = [spikes_to_binned_matrix(spk_data, t_step) for spk_data in lst_spk_data]
        lst_nrmlz = [m.var(axis=1, keepdims=True) for m in lst_M]
        i = 0
        for nrn_type1, m1, nrmlz1 in zip(lst_nrn_types, lst_M, lst_nrmlz):
            disable_diagonal = True
            for nrn_type2, m2, nrmlz2 in zip(lst_nrn_types[i:], lst_M[i:], lst_nrmlz[i:]):
                print("\tCorrelating {0} and {1}".format(str(nrn_type1), str(nrn_type2)))
                v = pair_correlation_payload(m1, m2, nrmlz1, nrmlz2, corr_bins, disable_diagonal=disable_diagonal)
                disable_diagonal = False
                yield v, {"run": "data", "type_1": nrn_type1, "type_2": nrn_type2}
            i += 1

        print("Calculating randomized binned spiking matrices...") # TODO: Configure number and type of control
        lst_M = [spikes_to_binned_matrix(spk_data, t_step, randomize_t=True) for spk_data in lst_spk_data]
        lst_nrmlz = [m.var(axis=1, keepdims=True) for m in lst_M]
        i = 0
        for nrn_type1, m1, nrmlz1 in zip(lst_nrn_types, lst_M, lst_nrmlz):
            disable_diagonal = True
            for nrn_type2, m2, nrmlz2 in zip(lst_nrn_types[i:], lst_M[i:], lst_nrmlz[i:]):
                print("\tCorrelating {0} and {1}".format(str(nrn_type1), str(nrn_type2)))
                v = pair_correlation_payload(m1, m2, nrmlz1, nrmlz2, corr_bins, disable_diagonal=disable_diagonal)
                disable_diagonal = False
                yield v, {"run": "randomized", "type_1": nrn_type1, "type_2": nrn_type2}
            i += 1
    return pair_correlations


def main():
    sims, cfg = parse_arguments()
    out_fn = cfg.pop("output_root")
    data, circ_dict = initial_setup(sims)

    # Specify in config to apply the analysis only to a subset of simulation conditions
    for k, v in cfg.get("condition_filter", {}).items():
        data = data.filter(**dict([(k, v)]))
    # Pool spikes over specified condition by concatenating
    spk_pool_cond = cfg.get("pool_spikes", {}).get("conditions", [])
    spk_pool_twin = cfg.get("pool_spikes", {}).get("t_win", None)
    spikes = data.pool(spk_pool_cond, func=pool_spikes_factory(spk_pool_twin))
    if "circuit_hash" not in spikes.conditions(): # "pool" gets rid of singleton dimensions...
        assert len(circ_dict) == 1
        spikes.add_label("circuit_hash", list(circ_dict)[0])

    # Get parameters for spike binning from config
    binsize = cfg.get("binsize", 20.0)

    # Get definition of neuron classes from config
    target_gid_dicts = create_gid_dict(circ_dict, cfg["neuron_classes"], cfg.get("base_target", "Mosaic"),
                                       max_per_population=cfg.get("max_per_population", None))

    # Split spike results into individual results for the different classes
    spikes_per_target = spikes.transform(["circuit_hash"], split_spikes_factory(target_gid_dicts), xy=True)

    # Execute...
    correlation_res = spikes_per_target.transform(["neuron_type"], pair_correlation_factory(binsize), xy=True)
    correlation_res.add_label("time_bin_size", binsize)

    correlation_res.to_pandas().agg(lambda x: x).to_pickle(os.path.join(out_fn, "results.pkl"))


if __name__ == "__main__":
    main()
