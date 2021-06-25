import os

import bluepy
import simProjectAnalysis as spa
import pandas
import numpy
import tqdm


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
        circ_hash = hashlib.md5(str(sim.circuit.config).encode("UTF-8")).hexdigest()
        if circ_hash not in circuit_dict:
            circuit_dict[circ_hash] = sim.circuit
        out.append(spa.ResultsWithConditions(value, circuit_hash=circ_hash,
                                             **cond_dict))
    return spa.ConditionCollection(out), circuit_dict


def read_spikes(sim):
    raw_spikes = sim.spikes.get()
    return numpy.vstack([raw_spikes.index.values, raw_spikes.values]).transpose()


def create_gid_dict(circ_dict, type_definitions, base_target):
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
            type_gids[type_lbl] = cells.index[valid].values
        out[circ_hash] = type_gids
    return out


def split_spikes_factory(target_gid_dicts):
    def split_spikes(circ_hashes, spks_in):
        for circ, spks in zip(circ_hashes, spks_in):
            gid_dict = target_gid_dicts[circ]
            for label, gids in tqdm.tqdm(gid_dict.items()):
                valid = numpy.in1d(spks[:, 1], gids)
                yield spks[valid], {"circuit_hash": circ, "neuron_type": label}
    return split_spikes


def histogram_factory(t_bins):
    def normalized_histogram(x):
        h = numpy.histogram(x[:, 0], bins=t_bins)[0]
        h = (h - numpy.mean(h)) / numpy.std(h)
        return h
    return normalized_histogram


def spike_time_bins_factory(t_bins):
    def spike_time_bins(x):
        spk_bins = numpy.digitize(x[:, 0], bins=t_bins) -1
        valid = (spk_bins >= 0) & (spk_bins < (len(t_bins) - 1))
        out = numpy.vstack([spk_bins[valid], x[valid, 1]]).transpose()
        return out.astype(int)
    return spike_time_bins


def subsample_factory(amount, seed=None):
    numpy.random.seed(seed)

    def subsample(spikes):
        u_gids = numpy.unique(spikes[:, 1])
        if isinstance(amount, float):
            to_pick = int(amount * len(u_gids))
        else:
            to_pick = amount
        picked = numpy.random.choice(u_gids, to_pick, replace=False)
        valid = numpy.in1d(spikes[:, 1], picked)
        return spikes[valid]
    return subsample


def correlation_payload(A, B):
    L = len(B)
    assert len(A) == L
    overlap_sz = L - numpy.abs(numpy.arange(L) - L/2)
    return numpy.convolve(A[-1::-1], B, 'same') / overlap_sz


def correlation_factory(t_win, binsize):
    def correlations(lst_type_names, lst_hists):
        for i in range(len(lst_type_names)):
            for j in range(i, len(lst_type_names)):
                lbl1 = lst_type_names[i]
                lbl2 = lst_type_names[j]
                res = correlation_payload(lst_hists[i], lst_hists[j])
                x = binsize * (numpy.arange(len(res)) - (len(res) / 2))
                valid = (x >= t_win[0]) & (x <= t_win[1])
                yield pandas.Series(res[valid], index=x[valid]), {"type1": lbl1, "type2": lbl2}
                if i != j:
                    yield pandas.Series(res[valid][-1::-1], index=x[valid]), {"type1": lbl2, "type2": lbl1}
    return correlations


def sta_factory(t_win, binsize, return_type):
    if return_type not in ["individuals", "mean"]:
        raise ValueError("Unknown return type: {0}".format(return_type))
    o1 = int(t_win[0] / binsize)
    o2 = int(t_win[1] / binsize) + 1
    o_idx = numpy.arange(t_win[0], t_win[1] + binsize, binsize)

    def sta(spikes, hist):
        valid = ((spikes[:, 0] + o1) >= 0) & ((spikes[:, 0] + o2) <= len(hist))
        spikes = spikes[valid]
        spikes = spikes[numpy.argsort(spikes[:, 1])]
        idxx = numpy.hstack([0, numpy.nonzero(numpy.diff(spikes[:, 1]) > 0)[0] + 1, len(spikes)])

        if return_type == "individuals":
            out = {}
            for a, b in tqdm.tqdm(zip(idxx[:-1], idxx[1:])):
                out[spikes[a, 1]] = numpy.vstack([hist[(i+o1):(i+o2)] for i in spikes[a:b, 0]]).mean(axis=0)
            return pandas.DataFrame(out, index=o_idx)
        elif return_type == "mean":
            out = []
            for a, b in tqdm.tqdm(zip(idxx[:-1], idxx[1:])):
                out.append(numpy.vstack([hist[(i + o1):(i + o2)] for i in spikes[a:b, 0]]).mean(axis=0))
            out = numpy.vstack(out).mean(axis=0)
            return pandas.Series(out, index=o_idx)

        # Option for returning mean over spikes
        # out = numpy.vstack([hist[(i + o1):(i + o2)] for i in spikes[:, 0]]).mean(axis=0)
        # return pandas.Series(out, index=o_idx)
    return sta


def main():
    sims, cfg = parse_arguments()
    out_fn = cfg.pop("output_root")
    data, circ_dict = initial_setup(sims)

    # Specify in config to apply the analysis only to a subset of simulation conditions
    for k, v in cfg.get("condition_filter", {}).items():
        data = data.filter(**dict([(k, v)]))
    spikes = data.map(read_spikes)

    # Get parameters for spike binning from config
    binsize = cfg.get("binsize", 2.0)
    t_start = cfg.get("t_start", 0.0)
    t_end = cfg.get("t_end", numpy.max(spikes.map(lambda x: x[:, 0].max()).get()) + 0.025)
    t_bins = numpy.arange(t_start, t_end + binsize, binsize)

    # Get definition of neuron classes from config
    target_gid_dicts = create_gid_dict(circ_dict, cfg["neuron_classes"], cfg.get("base_target", "Mosaic"))

    # Split spike results into individual results for the different classes
    spikes_per_target = spikes.transform(["circuit_hash"], split_spikes_factory(target_gid_dicts), xy=True)

    # Create normalized histograms
    hists_per_target = spikes_per_target.map(histogram_factory(t_bins))

    ana_mode = cfg.get("correlation", {})
    corr_win = ana_mode.get("t_win", [-250, 250])
    corr_func = ana_mode.get("type")
    # Fast mode: Simply convolve the normalized spiking histograms
    if corr_func == "convolution":
        correlation_res = hists_per_target.transform(["neuron_type"], correlation_factory(corr_win, binsize), xy=True)
    # Slower: Calculate spike triggered average of normalized histograms
    elif corr_func == "sta":
        binned_per_target = spikes_per_target.map(spike_time_bins_factory(t_bins))
        if "subsample" in cfg:
            binned_per_target = binned_per_target.map(subsample_factory(cfg["subsample"]))
        correlation_res = binned_per_target.extended_map(sta_factory(corr_win, binsize,
                                                                     ana_mode.get("return", "individuals")),
                                                         [hists_per_target],
                                                         ignore_conds=["neuron_type"], iterate_inner=True)
    else:
        raise ValueError("Unknown correlation mode: {0}".format(corr_func))

    correlation_res.add_label("binsize", binsize)
    correlation_res.add_label("t_start", t_start)
    correlation_res.add_label("t_end", t_end)

    correlation_res.to_pandas().agg(lambda x: x).to_pickle(os.path.join(out_fn, "results.pkl"))


if __name__ == "__main__":
    main()
