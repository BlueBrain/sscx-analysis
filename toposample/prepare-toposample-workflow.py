#!/usr/bin/env python
import os

import bluepy
import simProjectAnalysis as spa
import pandas
import numpy
import tqdm

from toposample import config
from toposample import indexing

from scipy import sparse


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


def prepare_common_config(out_dir):
    import json
    import shutil

    fn_in = os.path.join(os.path.split(__file__)[0], "_files", "common_config.json")
    # TODO: To make it work in workflow it will probably have to read directly from the repository url?
    with open(fn_in, "r") as fid:
        data = json.load(fid)

    cfg_root = os.path.join(out_dir, "config")
    if not os.path.isdir(cfg_root):
        os.makedirs(cfg_root)
    with open(os.path.join(cfg_root, "common_config.json"), "w") as fid:
        json.dump(data, fid, indent=2)

    data["paths"]["inputs"]["dir"] = os.path.abspath(os.path.join(cfg_root, data["paths"]["inputs"]["dir"]))
    data["paths"]["analyzed"]["dir"] = os.path.abspath(os.path.join(cfg_root, data["paths"]["analyzed"]["dir"]))
    data["paths"]["config"]["dir"] = os.path.abspath(os.path.join(cfg_root, data["paths"]["config"]["dir"]))
    data["paths"]["other"]["dir"] = os.path.abspath(os.path.join(cfg_root, data["paths"]["other"]["dir"]))

    for dir in [data["paths"]["inputs"]["dir"], data["paths"]["analyzed"]["dir"],
                data["paths"]["config"]["dir"], data["paths"]["other"]["dir"]]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    return data


def initial_setup(sims):
    import hashlib
    conditions = sims.index.names
    out = []
    circuit_dict = {}
    for cond, path in sims.iteritems():
        cond_dict = dict(zip(conditions, cond))
        value = bluepy.Simulation(path)
        cfg = value.circuit.config.copy()
        cfg.pop("targets", None)
        circ_hash = hashlib.md5(str(cfg).encode("UTF-8")).hexdigest()
        if circ_hash not in circuit_dict:
            circuit_dict[circ_hash] = value.circuit
        out.append(spa.ResultsWithConditions(value, circuit_hash=circ_hash,
                                             **cond_dict))
    return spa.ConditionCollection(out), circuit_dict


def read_spikes(sim):
    raw_spikes = sim.spikes.get()
    return numpy.vstack([raw_spikes.index.values, raw_spikes.values]).transpose()


def read_time_windows(sim):
    """
    :param sim: bluepy.Simulation
    :return: stim_ids, a numpy.array of length L, with integers between [0, n_stim]. Specifying the identity of stimuli
             t_wins, a numpy.array of shape L x 2, where t_wins[i, 0] and t_wins[i, 1] are the start and end time of
             the response to stimulus stim_ids[i]
    """
    t_wins = numpy.array([[0.0, float(sim.config.Run["Duration"])]])
    stim_ids = numpy.array([0])

    json_fn = os.path.join(sim.config.Run["OutputRoot"], "BlueConfig.json")
    if os.path.exists(json_fn):
        import json
        with open(json_fn, "r") as fid:
            data = json.load(fid)
        if "stim_t_wins" in data:
            t_wins = numpy.array(data["stim_t_wins"])
            assert t_wins.ndim == 2 and t_wins.shape[1] == 2
            stim_ids = numpy.array(data.get("stim_ids", [0] * len(t_wins)))
            assert len(stim_ids) == len(t_wins)
    return stim_ids, t_wins


def concatenate_sims(lst_spikes_stims):
    spikes, stims = zip(*lst_spikes_stims)
    stim_ids, t_wins = zip(*stims)

    t_win_width = numpy.diff(numpy.vstack(t_wins), axis=1)
    assert len(numpy.unique(t_win_width)) == 1, "All time windows must be equal size!"
    t_win_width = t_win_width[0]

    out_spikes = []
    t = 0.0
    # TODO: Someone review this logic....
    for spk, t_win in zip(spikes, t_wins):
        for a, b in t_win:
            in_win = (spk[:, 0] >= a) & (spk[:, 0] < b)
            spk[in_win, 0] = spk[in_win, 0] - a + t
            out_spikes.append(spk[in_win])
            t += t_win_width
    out_stims = numpy.hstack(stim_ids)
    out_spikes = numpy.vstack(out_spikes)
    return out_stims, out_spikes, t_win_width


def get_neuron_info(circ, group, sim_target):
    gids = numpy.intersect1d(circ.cells.ids(group), circ.cells.ids(sim_target))
    return circ.cells.get(group=gids, properties=["mtype", "layer", "x", "y", "z"])


def get_con_mat(circ, neuron_info, lst_projections=[]):
    conv = indexing.GidConverter(neuron_info)
    indptr = [0]
    indices = []

    con = [circ.connectome]
    for proj in lst_projections:
        con.append(circ.projection(proj))

    for gid in tqdm.tqdm(neuron_info.index):
        aff_gids = numpy.hstack([_con.afferent_gids(gid) for _con in con])
        aff = conv.indices(numpy.intersect1d(aff_gids, neuron_info.index))
        indices.extend(aff)
        indptr.append(len(indices))
    data = numpy.ones_like(indices, dtype=bool)
    return sparse.csc_matrix((data, indices, indptr), shape=(len(neuron_info), len(neuron_info)))


def write_structural_info(neuron_info, con_mat, common_cfg):
    inputs = common_cfg["paths"]["inputs"]
    fn_info = os.path.join(inputs["dir"], inputs["files"]["neuron_info"])
    fn_con = os.path.join(inputs["dir"], inputs["files"]["adjacency_matrix"])

    neuron_info.to_pickle(fn_info)
    parse.save_npz(fn_con, con_mat)


def write_simulation_results(stims, spikes, common_cfg):
    inputs = common_cfg["paths"]["inputs"]
    fn_spikes = os.path.join(inputs["dir"], inputs["files"]["raw_spikes"])
    fn_stims = os.path.join(inputs["dir"], inputs["files"]["stimuli"])

    numpy.save(fn_spikes, spikes)
    numpy.save(fn_stims, stims)


def main():
    sims, cfg = parse_arguments()
    out_fn = cfg.pop("output_root")

    common_cfg = prepare_common_config(out_fn)
    sim_struc, circuit_dict = initial_setup(sims)

    assert len(circuit_dict) == 1, "Simulations need to use the same circuit!"
    circuit = list(circuit_dict.values())[0]
    sim_struc.remove_label("circuit_hash")

    for k, v in cfg.get("filters", {}).items():
        sim_struc = sim_struc.filter(k, v)

    sim_target = numpy.unique(sim_struc.map(lambda sim: sim.target).get())
    assert len(sim_target) == 1, "All simulations must use the same CircuitTarget!"
    sim_target = sim_target[0]

    neuron_info = get_neuron_info(circuit, cfg["base_target"], sim_target)
    con_mat = get_con_mat(circuit, neuron_info, cfg.get("projections", []))
    write_structural_info(neuron_info, con_mat, common_cfg)

    spikes = sim_struc.map(read_spikes)
    stim_t_wins = sim_struc.map(read_time_windows)

    spikes_n_stims = spikes.extended_map(lambda *args: tuple(args), [stim_t_wins], iterate_inner=True)
    spikes_n_stims = spikes_n_stims.pool(cfg["pool_conditions"], func=concatenate_sims)

    assert len(spikes_n_stims.get()) == 1, "Need to pool or filter more?"
    stims, spikes, t_win_width = spikes_n_stims.get2()
    # TODO: Write t_win_width parameter to the "extract_time_windows" config.
    write_simulation_results(stims, spikes, common_cfg)
