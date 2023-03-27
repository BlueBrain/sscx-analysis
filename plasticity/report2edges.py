"""
Loads synapse report and writes new edge file based on the `rho` values saved at the last time step.
That is, it updates initial `Use0_TM` and `gmax0_AMPA` values based on the rhos
but since it doesn't recalculate `c_pre` it can't update `theta_d` and `theta_p`
thus the resulting edge file should only be used in non-plastic simulations (aka. as a "frozen circuit")
author: András Ecker, last update: 03.2023
"""

import os
import shutil
import h5py
from bluepy import Simulation

import utils

CIRCUITS_DIR = "/gpfs/bbp.cscs.ch/project/proj96/circuits/"


def get_edgef_names(project_name, sim_id, sim_path):
    """Get old and new edge file names"""
    # seems a bit useless to load the `Simulation` object for this...
    orig_edge_fname = Simulation(sim_path).config["Run_Default"]["nrnPath"]
    new_edge_fname = os.path.join(CIRCUITS_DIR, project_name, str(sim_id), "edges.sonata")
    return orig_edge_fname, new_edge_fname


def _get_population(h5f_name):
    """Gets population from sonata edge file"""
    with h5py.File(h5f_name, "r") as h5f:
        populations = list(h5f["edges"])
        if len(populations) > 1:
            raise RuntimeError("Multiple populations in the file")
    return populations[0]


def get_property(h5f_name, edge_property):
    """Gets `edge_property` array from sonata edge file"""
    population = _get_population(h5f_name)
    with h5py.File(h5f_name, "r") as h5f:
        h5f_group = h5f["edges/%s/0" % population]
        if edge_property not in h5f_group:
            raise RuntimeError("%s not in the edge file properties" % edge_property)
        return h5f_group[edge_property][:]


def update_population_properties(h5f_name, edge_properties, force=False):
    """Update sonata edge population with new properties"""
    assert isinstance(edge_properties, dict)
    population = _get_population(h5f_name)
    with h5py.File(h5f_name, "r+") as h5f:
        h5f_group = h5f["edges/%s/0/" % population]
        size = h5f_group["%s" % list(h5f_group)[0]].size
        exists = set(h5f_group) & set(edge_properties.keys())
        if not force and exists:
            raise RuntimeError("Some properties already exist: %s." % exists)
        # add edge properties
        for name, values in edge_properties.items():
            assert len(values) == size
            if force and name in h5f_group:
                del h5f_group[name]
            h5f_group.create_dataset(name, data=values)


def main(project_name, sim_id, force_np=False):
    # load report, get depressed and potentiated syn. idx.
    sim_path = utils.load_sim_paths(project_name).loc[sim_id]
    _, data = utils.get_all_synapses_at_t(sim_path, "rho", t=-1)
    syn_idx = data.index.to_numpy()
    syn_idx_d = data[data["rho"] <= 0.5].index.to_numpy()
    syn_idx_p = data[data["rho"] > 0.5].index.to_numpy()
    del data
    # create a copy of the original edge file (to be modified below)
    orig_edge_fname, new_edge_fname = get_edgef_names(project_name, sim_id, sim_path)
    utils.ensure_dir(os.path.dirname(new_edge_fname))
    shutil.copyfile(orig_edge_fname, new_edge_fname)
    # efficacy
    orig_edge_property = get_property(orig_edge_fname, "rho0_GB")
    new_edge_property = orig_edge_property.copy()
    new_edge_property[syn_idx_d] = 0
    new_edge_property[syn_idx_p] = 1
    del orig_edge_property
    update_population_properties(new_edge_fname, {"rho0_GB": new_edge_property}, True)
    # release probability
    orig_edge_properties = {edge_property: get_property(orig_edge_fname, edge_property)
                            for edge_property in ["Use_d_TM", "Use_p_TM", "u_syn"]}
    new_edge_property = orig_edge_properties["u_syn"].copy()
    new_edge_property[syn_idx_d] = orig_edge_properties["Use_d_TM"][syn_idx_d]
    new_edge_property[syn_idx_p] = orig_edge_properties["Use_p_TM"][syn_idx_p]
    del orig_edge_properties
    update_population_properties(new_edge_fname, {"u_syn": new_edge_property}, True)
    # max (AMPA) conductance (NMDA will be set based on the AMPA one at sim. init.)
    orig_edge_properties = {edge_property: get_property(orig_edge_fname, edge_property)
                            for edge_property in ["gmax_d_AMPA", "gmax_p_AMPA", "conductance"]}
    new_edge_property = orig_edge_properties["conductance"].copy()
    new_edge_property[syn_idx_d] = orig_edge_properties["gmax_d_AMPA"][syn_idx_d]
    new_edge_property[syn_idx_p] = orig_edge_properties["gmax_p_AMPA"][syn_idx_p]
    del orig_edge_properties
    update_population_properties(new_edge_fname, {"conductance": new_edge_property}, True)
    # plasticity thresholds
    if force_np:  # force the new edge file to be non-plastic (by setting negative thresholds)
        orig_edge_properties = {edge_property: get_property(orig_edge_fname, edge_property)
                                for edge_property in ["theta_d", "theta_p"]}
        new_edge_property = orig_edge_properties["theta_d"].copy()
        new_edge_property[syn_idx] = -1.
        new_edge_properties = {"theta_d": new_edge_property}
        new_edge_property = orig_edge_properties["theta_p"].copy()
        new_edge_property[syn_idx] = -1.
        new_edge_properties["theta_p"] = new_edge_property
        del orig_edge_properties
        update_population_properties(new_edge_fname, new_edge_properties, True)


if __name__ == "__main__":
    project_name, sim_id = "3e3ef5bc-b474-408f-8a28-ea90ac446e24", 1
    main(project_name, sim_id)




