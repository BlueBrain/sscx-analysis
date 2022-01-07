"""
Uses `conntility` to get (and save) time dependent weighted connectomes
author: András Ecker, last update: 01.2022
"""

import os
from utils import load_mapping_df, load_sim_paths
from bluepy import Simulation
from conntility.connectivity import TimeDependentMatrix


if __name__ == "__main__":
    project_name = "e0fbb0c8-07a4-49e0-be7d-822b2b2148fb"
    report_cfg = {"t_start": 0.0, "t_end": 62000.0, "t_step": 1000.0, "report_name": "rho",
                  "static_prop_name": "rho0_GB"}
    load_cfg = {"loading": {"base_target": "hex_O1", "properties": ["x", "y", "z", "mtype", "synapse_class"],
                            "atlas": [{"data": "[PH]y", "properties": ["[PH]y"]}]},
                "filtering": [{"column": "synapse_class", "value": "EXC"}]}

    mapping_df = load_mapping_df()
    sim_paths = load_sim_paths(project_name)
    for _, sim_path in sim_paths.iteritems():
        sim = Simulation(sim_path)
        T = TimeDependentMatrix.from_report(sim, report_cfg, load_cfg, mapping_df)
        T.to_h5(os.path.join(os.path.split(sim_path)[0], "td_edges_%s.h5" % report_cfg["report_name"]))

