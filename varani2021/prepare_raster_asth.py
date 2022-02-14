# -*- coding: utf-8 -*-
"""
Prepares lookups for raster asthetics
last modified: András Ecker 06.2021
"""

import pickle
import numpy as np
import pandas as pd
from bluepy import Circuit
from bluepy.enums import Cell

RED = "#e32b14"
BLUE = "#3271b8"
GREEN = "#67b32e"
ORANGE = "#c9a021"


def group_inh_gids(c, target):
    """Groups inhibitory gids based on 3 main markers"""
    inh_gids = {marker: [] for marker in ["PV", "Sst", "5HT3aR"]}
    inh_gids["PV"].extend(c.cells.ids({"$target": target, Cell.MTYPE: {"$regex": "L(23|4|5|6)_(LBC|NBC|CHC)"}}))
    inh_gids["Sst"].extend(c.cells.ids({"$target": target, Cell.MTYPE: {"$regex": "L(23|4|5|6)_MC"}}))
    inh_gids["Sst"].extend(c.cells.ids({"$target": target, Cell.MTYPE: {"$regex": "L(23|4|5|6)_(DBC|BTC)"},
                           Cell.ETYPE: "cACint"}))
    inh_gids["5HT3aR"].extend(c.cells.ids({"$target": target, Cell.MTYPE: {"$regex": "L(23|4|5|6)_(DBC|BTC)"},
                              Cell.ETYPE: ["bNAC", "bAC", "cNAC", "dNAC", "cIR", "bIR", "bSTUT"]}))
    inh_gids["5HT3aR"].extend(c.cells.ids({"$target": target, Cell.MTYPE: {"$regex": "L(23|4|5|6)_(SBC|BP|NGC)"}}))
    inh_gids["5HT3aR"].extend(c.cells.ids({"$target": target, Cell.LAYER: 1}))
    return inh_gids


def get_raster_asthetics(c, depths):
    """Groups gids and creates lookups for y-coordinate, color, etc. for raster plots"""
    gids = group_inh_gids(c, "hex_O1")
    gids["PC"] = c.cells.ids({"$target": "hex_O1", Cell.SYNAPSE_CLASS: "EXC"})

    ys = {}
    colors = {}
    groups = {}
    types = ["PC", "PV", "Sst", "5HT3aR"]
    cols = [RED, BLUE, GREEN, ORANGE]
    for type_, color in zip(types, cols):
        for i, gid in enumerate(gids[type_]):
            ys[gid] = depths.at[gid, "depth"]
            colors[gid] = color
            groups[gid] = type_
    yticks = [depths.min()["depth"], depths.max()["depth"]]
    yticklables = ["%.2f" % ytick for ytick in yticks]
    for layer in range(1, 7):
        layer_gids = c.cells.ids({"$target": "hex_O1", Cell.LAYER: layer})
        yticks.append(depths.loc[np.isin(depths.index, layer_gids), "depth"].mean())
        yticklables.append("Layer%s\n(%i)" % (layer, len(layer_gids)))
    return {"ys": ys, "colors": colors, "groups": groups,
            "yticks": yticks, "yticklabels": yticklables,
            "types": types, "type_colors": cols}


if __name__ == "__main__":
    c = Circuit("/gpfs/bbp.cscs.ch/project/proj96/scratch/home/ecker/plastyfire/BlueConfig")
    depths = pd.read_csv("/gpfs/bbp.cscs.ch/project/proj96/circuits/plastic_v1/hex_O1_depths.csv", index_col=0)
    raster_asth = get_raster_asthetics(c, depths)
    pklf_name = "raster_asth.pkl"
    with open(pklf_name, "wb") as f:
        pickle.dump(raster_asth, f, protocol=pickle.HIGHEST_PROTOCOL)


