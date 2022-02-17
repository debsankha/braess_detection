import networkx as nx
import numpy as np
from . import voronoigraph as vg
from . import braess_tools as bt

import sys
sys.path.insert(0, "../../../random-powergrid")
import rpgm_algo


def assign_random_sources_and_sinks(Gr:bt.AugmentedGraph, frac:float=0.25):
    num_nodes = len(Gr.nodes_arr)
    num_srcs = num_sinks = num_nodes // 4

    while True:
        src_idxs = np.random.choice(range(num_nodes), num_srcs, replace=False)
        sink_idxs = np.random.choice(list(set(range(num_nodes)) - set(src_idxs)), num_sinks, replace=False)

        # Construct the input dict
        I_dict = {n: 0 for n in Gr.nodes_arr}
        for src_idx in src_idxs:
            src = Gr.idx2nodes[src_idx]
            I_dict[src] = 1
        for sink_idx in sink_idxs:
            sink = Gr.idx2nodes[sink_idx]
            I_dict[sink] = -1

        # exclude the cases where maxflow edge is a bridge
        F = bt.steady_flows_vector(Gr, I_dict)
        maxflow_edge = Gr.idx2edges[np.argmax(np.abs(F))]
        if not bt.is_bridge(Gr.G, maxflow_edge):
            return I_dict

def generate_square_grid_with_random_inputs(grid_size=10, src_frac=0.25):
    G = nx.grid_2d_graph(grid_size, grid_size)
    Gr = bt.AugmentedGraph(G)

    # randomly choose sources and sinks
    I_dict = assign_random_sources_and_sinks(Gr, frac=src_frac)
    return G, Gr, I_dict

def generate_random_voronoi_graph_with_random_inputs(num_points=20, src_frac=0.25):
    while True:
        VG = vg.VoronoiPlanarGraph(num_points)
        G = VG.graph
        H = VG.dual_graph
        if nx.is_connected(G) and nx.is_connected(H):
            break
    Gr = bt.AugmentedGraph(G)
    # randomly choose sources and sinks
    I_dict = assign_random_sources_and_sinks(Gr, frac=src_frac)
    return G, Gr, I_dict

def generate_ieee300_network_with_random_inputs(src_frac=0.25):
    G = nx.read_gpickle('../../data/ieee300/ieee300_unweighted_graph.gpkl')
    Gr = bt.AugmentedGraph(G)
    # randomly choose sources and sinks
    I_dict = assign_random_sources_and_sinks(Gr, frac=src_frac)
    return G, Gr, I_dict

def generate_random_powergrid_network_with_random_inputs(num_nodes, src_frac=0.25):
    G = rpgm_algo.main(n=num_nodes).to_networkx()
    Gr = bt.AugmentedGraph(G)
    # randomly choose sources and sinks
    I_dict = assign_random_sources_and_sinks(Gr, frac=src_frac)
    return G, Gr, I_dict
