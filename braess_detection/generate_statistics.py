import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import itertools as it

from braess_detection.network_generation import generate_random_voronoi_graph_with_random_inputs,\
    generate_square_grid_with_random_inputs, generate_ieee300_network_with_random_inputs
from braess_detection.braess_tools import evaluate_rerouting_heuristic_classifier

def run_n_iterations_for_network(network_type, out_file, num_repeat=10):
    all_dfs = []

    #with ProcessPoolExecutor() as e:
    all_dfs = list(map(_single_iteration, it.repeat(network_type), range(num_repeat)))

    total_df = pd.concat(all_dfs, ignore_index=True)
    print(f"writing to {out_file}")
    total_df.to_csv(out_file)

def _single_iteration(network_type, seed):
    np.random.seed(seed)
    is_lattice = False
    if network_type == 'voronoi':
        G, Gr, I_dict = generate_random_voronoi_graph_with_random_inputs(num_points=20)
    elif network_type == 'lattice':
        G, Gr, I_dict = generate_square_grid_with_random_inputs(grid_size=30)
        is_lattice = True
    elif network_type == 'ieee300':
        G, Gr, I_dict = generate_ieee300_network_with_random_inputs()
    else:
        raise ValueError(f"network_type={network_type} not understood")
    res = evaluate_rerouting_heuristic_classifier(G, Gr, I_dict, thres=0, is_lattice=is_lattice)
    # res is a dict of dicts, keys keing node labels. We will store it as a csv
    # we will also lose information of the node labels.
    df = pd.DataFrame([data for node, data in res.items()], index=None)
    df.loc[:, 'seed'] = seed
    return df


if __name__=='__main__':
    num_repeat = 200
    print("Doing voronoi")
    run_n_iterations_for_network(network_type='voronoi', out_file='../../data/voronoi.csv', num_repeat=num_repeat)
    print("Doing ieee300")
    run_n_iterations_for_network(network_type='ieee300', out_file='../../data/ieee300.csv', num_repeat=num_repeat)
    print("Doing lattice")
    run_n_iterations_for_network(network_type='lattice', out_file='../../data/lattice.csv', num_repeat=num_repeat)

