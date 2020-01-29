import pandas as pd

from braess_detection.network_generation import generate_random_voronoi_graph_with_random_inputs,\
    generate_square_grid_with_random_inputs, generate_ieee300_network_with_random_inputs
from braess_detection.braess_tools import AugmentedGraph, evaluate_rerouting_heuristic_classifier

def run_n_iterations_for_network(network_type, out_file, num_repeat=10):
    print(f"Doing {network_type}")
    all_dfs = []
    for rep_idx in range(num_repeat):
        print("doing iteration {}".format(rep_idx))
        if network_type == 'voronoi':
            G, Gr, I_dict = generate_random_voronoi_graph_with_random_inputs(num_points=20)
        elif network_type == 'lattice':
            G, Gr, I_dict = generate_square_grid_with_random_inputs(grid_size=10)
        elif network_type == 'ieee300':
            G, Gr, I_dict = generate_ieee300_network_with_random_inputs()
        else:
            raise ValueError(f"network_type={network_type} not understood")
        res = evaluate_rerouting_heuristic_classifier(G, Gr, I_dict, thres=10e-5)
        # res is a dict of dicts, keys keing node labels. We will store it as a csv
        # we will also lose information of the node labels.
        df = pd.DataFrame([data for node, data in res.items()], index=None)
        df.loc[:, 'rep_idx'] = rep_idx
        all_dfs.append(df)

    total_df = pd.concat(all_dfs, ignore_index=True)
    print(f"writing to {out_file}")
    total_df.to_csv(out_file)


if __name__=='__main__':
    num_repeat = 5
    print("Doing voronoi")
    run_n_iterations_for_network(network_type='voronoi', out_file='../data/voronoi.csv', num_repeat=num_repeat)
    print("Doing lattice")
    run_n_iterations_for_network(network_type='lattice', out_file='../data/lattice.csv', num_repeat=num_repeat)
    print("Doing ieee300")
    run_n_iterations_for_network(network_type='ieee300', out_file='../data/ieee300.csv', num_repeat=num_repeat)


