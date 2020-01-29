import networkx as nx
import numpy as np
from typing import Dict, List, Set, Iterable, Tuple


class AugmentedGraph:
    """
    An nx.Graph with some extra convenience methods and attributes that makes it easier to work with flows along the
    edges. For example:
    1. It stores an arbitrarily assigned *order* of nodes and edges, accessible via self.nodes_arr amd self.edges_arr.
    2. Mappings between a node (or edge) and its order is accesible via the dictionaries
      - self.nodes2idx
      - self.edges2idx
      - self.idx2nodes
      - self.idx2edges
    3. It assigns an arbitrarily chosen *orientation* of each edge. Accessible via self.edge_orientation.
    4. It stores the oriented node-edge incidence matrix (self.M) where
      - The rows and columns are ordered identically as the order of nodes and edges (see 1).
      - For each edge (u,v), u is the head (positive in the incidence matrix) and v is the tail (positive in the
        incidence matrix).
    5. Returns the Laplacian matrix and its Moore-Penrose pseudoinverse, again in the same consistent node order.
    """
    def __init__(self, graph, weight_attr=None):
        self.G = graph
        self.weight_attr = weight_attr
        self.edges_arr = [e for e in self.G.edges()]
        self.nodes_arr = [e for e in self.G.nodes()]
        if weight_attr is None:
            self.weight_arr = np.array([1 for i in self.edges_arr])
        else:
            self.weight_arr = np.array([self.G[u][v][self.weight_attr] for u,v in self.edges_arr])

        self.nodes2idx = {n:idx for idx,n in enumerate(self.nodes_arr)}
        self.edges2idx = {e:idx for idx,e in enumerate(self.edges_arr)}
        self.idx2edges = {idx:e for idx,e in enumerate(self.edges_arr)}
        self.idx2nodes = {idx:n for idx,n in enumerate(self.nodes_arr)}

        # To understand why this *-1 is necessary, read the docs of nx.incidence_matrix
        self.M = nx.incidence_matrix(self.G, nodelist=self.nodes_arr, edgelist=self.edges_arr, oriented=True).toarray()*-1
        self.L = nx.laplacian_matrix(self.G, nodelist=self.nodes_arr, weight=self.weight_attr).toarray()
        self.Ld = np.linalg.pinv(self.L)
        self.MtLd = np.dot(self.M.T, self.Ld)

    def edge_weight(self, u,v):
        if self.weight_attr is None:
            return 1
        else:
            return self.G[u][v][self.weight_attr]

    def edge_orientation(self, u,v):
        """
        :param u: A node of self
        :param v: A node of self
        :return: (u,v) if (u,v) is in self.edges_arr, (v,u) is (v,u) is in the edges_arr. Raises ValueError
            if u and v are not neighbours.
        """
        if not self.G.has_edge(u, v):
            raise ValueError(f"({u},{v}) is not an edge.")
        if (u,v) in self.edges_arr:
            return u,v
        else:
            assert (v,u) in self.edges_arr
            return v,u

    def edges_ordered(self):
        return [self.idx2edges[idx] for idx in range(self.G.number_of_edges())]

    def nodes_ordered(self):
        return [self.idx2nodes[idx] for idx in range(self.G.number_of_nodes())]


def steady_flows_vector(Gr:AugmentedGraph, inputs:Dict):
    """
    :param Gr: A Graph
    :param inputs: Dict. Inputs at each node.
    :return: A vector specifying the flows across each edge.

    Note:
    The returned vector is to be interpreted as follows. If the nth edge in Gr.edges_arr is (a,b), then the nth element
    of the flow vector is the flow from a to b.
    """
    I = convert_node_keyed_dict_to_arr(Gr, inputs)
    return Gr.weight_arr*np.dot(Gr.MtLd, I)


def steady_flows_dict(Gr:AugmentedGraph, inputs:Dict):
    """
    Just the same as steady_flows_vector, just the return value is a dict instead of a vector.
    The dict keys are the elements of Gr.edges_arr.

    Note:
    The returned dict is to be interpreted as follows. retval[(a,b)] is the flow from a to b.
    """
    return convert_list_to_edge_keyed_dict(Gr, steady_flows_vector(Gr, inputs))


def current_dueto_dipole_at_edge_vector(Gr:AugmentedGraph, src_node, sink_node, current_magnitude = 1):
    I = {node:0 for node in Gr.nodes_arr}
    I[src_node] = current_magnitude
    I[sink_node] = -current_magnitude

    return steady_flows_vector(Gr, I)


def current_dueto_dipole_at_edge_dict(Gr:AugmentedGraph, src_node, sink_node, current_magnitude = 1):
    curr_vec = current_dueto_dipole_at_edge_vector(Gr, src_node, sink_node, current_magnitude)
    return convert_list_to_edge_keyed_dict(Gr, curr_vec)


def maxflow_change_on_each_edge_strengthening(Gr: AugmentedGraph, steady_flows_vec: List[float],
                                              maxflow_src, maxflow_sink):
    """
    Let the maxflow be across edge (s,t) from s to t. We will compute
    dF_{st}/dK_{ab}, for all edges (a,b).

    We utilize the property:
    dF_{st}/dK_{a,b} is equal to the current at edge (a,b) due to a dipole current source with +I at node t amd -I at
    node s, I = F_{ab}K_{st}/K^2_{ab}

    :param Gr: A graph.
    :param steady_flows_vec: A vector specifying the flows across each edge of Gr.
    :param maxflow_src/maxflow_sink: The edge with the maximum flow. The maximum flow is directed from src to sink.
    :return: A dict keyed by edges.
    """
    current_dueto_dipole_at_maxflow = current_dueto_dipole_at_edge_vector(Gr, src_node=maxflow_sink, sink_node=maxflow_src,
                                                                   current_magnitude=1)
    conversion_factor = steady_flows_vec * Gr.edge_weight(maxflow_src, maxflow_sink) / Gr.weight_arr ** 2
    return convert_list_to_edge_keyed_dict(Gr, current_dueto_dipole_at_maxflow*conversion_factor)


def convert_list_to_edge_keyed_dict(Gr: AugmentedGraph, vec: Iterable):
    """
    :param Gr: A graph
    :param vec: An iterable with length G.number_of_edges(). Encodes a property for each edge. The order is
        assumed to be Gr.
    :return: A dict keysed by edges, the values being the elments of vec.
    """
    return {e:vec[Gr.edges2idx[e]] for e in Gr.edges_ordered()}

def convert_node_keyed_dict_to_arr(Gr:AugmentedGraph, d: Dict):
    return np.array([d[node] for node in Gr.nodes_ordered()])


def evaluate_rerouting_heuristic_classifier(G:nx.Graph, Gr:AugmentedGraph, I:Dict, dists: Dict=None, thres=0):
    """
    Given a linear flow network, specified in terms of a graph G and an input vector I, computes how well the rerouting
    classifier works for predicting Braessian edges.

    :param G: A graph
    :param Gr: The same graph, in AugmentedGraph form.
    :param I: Nodal inputs, specified as a dictionary.
    :param dists: A dict specifying the distance between each node pairs in G.
    :return: A dict of dict of the form: {(1,2): {'braessian': True, 'aligned': False, 'susc': 0.123, 'dist': 1},
                                          (5,2): {'braessian': False, 'aligned': True, 'susc': -0.523, 'dist': 2},
                                         }
    """
    if dists is None:
        dists = dict(nx.all_pairs_shortest_path_length(G))

    # compute the steady flows
    stflows_vec = steady_flows_vector(Gr, I)
    stflows_dict = convert_list_to_edge_keyed_dict(Gr, stflows_vec)
    # compute the maximum flow
    maxflow_idx = np.argmax(np.abs(stflows_vec))
    maxflow = stflows_vec[maxflow_idx]
    maxflow_edge = Gr.edges_arr[maxflow_idx]
    if maxflow > 0:
        maxflow_src, maxflow_sink = maxflow_edge
    else:
        maxflow_sink, maxflow_src = maxflow_edge

    # now compute how the maxflow changes when each edge is infinitesimally strengthened
    susceptibilities_at_maxflow = maxflow_change_on_each_edge_strengthening(Gr, stflows_vec, maxflow_src,
                                                                            maxflow_sink)
    # Now compute flow alignments
    alignments_with_maxflow = flow_alignment_with_edge(G, Gr, maxflow_edge, stflows_dict)
    aligned, notaligned, cant_say = alignments_with_maxflow

    edge_dist = lambda e1,e2:min(dists[e1[0]][e2[0]], dists[e1[0]][e2[1]], dists[e1[1]][e2[0]], dists[e1[1]][e2[1]])
    return_dict = dict()
    for e in Gr.edges_arr:
        if e == maxflow_edge:
            continue
        inner_dict = dict()
        inner_dict['edge_idx'] = Gr.edges2idx[e]
        if e in aligned:
            inner_dict['aligned'] = True
        elif e in notaligned:
            inner_dict['aligned'] = False
        else:
            assert e in cant_say
            inner_dict['aligned'] = None
        inner_dict['susc'] = susceptibilities_at_maxflow[e]
        inner_dict['dist'] = edge_dist(e, maxflow_edge)
        susc = susceptibilities_at_maxflow[e]
        if abs(susc) < thres:
            inner_dict['braessian'] = None
        elif susc > 0:
            inner_dict['braessian'] = True
        else:
            inner_dict['braessian'] = False
        return_dict[e] = inner_dict
    return return_dict


def flow_alignment_with_edge(G:nx.Graph, Gr:AugmentedGraph, edge: Tuple[object, object], flows:Dict):
    """
    Given a graph, steady state flows, and a chosen edge; computes which edges are aligned by flow rerouting
    to the chosen edge.

    :param G: A graph
    :param Gr: AugmentedGraph of G
    :param edge: The chosen edge. Must be in G, and Gr.edges_arr.
    :param flows: A dict with keys of teh form (u,v) (must be in Gr.edges_arr) and values being floats. Specifies the
        flow from u to v.
    :return:
        aligned, notaligned, cant_say: Three mutually disjoint subsets of Gr.edges_arr.
    """
    if is_bridge(G, edge):
        # don't bother with any calculation.
        aligned_edges = set()
        notaligned_edges = set()
        cant_say = {e for e in G.edges() if e != edge}
        return aligned_edges, notaligned_edges, cant_say

    aligned_edges = set()
    nonaligned_edges = set()
    cant_say = set()

    for e in Gr.edges_arr:
        if set(e) == set(edge):
            continue

        alignment = is_edge_aligned_rerouting_heuristic(G, edge, e, flows)
        if alignment == 'aligned':
            aligned_edges.add(e)
        elif alignment == 'not aligned':
            nonaligned_edges.add(e)
        else:
            assert alignment == 'cant_say'
            cant_say.add(e)
    return aligned_edges, nonaligned_edges, cant_say


def is_edge_aligned_rerouting_heuristic(G, edge, e, flows):
    """
    Computes if e is aligned by flow rerouting to edge.

    For description of the arguments, see the docstring of flow_alignment_with_edge.:wa

    """
    if flows[edge] > 0:
        h_target,t_target = edge
    else:
        t_target,h_target = edge

    if is_bridge(G, e):
        return 'cant_say'

    h, t = e
    # first, determine head and tail
    if flows[(h,t)] < 0:
        t, h = h, t
    # now, compute if aligned
    try:
        shortest_h_target_h_t_t_target_path = shortest_rerouting_path(G, h_target, h, t, t_target)
        nal_path_length = len(shortest_h_target_h_t_t_target_path)
    except nx.NetworkXNoPath:
        nal_path_length = np.inf
    try:
        shortest_h_target_t_h_t_target_path = shortest_rerouting_path(G, h_target, t, h, t_target)
        al_path_length = len(shortest_h_target_t_h_t_target_path )
    except nx.NetworkXNoPath:
        al_path_length = np.inf

    if nal_path_length < al_path_length:
        return 'not aligned'
    elif al_path_length < nal_path_length:
        return 'aligned'
    else:
        return 'cant_say'


def shortest_rerouting_path(G, a, b, c, d):
    """
    Given a graph G containing edges (a,d) and (b,c)
    returns the shortest simple path that traverses a,...,b,c,..., d

    If no such path can be found, raises nx.NetworkXNoPath
    """
    H = G.copy()
    H.remove_edge(a,d)
    H.remove_edge(b,c)
    dist_c_d = nx.shortest_path_length(H,c,d)

    shortest_path_length = np.inf

    for path in nx.all_shortest_paths(H, a, b):
        if c in path or d in path:
            continue
        # so a (a,b,c,d) path may exist. check:
        K = H.copy()
        K.remove_edges_from(zip(path[:-1], path[1:]))
        K.remove_nodes_from(path[1:-1])

        try:
            otherpath = nx.shortest_path(K, c, d)
            new_shortest_path_length = len(path) + len(otherpath) + 1
            if new_shortest_path_length < shortest_path_length:
                shortest_path_length = new_shortest_path_length
                shortest_path = path + otherpath
            if len(otherpath) == dist_c_d:
                break
        except nx.NetworkXNoPath:
            pass
    if shortest_path_length < np.inf:
        return shortest_path
    else:
        raise nx.NetworkXNoPath

def shortest_rerouting_path_old(G, a, b, c, d):
    """
    Given a graph G containing edges (a,d) and (b,c)
    returns the shortest simple path that traverses a,...,b,c,..., d

    If no such path can be found, raises nx.NetworkXNoPath
    """
    H = G.copy()
    H.remove_edge(a,d)

    shortest_path_length = np.inf

    for path in nx.all_shortest_paths(H, a, b):
        if c not in path:
            # so a (a, b, c, d) path may exist. check:
            K = H.copy()
            K.remove_edges_from(zip(path[:-1], path[1:]))
            K.remove_nodes_from(path[1:-1])

            try:
                for otherpath in nx.all_shortest_paths(K, c, d):
                    if b not in otherpath:
                        is_nal = True
                        new_shortest_path_length = len(path) + len(otherpath) + 1
                        if new_shortest_path_length < shortest_path_length:
                            shortest_path_length = new_shortest_path_length
                            shortest_path = path+otherpath

            except nx.NetworkXNoPath:
                pass

    if shortest_path_length < np.inf:
        return shortest_path
    else:
        raise nx.NetworkXNoPath


def is_bridge(G, e):
    """
    Returns if e is a bridge. Assumes G is connected.
    """
    assert nx.is_connected(G)
    H = G.copy()
    H.remove_edge(*e)

    return not nx.is_connected(H)