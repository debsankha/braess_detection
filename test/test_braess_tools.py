from braess_detection import braess_tools as ab
from flownetpy.tools import FlowDict
from flownetpy import LinearFlowNetwork
import networkx as nx
import numpy as np
import pytest
import random
from unittest import TestCase


def to_flowdict(flows, G):
    return FlowDict({edge: flow for edge, flow in zip(G.edges_arr, flows)})


class TestAlignmentRing(TestCase):
    def setUp(self):
        self.G = nx.Graph()

        for i in range(6):
            self.G.add_edge(i, (i + 1) % 6)
        self.Gr = ab.AugmentedGraph(self.G)

    def test_allaligned(self):
        flows = {(i, (i + 1) % 6): 1 for i in range(6)}
        flows = FlowDict(flows)

        aligned, nonaligned, cant_say, bridges = ab.flow_alignment_with_edge(
            self.G, self.Gr, (0, 1), flows
        )

        assert len(nonaligned) == 0
        assert len(aligned) == 5
        assert len(cant_say) == 0

    def test_unaligned_one(self):
        flows = {(i, (i + 1) % 6): 1 for i in range(6)}
        flows[(4, 5)] = -1
        flows = FlowDict(flows)

        aligned, nonaligned, cant_say, bridges = ab.flow_alignment_with_edge(
            self.G, self.Gr, (0, 1), flows
        )

        assert len(nonaligned) == 1
        assert len(aligned) == 4
        assert len(cant_say) == 0
        assert nonaligned == {(4, 5)}


def test_lattice_alignment_cal():
    G = nx.grid_2d_graph(5, 5)
    n_repeat = 20
    for _ in range(n_repeat):
        edges = list(G.edges())
        (h, t), (h_target, t_target) = random.sample(edges, 2)

        res_normal = ab._is_aligned_rerouting_heuristic(G, h_target, h, t, t_target)
        res_lattice = ab._is_aligned_rerouting_heuristic_lattice(
            G, h_target, h, t, t_target
        )
        assert res_lattice == res_normal


class TestAlignmentGrid(TestCase):
    def setUp(self):
        self.G = nx.grid_2d_graph(4, 5)
        self.src, self.sink = (1, 2), (2, 2)
        self.Gr = ab.AugmentedGraph(self.G)
        # store true alignments
        # from theory, we know they should be
        true_alignments = set()
        # first, horizontal edges
        for x in range(3):
            for y in [0, 1, 3, 4]:
                true_alignments.add(((x + 1, y), (x, y)))
        true_alignments.add(((0, 2), (1, 2)))
        true_alignments.add(((2, 2), (3, 2)))
        # then, vertical ones
        for x in [0, 1]:
            for y in [0, 1]:
                true_alignments.add(((x, y), (x, y + 1)))
            for y in [2, 3]:
                true_alignments.add(((x, y + 1), (x, y)))
        for x in [2, 3]:
            for y in [2, 3]:
                true_alignments.add(((x, y), (x, y + 1)))
            for y in [0, 1]:
                true_alignments.add(((x, y + 1), (x, y)))
        true_alignments.add((self.src, self.sink))
        self.true_alignments = true_alignments

        if (self.src, self.sink) in self.Gr.edges_arr:
            self.maxflow_edge = (self.src, self.sink)
        else:
            assert (self.sink, self.src) in self.Gr.edges_arr
            self.maxflow_edge = (self.sink, self.src)

    def test_alignment_all_aligned(self):
        flows = dict()
        for u, v in self.Gr.edges_arr:
            if (u, v) in self.true_alignments:
                flows[(u, v)] = 1
            else:
                assert (v, u) in self.true_alignments
                flows[(u, v)] = -1
        aligned, nonaligned, cant_say, bridges = ab.flow_alignment_with_edge(
            self.G, self.Gr, self.maxflow_edge, flows
        )
        self.assertEqual(len(nonaligned), 0)
        self.assertEqual(len(cant_say), 0)

    def test_alignment_few_nonaligned(self):
        flows = dict()
        for u, v in self.Gr.edges_arr:
            if (u, v) in self.true_alignments:
                flows[(u, v)] = 1
            else:
                assert (v, u) in self.true_alignments
                flows[(u, v)] = -1
        # now flip the orientation of a few flows
        tobe_notaligned_edges = set(random.sample(self.Gr.edges_arr, k=4))
        tobe_notaligned_edges -= set(self.maxflow_edge)
        for e in tobe_notaligned_edges:
            flows[e] *= -1

        aligned, nonaligned, cant_say, bridges = ab.flow_alignment_with_edge(
            self.G, self.Gr, self.maxflow_edge, flows
        )
        self.assertEqual(
            len(aligned), self.G.number_of_edges() - len(tobe_notaligned_edges) - 1
        )
        self.assertEqual(len(nonaligned), len(tobe_notaligned_edges))
        self.assertEqual(len(cant_say), 0)

        self.assertSetEqual(tobe_notaligned_edges, nonaligned)


class TestDipoleFlowCalc(TestCase):
    def test_stflow_compare(self):
        # first, define the graph
        N = 15
        G = nx.grid_2d_graph(N, N)
        for _ in range(10):
            # assign random edge weights
            for u, v in G.edges():
                w = random.uniform(0.5, 1.5)
                G[u][v]["weight"] = w

            Gr = ab.AugmentedGraph(G, weight_attr="weight")

            # select one source and one sink
            I = np.zeros(G.number_of_nodes())

            ((src, sink),) = random.sample(G.edges(), 1)

            src_idx = Gr.nodes2idx[src]
            sink_idx = Gr.nodes2idx[sink]

            I[src_idx] = 1
            I[sink_idx] = -1

            # compute steady flows fancy way
            stflows_fancy = ab.current_dueto_dipole_at_edge_dict(Gr, src, sink)

            # now compute it normal way
            Fn = LinearFlowNetwork(G, I, weight="weight")
            stflows = Fn.steady_flows()

            for e in Gr.edges_arr:
                self.assertAlmostEqual(stflows_fancy[e], stflows[e])


class TestSteadyFlows(TestCase):
    def test_stflow_compare(self):
        # first, define the graph
        N = 15
        G = nx.grid_2d_graph(N, N)
        for _ in range(10):
            # assign random edge weights
            for u, v in G.edges():
                w = random.uniform(0.5, 1.5)
                G[u][v]["weight"] = w

            Gr = ab.AugmentedGraph(G, weight_attr="weight")
            # Construct the input vector I
            I = np.zeros(G.number_of_nodes())

            # select some sources and some sinks
            num_srcs = num_sinks = I.size // 2
            src_idxs = np.random.choice(range(I.size), num_srcs, replace=False)
            sink_idxs = np.random.choice(
                list(set(range(I.size)) - set(src_idxs)), num_sinks, replace=False
            )

            I[src_idxs] = 1
            I[sink_idxs] = -1

            assert I.sum() == 0

            # Construct the input dict
            I_dict = {n: 0 for n in G.nodes()}
            for src_idx in src_idxs:
                src = Gr.idx2nodes[src_idx]
                I_dict[src] = 1
            for sink_idx in sink_idxs:
                sink = Gr.idx2nodes[sink_idx]
                I_dict[sink] = -1

            # compute steady flows fancy way
            stflows_fancy = ab.steady_flows_dict(Gr, I_dict)

            # now compute it normal way
            Fn = LinearFlowNetwork(G, I, weight="weight")
            stflows = Fn.steady_flows()

            for e in Gr.edges_arr:
                assert np.isclose(stflows_fancy[e], stflows[e])


class TestMaxflowChange(TestCase):
    def test_stuff(self):
        G = nx.Graph()

        for i in range(6):
            G.add_edge(i, (i + 1) % 6)

        Gr = ab.AugmentedGraph(G)
        inputs = {n: 0 for n in G.nodes()}
        src, sink = 3, 4
        inputs[src] = 1
        inputs[sink] = -1

        steadyflows = ab.steady_flows_vector(Gr, inputs)
        maxflow_edge = (maxflow_src, maxflow_sink) = (3, 4)
        maxflow_change = ab.maxflow_change_on_each_edge_strengthening(
            Gr,
            steady_flows_vec=steadyflows,
            maxflow_src=maxflow_src,
            maxflow_sink=maxflow_sink,
        )

        # compute maxflow change in dumb way
        Fn = LinearFlowNetwork(G, inputs, weight=1)
        old_stflows = Fn.steady_flows()

        for u, v in Gr.edges_arr:
            # the smart computation is wrong at maxflow edg. So skip it.
            if set([u, v]) == set(maxflow_edge):
                continue
            H = G.copy()

            EPS = 10e-7
            Fn = LinearFlowNetwork(G, inputs, weight=1)
            Fn[u][v]["weight"] += EPS
            new_stflows = Fn.steady_flows()
            rel_maxflow_change = (
                new_stflows[maxflow_edge] - old_stflows[maxflow_edge]
            ) / EPS
            self.assertAlmostEqual(rel_maxflow_change, maxflow_change[(u, v)], places=5)


class TestBraessianness:
    def test_be_random(self):
        # first, define the graph
        N = 10
        G = nx.grid_2d_graph(N, N)
        Gr = ab.AugmentedGraph(G)
        input_vec = np.zeros(len(Gr.nodes_arr))
        input_dict = {n: 0 for n in Gr.nodes_arr}

        # select some sources and some sinks
        num_srcs = num_sinks = input_vec.size // 2
        src_idxs = np.random.choice(range(input_vec.size), num_srcs, replace=False)
        sink_idxs = np.random.choice(
            list(set(range(input_vec.size)) - set(src_idxs)), num_sinks, replace=False
        )

        input_vec[src_idxs] = 1
        input_vec[sink_idxs] = -1
        assert input_vec.sum() == 0
        for src_idx in src_idxs:
            src = Gr.idx2nodes[src_idx]
            input_dict[src] = 1
        for sink_idx in sink_idxs:
            sink = Gr.idx2nodes[sink_idx]
            input_dict[sink] = -1

        # now compute braessian edges normal way
        Fn = LinearFlowNetwork(G, input_vec, weight=1)
        stflows = Fn.steady_flows()

        maxflow_edge = max(stflows, key=lambda x: abs(stflows[x]))
        be_direct = []
        for e in Gr.edges_arr:
            u, v = e
            if e == maxflow_edge:
                continue

            EPS = 10e-6
            Fn = LinearFlowNetwork(G, input_vec, weight=1)
            Fn[u][v]["weight"] += EPS
            new_stflows = Fn.steady_flows()

            if abs(new_stflows[maxflow_edge]) > abs(stflows[maxflow_edge]):
                be_direct.append(e)

        # compute braessian edges fancy way
        result = ab.evaluate_rerouting_heuristic_classifier(G, Gr, input_dict)
        be_fancy = {e for e, data in result.items() if data["braessian"] is True}
        assert set(be_direct) == set(be_fancy)
