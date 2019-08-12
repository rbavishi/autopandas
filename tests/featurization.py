import unittest

import pandas as pd
import numpy as np
from autopandas_v2.ml.featurization.featurizer import RelationGraph
from autopandas_v2.ml.featurization.graph import GraphEdge, GraphEdgeType, GraphNodeType, GraphNode
from autopandas_v2.ml.featurization.options import GraphOptions

get_node_type = GraphNodeType.get_node_type


class TestRelationGraphFeaturizer(unittest.TestCase):
    def test_basic_max(self):
        input_df = pd.DataFrame([[1, 2], [2, 3], [2, 0]])
        input_00 = GraphNode("I0", '[0,0]', get_node_type(input_df.iat[0, 0]))
        input_01 = GraphNode("I0", '[0,1]', get_node_type(input_df.iat[0, 1]))
        input_10 = GraphNode("I0", '[1,0]', get_node_type(input_df.iat[1, 0]))
        input_11 = GraphNode("I0", '[1,1]', get_node_type(input_df.iat[1, 1]))
        input_20 = GraphNode("I0", '[2,0]', get_node_type(input_df.iat[2, 0]))
        input_21 = GraphNode("I0", '[2,1]', get_node_type(input_df.iat[2, 1]))
        output_df = pd.DataFrame([[2, 3]])
        output_00 = GraphNode("O0", '[0,0]', get_node_type(output_df.iat[0, 0]))
        output_01 = GraphNode("O0", '[0,1]', get_node_type(output_df.iat[0, 1]))
        options = GraphOptions()
        options.NODE_TYPES = True
        rel_graph: RelationGraph = RelationGraph(options)
        rel_graph.from_input_output([input_df], output_df)
        rel_graph_edges = rel_graph.edges
        #  positional edges
        positional_edges = [
            GraphEdge(input_00, input_01, GraphEdgeType.ADJACENCY),
            GraphEdge(input_00, input_10, GraphEdgeType.ADJACENCY),
            GraphEdge(input_10, input_11, GraphEdgeType.ADJACENCY),
            GraphEdge(input_10, input_20, GraphEdgeType.ADJACENCY),
            GraphEdge(input_20, input_21, GraphEdgeType.ADJACENCY),
            GraphEdge(input_01, input_11, GraphEdgeType.ADJACENCY),
            GraphEdge(input_11, input_21, GraphEdgeType.ADJACENCY),
            GraphEdge(output_00, output_01, GraphEdgeType.ADJACENCY)
        ]
        for edge in positional_edges:
            self.assertTrue(edge in rel_graph_edges,
                            "Could not find edge %s in set of edges:\n%s" % (edge, rel_graph_edges))
        #  equality edges
        equality_edges = [
            GraphEdge(input_10, output_00, GraphEdgeType.EQUALITY),
            GraphEdge(input_20, output_00, GraphEdgeType.EQUALITY),
            GraphEdge(input_01, output_00, GraphEdgeType.EQUALITY),  # redundant
            GraphEdge(input_11, output_01, GraphEdgeType.EQUALITY)
        ]
        for edge in equality_edges:
            self.assertTrue(edge in rel_graph_edges,
                            "Could not find edge %s in set of edges:\n%s" % (edge, rel_graph_edges))

    def test_max_series(self):
        input_df = pd.DataFrame([[1, 2], [2, 3], [2, 0]])
        input_00 = GraphNode("I0", '[0,0]', get_node_type(input_df.iat[0, 0]))
        input_01 = GraphNode("I0", '[0,1]', get_node_type(input_df.iat[0, 1]))
        input_10 = GraphNode("I0", '[1,0]', get_node_type(input_df.iat[1, 0]))
        input_11 = GraphNode("I0", '[1,1]', get_node_type(input_df.iat[1, 1]))
        input_20 = GraphNode("I0", '[2,0]', get_node_type(input_df.iat[2, 0]))
        input_21 = GraphNode("I0", '[2,1]', get_node_type(input_df.iat[2, 1]))
        output = pd.DataFrame.max(input_df)
        output_00 = GraphNode("O0", '[0,0]', get_node_type(output.iat[0]))
        output_10 = GraphNode("O0", '[1,0]', get_node_type(output.iat[1]))
        options = GraphOptions()
        options.NODE_TYPES = True
        rel_graph: RelationGraph = RelationGraph(options)
        rel_graph.from_input_output([input_df], output)
        rel_graph_edges = rel_graph.edges
        #  positional edges
        positional_edges = [
            GraphEdge(input_00, input_01, GraphEdgeType.ADJACENCY),
            GraphEdge(input_00, input_10, GraphEdgeType.ADJACENCY),
            GraphEdge(input_10, input_11, GraphEdgeType.ADJACENCY),
            GraphEdge(input_10, input_20, GraphEdgeType.ADJACENCY),
            GraphEdge(input_20, input_21, GraphEdgeType.ADJACENCY),
            GraphEdge(input_01, input_11, GraphEdgeType.ADJACENCY),
            GraphEdge(input_11, input_21, GraphEdgeType.ADJACENCY),
            GraphEdge(output_00, output_10, GraphEdgeType.ADJACENCY)
        ]
        for edge in positional_edges:
            self.assertTrue(edge in rel_graph_edges,
                            "Could not find edge %s in set of edges:\n%s" % (edge, rel_graph_edges))
        #  equality edges
        equality_edges = [
            GraphEdge(input_10, output_00, GraphEdgeType.EQUALITY),
            GraphEdge(input_20, output_00, GraphEdgeType.EQUALITY),
            GraphEdge(input_01, output_00, GraphEdgeType.EQUALITY),  # redundant
            GraphEdge(input_11, output_10, GraphEdgeType.EQUALITY)
        ]
        for edge in equality_edges:
            self.assertTrue(edge in rel_graph_edges,
                            "Could not find edge %s in set of edges:\n%s" % (edge, rel_graph_edges))

    def test_values(self):
        input_df = pd.DataFrame([[1, 2], [3, 4]])
        input_00 = GraphNode("I0", '[0,0]', get_node_type(input_df.iat[0, 0]))
        input_01 = GraphNode("I0", '[0,1]', get_node_type(input_df.iat[0, 1]))
        input_10 = GraphNode("I0", '[1,0]', get_node_type(input_df.iat[1, 0]))
        input_11 = GraphNode("I0", '[1,1]', get_node_type(input_df.iat[1, 1]))
        output = input_df.values
        output_00 = GraphNode("O0", '[0,0]', get_node_type(output[0, 0]))
        output_01 = GraphNode("O0", '[0,1]', get_node_type(output[0, 1]))
        output_10 = GraphNode("O0", '[1,0]', get_node_type(output[1, 0]))
        output_11 = GraphNode("O0", '[1,1]', get_node_type(output[1, 1]))
        options = GraphOptions()
        options.NODE_TYPES = True
        rel_graph: RelationGraph = RelationGraph(options)
        rel_graph.from_input_output([input_df], output)
        rel_graph_edges = rel_graph.edges
        #  positional edges
        positional_edges = [
            GraphEdge(input_00, input_01, GraphEdgeType.ADJACENCY),
            GraphEdge(input_00, input_10, GraphEdgeType.ADJACENCY),
            GraphEdge(input_10, input_11, GraphEdgeType.ADJACENCY),
            GraphEdge(input_01, input_11, GraphEdgeType.ADJACENCY),
            GraphEdge(output_00, output_01, GraphEdgeType.ADJACENCY),
            GraphEdge(output_00, output_10, GraphEdgeType.ADJACENCY),
            GraphEdge(output_10, output_11, GraphEdgeType.ADJACENCY),
            GraphEdge(output_01, output_11, GraphEdgeType.ADJACENCY)
        ]
        for edge in positional_edges:
            self.assertTrue(edge in rel_graph_edges,
                            "Could not find edge %s in set of edges:\n%s" % (edge, rel_graph_edges))
        equality_edges = [
            GraphEdge(input_00, output_00, GraphEdgeType.EQUALITY),
            GraphEdge(input_10, output_10, GraphEdgeType.EQUALITY),
            GraphEdge(input_01, output_01, GraphEdgeType.EQUALITY),
            GraphEdge(input_11, output_11, GraphEdgeType.EQUALITY)
        ]
        for edge in equality_edges:
            self.assertTrue(edge in rel_graph_edges,
                            "Could not find edge %s in set of edges:\n%s" % (edge, rel_graph_edges))

    def test_dict(self):
        input_df = pd.DataFrame([[1, 2], [3, 4]])
        input_00 = GraphNode("I0", '[0,0]', get_node_type(input_df.iat[0, 0]))
        input_01 = GraphNode("I0", '[0,1]', get_node_type(input_df.iat[0, 1]))
        input_10 = GraphNode("I0", '[1,0]', get_node_type(input_df.iat[1, 0]))
        input_11 = GraphNode("I0", '[1,1]', get_node_type(input_df.iat[1, 1]))
        output = {"A": [1, 3], "B": [2, 4]}
        output_00 = GraphNode("O0", '[0,0]', get_node_type(output['A'][0]))
        output_01 = GraphNode("O0", '[0,1]', get_node_type(output['B'][0]))
        output_10 = GraphNode("O0", '[1,0]', get_node_type(output['A'][1]))
        output_11 = GraphNode("O0", '[1,1]', get_node_type(output['B'][1]))
        options = GraphOptions()
        options.NODE_TYPES = True
        rel_graph: RelationGraph = RelationGraph(options)
        rel_graph.from_input_output([input_df], output)
        rel_graph_edges = rel_graph.edges
        positional_edges = [
            GraphEdge(input_00, input_01, GraphEdgeType.ADJACENCY),
            GraphEdge(input_00, input_10, GraphEdgeType.ADJACENCY),
            GraphEdge(input_10, input_11, GraphEdgeType.ADJACENCY),
            GraphEdge(input_01, input_11, GraphEdgeType.ADJACENCY),
            GraphEdge(output_00, output_01, GraphEdgeType.ADJACENCY),
            GraphEdge(output_00, output_10, GraphEdgeType.ADJACENCY),
            GraphEdge(output_10, output_11, GraphEdgeType.ADJACENCY),
            GraphEdge(output_01, output_11, GraphEdgeType.ADJACENCY)
        ]
        for edge in positional_edges:
            self.assertTrue(edge in rel_graph_edges,
                            "Could not find edge %s in set of edges:\n%s" % (edge, rel_graph_edges))
        equality_edges = [
            GraphEdge(input_00, output_00, GraphEdgeType.EQUALITY),
            GraphEdge(input_10, output_10, GraphEdgeType.EQUALITY),
            GraphEdge(input_01, output_01, GraphEdgeType.EQUALITY),
            GraphEdge(input_11, output_11, GraphEdgeType.EQUALITY)
        ]
        for edge in equality_edges:
            self.assertTrue(edge in rel_graph_edges,
                            "Could not find edge %s in set of edges:\n%s" % (edge, rel_graph_edges))

    def test_groupby_output(self):
        input_df = pd.DataFrame({
            "Name": ["Alice", "Bob", "Mallory", "Mallory", "Bob", "Mallory"],
            "City": ["Seattle", "Seattle", "Portland", "Seattle", "Seattle", "Portland"]})
        output = input_df.groupby("Name")
        options = GraphOptions()
        options.NODE_TYPES = True
        options.ADJACENCY_EDGES = False
        rel_graph: RelationGraph = RelationGraph(options)
        rel_graph.from_input_output([input_df], output)
        rel_graph_edges = rel_graph.edges
        alice_nodes_in = [
            GraphNode("I0", '[0,0]', GraphNodeType.STR)
        ]
        alice_nodes_out = [
            GraphNode("O0_0", '[0,0]', GraphNodeType.STR)
        ]
        bob_nodes_in = [
            GraphNode("I0", '[1,0]', GraphNodeType.STR),
            GraphNode("I0", '[4,0]', GraphNodeType.STR)
        ]
        bob_nodes_out = [
            GraphNode("O0_1", '[0,0]', GraphNodeType.STR),
            GraphNode("O0_1", '[1,0]', GraphNodeType.STR)
        ]
        mallory_nodes_in = [
            GraphNode("I0", '[2,0]', GraphNodeType.STR),
            GraphNode("I0", '[3,0]', GraphNodeType.STR),
            GraphNode("I0", '[5,0]', GraphNodeType.STR)
        ]
        mallory_nodes_out = [
            GraphNode("O0_2", '[0,0]', GraphNodeType.STR),
            GraphNode("O0_2", '[1,0]', GraphNodeType.STR),
            GraphNode("O0_2", '[2,0]', GraphNodeType.STR)
        ]
        seattle_nodes_in = [
            GraphNode("I0", '[0,1]', GraphNodeType.STR),
            GraphNode("I0", '[1,1]', GraphNodeType.STR),
            GraphNode("I0", '[3,1]', GraphNodeType.STR),
            GraphNode("I0", '[4,1]', GraphNodeType.STR),
        ]
        seattle_nodes_out = [
            GraphNode("O0_0", '[0,1]', GraphNodeType.STR),
            GraphNode("O0_1", '[0,1]', GraphNodeType.STR),
            GraphNode("O0_2", '[1,1]', GraphNodeType.STR)
        ]
        portland_nodes_in = [
            GraphNode("I0", '[2,1]', GraphNodeType.STR),
            GraphNode("I0", '[5,1]', GraphNodeType.STR)
        ]
        portland_nodes_out = [
            GraphNode("O0_2", '[0,1]', GraphNodeType.STR),
            GraphNode("O0_2", '[2,1]', GraphNodeType.STR)
        ]

        def check_edges(in_nodes, out_nodes):
            for in_node in in_nodes:
                for out_node in out_nodes:
                    edge = GraphEdge(in_node, out_node, GraphEdgeType.EQUALITY)
                    self.assertTrue(edge in rel_graph_edges,
                                    "Could not find edge %s in set of edges:\n%s" % (edge, rel_graph_edges))

        check_edges(alice_nodes_in, alice_nodes_out)
        check_edges(bob_nodes_in, bob_nodes_out)
        check_edges(mallory_nodes_in, mallory_nodes_out)
        check_edges(portland_nodes_in, portland_nodes_out)
        check_edges(seattle_nodes_in, seattle_nodes_out)

    def test_groupby_input(self):
        df = pd.DataFrame({
            "Name": ["Alice", "Bob", "Mallory", "Mallory", "Bob", "Mallory"],
            "City": ["Seattle", "Seattle", "Portland", "Seattle", "Seattle", "Portland"]})
        input_ = df.groupby("Name")
        output = input_.count().reset_index()
        options = GraphOptions()
        options.NODE_TYPES = True
        options.ADJACENCY_EDGES = False
        rel_graph: RelationGraph = RelationGraph(options)
        rel_graph.from_input_output([input_], output)
        rel_graph_edges = rel_graph.edges
        alice_nodes_in = [
            GraphNode("I0_0", '[0,0]', GraphNodeType.STR)
        ]

        alice_nodes_out = [
            GraphNode("O0", '[0,0]', GraphNodeType.STR)
        ]

        bob_nodes_in = [
            GraphNode("I0_1", '[0,0]', GraphNodeType.STR),
            GraphNode("I0_1", '[1,0]', GraphNodeType.STR)
        ]
        bob_nodes_out = [
            GraphNode("O0", '[1,0]', GraphNodeType.STR)
        ]

        mallory_nodes_in = [
            GraphNode("I0_2", '[0,0]', GraphNodeType.STR),
            GraphNode("I0_2", '[1,0]', GraphNodeType.STR),
            GraphNode("I0_2", '[2,0]', GraphNodeType.STR)
        ]
        mallory_nodes_out = [
            GraphNode("O0", '[2,0]', GraphNodeType.STR)
        ]

        def check_edges(in_nodes, out_nodes):
            for in_node in in_nodes:
                for out_node in out_nodes:
                    edge = GraphEdge(in_node, out_node, GraphEdgeType.EQUALITY)
                    self.assertTrue(edge in rel_graph_edges,
                                    "Could not find edge %s in set of edges:\n%s" % (edge, rel_graph_edges))

        check_edges(alice_nodes_in, alice_nodes_out)
        check_edges(bob_nodes_in, bob_nodes_out)
        check_edges(mallory_nodes_in, mallory_nodes_out)

    def test_idx_multi(self):
        tuples = [("bar", "one"), ("bar", "two")]
        index = pd.MultiIndex.from_tuples(tuples)
        data = [[0], [1]]
        input_df = pd.DataFrame(data, index=index)
        #          0
        # bar one  0
        #     two  1
        output_df = input_df.unstack()
        #       0
        #     one two
        # bar   0   1
        options = GraphOptions()
        options.COLUMN_NODES = True
        options.INDEX_NODES = True
        options.ADJACENCY_EDGES = True
        options.EQUALITY_EDGES = True
        options.NODE_TYPES = True
        options.INDEX_EDGES = True
        rel_graph: RelationGraph = RelationGraph(options)
        rel_graph.from_input_output([input_df], output_df)
        rel_graph_edges = rel_graph.edges

        bar_in_0 = GraphNode("I0", '[0,-2]', GraphNodeType.INDEX)
        bar_in_1 = GraphNode("I0", '[1,-2]', GraphNodeType.INDEX)
        bar_out = GraphNode("O0", '[0,-1]', GraphNodeType.INDEX)

        one_in = GraphNode("I0", '[0,-1]', GraphNodeType.INDEX)
        two_in = GraphNode("I0", '[1,-1]', GraphNodeType.INDEX)

        one_out = GraphNode("O0", '[-1,0]', GraphNodeType.COLUMN)
        two_out = GraphNode("O0", '[-1,1]', GraphNodeType.COLUMN)

        in_0 = GraphNode("I0", '[0,0]', GraphNodeType.INT)
        in_1 = GraphNode("I0", '[1,0]', GraphNodeType.INT)

        out_0 = GraphNode("O0", '[0,0]', GraphNodeType.INT)
        out_1 = GraphNode("O0", '[0,1]', GraphNodeType.INT)

        adjacency_edges = [
            GraphEdge(bar_in_0, bar_in_1, GraphEdgeType.ADJACENCY),
            GraphEdge(bar_in_0, one_in, GraphEdgeType.ADJACENCY),
            GraphEdge(bar_in_1, two_in, GraphEdgeType.ADJACENCY),
            GraphEdge(one_in, two_in, GraphEdgeType.ADJACENCY)
        ]

        for edge in adjacency_edges:
            self.assertTrue(edge in rel_graph_edges,
                            "Could not find edge %s in set of edges:\n%s" % (edge, rel_graph_edges))
        indexing_edges = [
            GraphEdge(bar_in_0, in_0, GraphEdgeType.INDEX),
            GraphEdge(one_in, in_0, GraphEdgeType.INDEX),
            GraphEdge(bar_in_1, in_1, GraphEdgeType.INDEX),
            GraphEdge(two_in, in_1, GraphEdgeType.INDEX),
            GraphEdge(bar_out, out_0, GraphEdgeType.INDEX),
            GraphEdge(bar_out, out_1, GraphEdgeType.INDEX)
        ]

        for edge in indexing_edges:
            self.assertTrue(edge in rel_graph_edges,
                            "Could not find edge %s in set of edges:\n%s" % (edge, rel_graph_edges))

        equality_edges = [
            GraphEdge(bar_in_0, bar_out, GraphEdgeType.EQUALITY),
            GraphEdge(bar_in_1, bar_out, GraphEdgeType.EQUALITY),
            GraphEdge(one_in, one_out, GraphEdgeType.EQUALITY),
            GraphEdge(two_in, two_out, GraphEdgeType.EQUALITY)
        ]

        for edge in equality_edges:
            self.assertTrue(edge in rel_graph_edges,
                            "Could not find edge %s in set of edges:\n%s" % (edge, rel_graph_edges))

    def test_column_multi(self):
        column_labels = [['bar', 'bar', 'baz', 'baz'], ['one', 'two', 'one', 'two']]
        tuples = list(zip(*column_labels))
        col_index = pd.MultiIndex.from_tuples(tuples)
        data = [[0, 1, 2, 3], [4, 5, 6, 7]]
        input_df = pd.DataFrame(data, columns=col_index)
        #   bar     baz
        #   one two one two
        # 0   0   1   2   3
        # 1   4   5   6   7
        output_df = input_df.stack().reset_index()
        #    level_0 level_1  bar  baz
        # 0        0     one    0    2
        # 1        0     two    1    3
        # 2        1     one    4    6
        # 3        1     two    5    7

        options = GraphOptions()
        options.COLUMN_NODES = True
        options.ADJACENCY_EDGES = True
        options.EQUALITY_EDGES = True
        options.NODE_TYPES = True
        options.INDEX_EDGES = True
        rel_graph: RelationGraph = RelationGraph(options)
        rel_graph.from_input_output([input_df], output_df)
        rel_graph_edges = rel_graph.edges

        col_nodes = [[GraphNode("I0", '[-2,0]', GraphNodeType.COLUMN),
                      GraphNode("I0", '[-2,1]', GraphNodeType.COLUMN),
                      GraphNode("I0", '[-2,2]', GraphNodeType.COLUMN),
                      GraphNode("I0", '[-2,3]', GraphNodeType.COLUMN)],
                     [GraphNode("I0", '[-1,0]', GraphNodeType.COLUMN),
                      GraphNode("I0", '[-1,1]', GraphNodeType.COLUMN),
                      GraphNode("I0", '[-1,2]', GraphNodeType.COLUMN),
                      GraphNode("I0", '[-1,3]', GraphNodeType.COLUMN)],
                     ]

        adjacency_edges = [
            GraphEdge(col_nodes[0][0], col_nodes[1][0], GraphEdgeType.ADJACENCY),
            GraphEdge(col_nodes[0][0], col_nodes[0][1], GraphEdgeType.ADJACENCY),
            GraphEdge(col_nodes[1][0], col_nodes[1][1], GraphEdgeType.ADJACENCY),
            GraphEdge(col_nodes[1][1], col_nodes[1][2], GraphEdgeType.ADJACENCY),
            GraphEdge(col_nodes[0][1], col_nodes[1][1], GraphEdgeType.ADJACENCY),
            GraphEdge(col_nodes[0][1], col_nodes[0][2], GraphEdgeType.ADJACENCY),
            GraphEdge(col_nodes[0][2], col_nodes[1][2], GraphEdgeType.ADJACENCY),
            GraphEdge(col_nodes[0][2], col_nodes[0][3], GraphEdgeType.ADJACENCY),
            GraphEdge(col_nodes[1][2], col_nodes[1][3], GraphEdgeType.ADJACENCY),
            GraphEdge(col_nodes[0][3], col_nodes[1][3], GraphEdgeType.ADJACENCY)
        ]

        for edge in adjacency_edges:
            self.assertTrue(edge in rel_graph_edges,
                            "Could not find edge %s in set of edges:\n%s" % (edge, rel_graph_edges))

        # indexing edges
        input_coli_elems = [
            [GraphNode("I0", '[0,0]', GraphNodeType.INT),
             GraphNode("I0", '[1,0]', GraphNodeType.INT)],
            [GraphNode("I0", '[0,1]', GraphNodeType.INT),
             GraphNode("I0", '[1,1]', GraphNodeType.INT)],
            [GraphNode("I0", '[0,2]', GraphNodeType.INT),
             GraphNode("I0", '[1,2]', GraphNodeType.INT)],
            [GraphNode("I0", '[0,3]', GraphNodeType.INT),
             GraphNode("I0", '[1,3]', GraphNodeType.INT)]
        ]

        def check_edges(in_nodes, out_nodes, edge_type):
            for in_node in in_nodes:
                for out_node in out_nodes:
                    edge = GraphEdge(in_node, out_node, edge_type)
                    self.assertTrue(edge in rel_graph_edges,
                                    "Could not find edge %s in set of edges:\n%s" % (edge, rel_graph_edges))

        for i in range(4):
            in_nodes = [col_nodes[0][i], col_nodes[1][i]]
            out_nodes = input_coli_elems[i]
            check_edges(in_nodes, out_nodes, GraphEdgeType.INDEX)

        # equality_edges
        bars = [col_nodes[0][0], col_nodes[0][1]]
        bazs = [col_nodes[0][2], col_nodes[0][3]]
        ones = [col_nodes[1][0], col_nodes[1][2]]
        twos = [col_nodes[1][1], col_nodes[1][3]]

        out_01 = GraphNode("O0", '[0,1]', GraphNodeType.STR)
        out_11 = GraphNode("O0", '[1,1]', GraphNodeType.STR)
        out_21 = GraphNode("O0", '[2,1]', GraphNodeType.STR)
        out_31 = GraphNode("O0", '[3,1]', GraphNodeType.STR)

        out_col_2 = GraphNode("O0", '[-1,2]', GraphNodeType.COLUMN)
        out_col_3 = GraphNode("O0", '[-1,3]', GraphNodeType.COLUMN)

        check_edges(bars, [out_col_2], GraphEdgeType.EQUALITY)
        check_edges(bazs, [out_col_3], GraphEdgeType.EQUALITY)

        check_edges(ones, [out_01, out_21], GraphEdgeType.EQUALITY)
        check_edges(twos, [out_11, out_31], GraphEdgeType.EQUALITY)

    def test_no_spurious_for_idx_arg(self):
        df = pd.DataFrame([[5, 2], [2, 3], [2, 0]], columns=["A", "B"])

        options = GraphOptions()
        options.COLUMN_NODES = True
        options.INDEX_NODES = True
        options.ADJACENCY_EDGES = True
        options.EQUALITY_EDGES = True
        options.NODE_TYPES = True
        options.INDEX_EDGES = True
        options.INFLUENCE_EDGES = False

        rel_graph: RelationGraph = RelationGraph(options)
        rel_graph.from_input_output([df, df.columns], df)

        index_type_nodes = [node for node in rel_graph.nodes if node.ntype == GraphNodeType.INDEX]
        column_type_nodes = [node for node in rel_graph.nodes if node.ntype == GraphNodeType.COLUMN]

        self.assertEqual(len(index_type_nodes), 6)
        self.assertEqual(len(column_type_nodes), 4)

    def test_no_spurious_for_list_arg(self):
        df = pd.DataFrame([[5, 2], [2, 3], [2, 0]], columns=["A", "B"])

        options = GraphOptions()
        options.COLUMN_NODES = True
        options.INDEX_NODES = True
        options.ADJACENCY_EDGES = True
        options.EQUALITY_EDGES = True
        options.NODE_TYPES = True
        options.INDEX_EDGES = True

        rel_graph: RelationGraph = RelationGraph(options)
        rel_graph.from_input_output([df, [1, 3, 4]], df)

        index_type_nodes = [node for node in rel_graph.nodes if node.ntype == GraphNodeType.INDEX]
        column_type_nodes = [node for node in rel_graph.nodes if node.ntype == GraphNodeType.COLUMN]

        self.assertEqual(len(index_type_nodes), 6)
        self.assertEqual(len(column_type_nodes), 4)

    def test_series_has_idx_and_cols(self):
        df = pd.DataFrame([[5, 2], [2, 3], [2, 0]], columns=["A", "B"])

        options = GraphOptions()
        options.COLUMN_NODES = True
        options.INDEX_NODES = True
        options.ADJACENCY_EDGES = True
        options.EQUALITY_EDGES = True
        options.NODE_TYPES = True
        options.INDEX_EDGES = True

        rel_graph: RelationGraph = RelationGraph(options)
        rel_graph.from_input_output([df], df["A"])

        index_type_nodes = [node for node in rel_graph.nodes if node.ntype == GraphNodeType.INDEX]
        column_type_nodes = [node for node in rel_graph.nodes if node.ntype == GraphNodeType.COLUMN]

        self.assertEqual(len(index_type_nodes), 6)
        self.assertEqual(len(column_type_nodes), 3)

    def test_groupby_has_artifacts(self):
        df = pd.DataFrame([[5, 2], [2, 3], [2, 0]], columns=["A", "B"])
        output = df.groupby(by="A")

        options = GraphOptions()
        options.COLUMN_NODES = True
        options.INDEX_NODES = True
        options.ADJACENCY_EDGES = True
        options.EQUALITY_EDGES = True
        options.NODE_TYPES = True
        options.INDEX_EDGES = True

        rel_graph: RelationGraph = RelationGraph(options)
        rel_graph.from_input_output([df], output)

        index_type_nodes = [node for node in rel_graph.nodes if node.ntype == GraphNodeType.INDEX]
        column_type_nodes = [node for node in rel_graph.nodes if node.ntype == GraphNodeType.COLUMN]

        self.assertEqual(len(index_type_nodes), 6)
        self.assertEqual(len(column_type_nodes), 6)

    def test_index_name_nodes(self):
        df = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two', 'two'],
                           'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
                           'baz': [1, 2, 3, 4, 5, 6]})
        output = df.pivot(index='foo', columns='bar', values='baz')

        options = GraphOptions()
        options.COLUMN_NODES = True
        options.INDEX_NODES = True
        options.INDEX_NAME_NODES = True
        options.ADJACENCY_EDGES = True
        options.EQUALITY_EDGES = True
        options.NODE_TYPES = True
        options.INDEX_EDGES = False

        rel_graph: RelationGraph = RelationGraph(options)
        rel_graph.from_input_output([df], output)
        index_name_nodes = [node for node in rel_graph.nodes if node.ntype == GraphNodeType.INDEX_NAME]
        column_name_nodes = [node for node in rel_graph.nodes if node.ntype == GraphNodeType.COL_INDEX_NAME]

        self.assertEqual(len(index_name_nodes), 1)
        self.assertEqual(len(column_name_nodes), 1)

    def test_index_name_nodes_multiindex(self):
        df = pd.DataFrame([(389.0, 'fly'), (24.0, 'fly'), (80.5, 'run'), (np.nan, 'jump')],
                          index=pd.MultiIndex.from_tuples(
                              [('bird', 'falcon'), ('bird', 'parrot'), ('mammal', 'lion'),
                               ('mammal', 'monkey')], names=['class', 'name']),
                          columns=pd.MultiIndex.from_tuples([('speed', 'max'), ('species', 'type')]))
        df.columns.names = ['name1', 'name2']

        options = GraphOptions()
        options.COLUMN_NODES = True
        options.INDEX_NODES = True
        options.INDEX_NAME_NODES = True
        options.ADJACENCY_EDGES = True
        options.EQUALITY_EDGES = True
        options.NODE_TYPES = True
        options.INDEX_EDGES = False

        rel_graph: RelationGraph = RelationGraph(options)
        rel_graph.from_input_output([df], df)
        index_name_nodes = [node for node in rel_graph.nodes if node.ntype == GraphNodeType.INDEX_NAME]
        column_name_nodes = [node for node in rel_graph.nodes if node.ntype == GraphNodeType.COL_INDEX_NAME]

        self.assertEqual(len(index_name_nodes), 4)  # Both in the input and output, so x2
        self.assertEqual(len(column_name_nodes), 4)  # Both in the input and output, so x2

    def test_index_name_equality_edges(self):
        df = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two', 'two'],
                           'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
                           'baz': [1, 2, 3, 4, 5, 6]})
        output = df.pivot(index='foo', columns='bar', values='baz')

        options = GraphOptions()
        options.COLUMN_NODES = True
        options.INDEX_NODES = True
        options.INDEX_NAME_NODES = True
        options.ADJACENCY_EDGES = False
        options.EQUALITY_EDGES = True
        options.NODE_TYPES = True
        options.INDEX_EDGES = False
        rel_graph: RelationGraph = RelationGraph(options)
        rel_graph.from_input_output([df], output)
        inp_col_nodes = [node for node in rel_graph.nodes if node.ntype == GraphNodeType.COLUMN
                         and node.source.startswith("I")]
        out_idx_name_nodes = [node for node in rel_graph.nodes if node.ntype == GraphNodeType.INDEX_NAME
                              and node.source.startswith("O")]
        out_col_idx_name_nodes = [node for node in rel_graph.nodes if node.ntype == GraphNodeType.COL_INDEX_NAME
                                  and node.source.startswith("O")]

        def check_edge_exists(in_node: GraphNode, out_node: GraphNode, graph: RelationGraph):
            for e in graph.edges:
                if (e.node1 == in_node and e.node2 == out_node) or (e.node1 == out_node and e.node2 == in_node):
                    return True

            return False

        inp_foo_node = [i for i in inp_col_nodes if i.identifier == '[-1,0]'][0]
        inp_bar_node = [i for i in inp_col_nodes if i.identifier == '[-1,1]'][0]
        out_foo_node = [i for i in out_idx_name_nodes if i.identifier == '[-1,-1]'][0]
        out_bar_node = [i for i in out_col_idx_name_nodes if i.identifier == '[-1,-1]'][0]

        self.assertTrue(check_edge_exists(inp_foo_node, out_foo_node, rel_graph))
        self.assertTrue(check_edge_exists(inp_bar_node, out_bar_node, rel_graph))

    def test_substr_edges(self):
        df = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two', 'two'],
                           'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
                           'baz': [1, 2, 3, 4, 5, 6]})
        out = pd.DataFrame({"mrr": ["wo", "no"],
                            'asdasd': ["A_1", "B_4"],
                            'nostr': [33, 12]})

        options = GraphOptions()
        options.SUBSTR_EDGES = True

        rel_graph: RelationGraph = RelationGraph(options)
        rel_graph.from_input_output([df], out)

        def check_edges(in_nodes, out_nodes):
            for in_node in in_nodes:
                for out_node in out_nodes:
                    edge = GraphEdge(in_node, out_node, GraphEdgeType.SUBSTR)
                    self.assertTrue(edge in rel_graph.edges,
                                    "Could not find edge %s in set of edges:\n%s" % (edge, rel_graph.edges))

        # test substrings from out to in
        two_nodes = [GraphNode("I0", '[3,0]', GraphNodeType.STR), GraphNode("I0", '[4,0]', GraphNodeType.STR),
                     GraphNode("I0", '[5,0]', GraphNodeType.STR)]
        wo_node = GraphNode("O0", '[0,0]', GraphNodeType.STR)

        check_edges(two_nodes, [wo_node])

        # test substrings from in to out
        A_in = [GraphNode("I0", '[0,1]', GraphNodeType.STR), GraphNode("I0", '[3,1]', GraphNodeType.STR)]
        A_out = [GraphNode("O0", '[0,1]', GraphNodeType.STR)]
        B_in = [GraphNode("I0", '[1,1]', GraphNodeType.STR), GraphNode("I0", '[4,1]', GraphNodeType.STR)]
        B_out = [GraphNode("O0", '[1,1]', GraphNodeType.STR)]
        check_edges(A_in, A_out)
        check_edges(B_in, B_out)

        # test substrings involving non-strings
        one_in = [GraphNode("I0", '[0,2]', GraphNodeType.INT), GraphNode("I0", '[1,-1]', GraphNodeType.INDEX)]
        one_out = [GraphNode("O0", '[0,1]', GraphNodeType.STR)]
        four_in = [GraphNode("I0", '[3,2]', GraphNodeType.INT), GraphNode("I0", '[4,-1]', GraphNodeType.INDEX)]
        four_out = [GraphNode("O0", '[1,1]', GraphNodeType.STR)]
        check_edges(one_in, one_out)
        check_edges(four_in, four_out)

        # test nothing else
        self.assertEqual(11, len([e for e in rel_graph.edges if e.etype == GraphEdgeType.SUBSTR]))
