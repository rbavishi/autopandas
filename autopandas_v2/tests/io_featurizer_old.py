import unittest

import pandas as pd
import numpy as np
from autopandas_v2.ml.featurization.io_featurizer_old import RelationGraphNode, RelationGraph, RelationGraphEdge, \
    RelationGraphEdgeType, RelationGraphNodeType, RelationGraphOptions

get_node_type = RelationGraphNodeType.get_node_type


class TestRelationGraphFeaturizer(unittest.TestCase):
    def test_basic_max(self):
        input_df = pd.DataFrame([[1, 2], [2, 3], [2, 0]])
        input_00 = RelationGraphNode("I0", (0, 0), get_node_type(input_df.iat[0, 0]))
        input_01 = RelationGraphNode("I0", (0, 1), get_node_type(input_df.iat[0, 1]))
        input_10 = RelationGraphNode("I0", (1, 0), get_node_type(input_df.iat[1, 0]))
        input_11 = RelationGraphNode("I0", (1, 1), get_node_type(input_df.iat[1, 1]))
        input_20 = RelationGraphNode("I0", (2, 0), get_node_type(input_df.iat[2, 0]))
        input_21 = RelationGraphNode("I0", (2, 1), get_node_type(input_df.iat[2, 1]))
        output_df = pd.DataFrame([[2, 3]])
        output_00 = RelationGraphNode("O0", (0, 0), get_node_type(output_df.iat[0, 0]))
        output_01 = RelationGraphNode("O0", (0, 1), get_node_type(output_df.iat[0, 1]))
        options = RelationGraphOptions()
        options.NODE_TYPES = True
        rel_graph: RelationGraph = RelationGraph.build_relation_graph([input_df], output_df, options)
        rel_graph_edges = rel_graph.edges
        #  positional edges
        positional_edges = [
            RelationGraphEdge(input_00, input_01, RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(input_00, input_10, RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(input_10, input_11, RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(input_10, input_20, RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(input_20, input_21, RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(input_01, input_11, RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(input_11, input_21, RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(output_00, output_01, RelationGraphEdgeType.ADJACENCY)
        ]
        for edge in positional_edges:
            self.assertTrue(edge in rel_graph_edges,
                            "Could not find edge %s in set of edges:\n%s" % (edge, rel_graph_edges))
        #  equality edges
        equality_edges = [
            RelationGraphEdge(input_10, output_00, RelationGraphEdgeType.EQUALITY),
            RelationGraphEdge(input_20, output_00, RelationGraphEdgeType.EQUALITY),
            RelationGraphEdge(input_01, output_00, RelationGraphEdgeType.EQUALITY),  # redundant
            RelationGraphEdge(input_11, output_01, RelationGraphEdgeType.EQUALITY)
        ]
        for edge in equality_edges:
            self.assertTrue(edge in rel_graph_edges,
                            "Could not find edge %s in set of edges:\n%s" % (edge, rel_graph_edges))

    def test_max_series(self):
        input_df = pd.DataFrame([[1, 2], [2, 3], [2, 0]])
        input_00 = RelationGraphNode("I0", (0, 0), get_node_type(input_df.iat[0, 0]))
        input_01 = RelationGraphNode("I0", (0, 1), get_node_type(input_df.iat[0, 1]))
        input_10 = RelationGraphNode("I0", (1, 0), get_node_type(input_df.iat[1, 0]))
        input_11 = RelationGraphNode("I0", (1, 1), get_node_type(input_df.iat[1, 1]))
        input_20 = RelationGraphNode("I0", (2, 0), get_node_type(input_df.iat[2, 0]))
        input_21 = RelationGraphNode("I0", (2, 1), get_node_type(input_df.iat[2, 1]))
        output = pd.DataFrame.max(input_df)
        output_00 = RelationGraphNode("O0", (0, 0), get_node_type(output.iat[0]))
        output_10 = RelationGraphNode("O0", (1, 0), get_node_type(output.iat[1]))
        options = RelationGraphOptions()
        options.NODE_TYPES = True
        rel_graph: RelationGraph = RelationGraph.build_relation_graph([input_df], output, options)
        rel_graph_edges = rel_graph.edges
        #  positional edges
        positional_edges = [
            RelationGraphEdge(input_00, input_01, RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(input_00, input_10, RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(input_10, input_11, RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(input_10, input_20, RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(input_20, input_21, RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(input_01, input_11, RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(input_11, input_21, RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(output_00, output_10, RelationGraphEdgeType.ADJACENCY)
        ]
        for edge in positional_edges:
            self.assertTrue(edge in rel_graph_edges,
                            "Could not find edge %s in set of edges:\n%s" % (edge, rel_graph_edges))
        #  equality edges
        equality_edges = [
            RelationGraphEdge(input_10, output_00, RelationGraphEdgeType.EQUALITY),
            RelationGraphEdge(input_20, output_00, RelationGraphEdgeType.EQUALITY),
            RelationGraphEdge(input_01, output_00, RelationGraphEdgeType.EQUALITY),  # redundant
            RelationGraphEdge(input_11, output_10, RelationGraphEdgeType.EQUALITY)
        ]
        for edge in equality_edges:
            self.assertTrue(edge in rel_graph_edges,
                            "Could not find edge %s in set of edges:\n%s" % (edge, rel_graph_edges))

    def test_values(self):
        input_df = pd.DataFrame([[1, 2], [3, 4]])
        input_00 = RelationGraphNode("I0", (0, 0), get_node_type(input_df.iat[0, 0]))
        input_01 = RelationGraphNode("I0", (0, 1), get_node_type(input_df.iat[0, 1]))
        input_10 = RelationGraphNode("I0", (1, 0), get_node_type(input_df.iat[1, 0]))
        input_11 = RelationGraphNode("I0", (1, 1), get_node_type(input_df.iat[1, 1]))
        output = input_df.values
        output_00 = RelationGraphNode("O0", (0, 0), get_node_type(output[0, 0]))
        output_01 = RelationGraphNode("O0", (0, 1), get_node_type(output[0, 1]))
        output_10 = RelationGraphNode("O0", (1, 0), get_node_type(output[1, 0]))
        output_11 = RelationGraphNode("O0", (1, 1), get_node_type(output[1, 1]))
        options = RelationGraphOptions()
        options.NODE_TYPES = True
        rel_graph: RelationGraph = RelationGraph.build_relation_graph([input_df], output, options)
        rel_graph_edges = rel_graph.edges
        #  positional edges
        positional_edges = [
            RelationGraphEdge(input_00, input_01, RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(input_00, input_10, RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(input_10, input_11, RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(input_01, input_11, RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(output_00, output_01, RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(output_00, output_10, RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(output_10, output_11, RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(output_01, output_11, RelationGraphEdgeType.ADJACENCY)
        ]
        for edge in positional_edges:
            self.assertTrue(edge in rel_graph_edges,
                            "Could not find edge %s in set of edges:\n%s" % (edge, rel_graph_edges))
        equality_edges = [
            RelationGraphEdge(input_00, output_00, RelationGraphEdgeType.EQUALITY),
            RelationGraphEdge(input_10, output_10, RelationGraphEdgeType.EQUALITY),
            RelationGraphEdge(input_01, output_01, RelationGraphEdgeType.EQUALITY),
            RelationGraphEdge(input_11, output_11, RelationGraphEdgeType.EQUALITY)
        ]
        for edge in equality_edges:
            self.assertTrue(edge in rel_graph_edges,
                            "Could not find edge %s in set of edges:\n%s" % (edge, rel_graph_edges))

    def test_dict(self):
        input_df = pd.DataFrame([[1, 2], [3, 4]])
        input_00 = RelationGraphNode("I0", (0, 0), get_node_type(input_df.iat[0, 0]))
        input_01 = RelationGraphNode("I0", (0, 1), get_node_type(input_df.iat[0, 1]))
        input_10 = RelationGraphNode("I0", (1, 0), get_node_type(input_df.iat[1, 0]))
        input_11 = RelationGraphNode("I0", (1, 1), get_node_type(input_df.iat[1, 1]))
        output = {"A": [1, 3], "B": [2, 4]}
        output_00 = RelationGraphNode("O0", (0, 0), get_node_type(output['A'][0]))
        output_01 = RelationGraphNode("O0", (0, 1), get_node_type(output['B'][0]))
        output_10 = RelationGraphNode("O0", (1, 0), get_node_type(output['A'][1]))
        output_11 = RelationGraphNode("O0", (1, 1), get_node_type(output['B'][1]))
        options = RelationGraphOptions()
        options.NODE_TYPES = True
        rel_graph: RelationGraph = RelationGraph.build_relation_graph([input_df], output, options)
        rel_graph_edges = rel_graph.edges
        positional_edges = [
            RelationGraphEdge(input_00, input_01, RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(input_00, input_10, RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(input_10, input_11, RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(input_01, input_11, RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(output_00, output_01, RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(output_00, output_10, RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(output_10, output_11, RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(output_01, output_11, RelationGraphEdgeType.ADJACENCY)
        ]
        for edge in positional_edges:
            self.assertTrue(edge in rel_graph_edges,
                            "Could not find edge %s in set of edges:\n%s" % (edge, rel_graph_edges))
        equality_edges = [
            RelationGraphEdge(input_00, output_00, RelationGraphEdgeType.EQUALITY),
            RelationGraphEdge(input_10, output_10, RelationGraphEdgeType.EQUALITY),
            RelationGraphEdge(input_01, output_01, RelationGraphEdgeType.EQUALITY),
            RelationGraphEdge(input_11, output_11, RelationGraphEdgeType.EQUALITY)
        ]
        for edge in equality_edges:
            self.assertTrue(edge in rel_graph_edges,
                            "Could not find edge %s in set of edges:\n%s" % (edge, rel_graph_edges))

    def test_groupby_output(self):
        input_df = pd.DataFrame({
            "Name": ["Alice", "Bob", "Mallory", "Mallory", "Bob", "Mallory"],
            "City": ["Seattle", "Seattle", "Portland", "Seattle", "Seattle", "Portland"]})
        output = input_df.groupby("Name")
        options = RelationGraphOptions()
        options.NODE_TYPES = True
        options.ADJACENCY_EDGES = False
        rel_graph: RelationGraph = RelationGraph.build_relation_graph([input_df], output, options)
        rel_graph_edges = rel_graph.edges
        alice_nodes_in = [
            RelationGraphNode("I0", (0, 0), RelationGraphNodeType.STR)
        ]
        alice_nodes_out = [
            RelationGraphNode("O0", (0, 0), RelationGraphNodeType.STR)
        ]
        bob_nodes_in = [
            RelationGraphNode("I0", (1, 0), RelationGraphNodeType.STR),
            RelationGraphNode("I0", (4, 0), RelationGraphNodeType.STR)
        ]
        bob_nodes_out = [
            RelationGraphNode("O1", (0, 0), RelationGraphNodeType.STR),
            RelationGraphNode("O1", (1, 0), RelationGraphNodeType.STR)
        ]
        mallory_nodes_in = [
            RelationGraphNode("I0", (2, 0), RelationGraphNodeType.STR),
            RelationGraphNode("I0", (3, 0), RelationGraphNodeType.STR),
            RelationGraphNode("I0", (5, 0), RelationGraphNodeType.STR)
        ]
        mallory_nodes_out = [
            RelationGraphNode("O2", (0, 0), RelationGraphNodeType.STR),
            RelationGraphNode("O2", (1, 0), RelationGraphNodeType.STR),
            RelationGraphNode("O2", (2, 0), RelationGraphNodeType.STR)
        ]
        seattle_nodes_in = [
            RelationGraphNode("I0", (0, 1), RelationGraphNodeType.STR),
            RelationGraphNode("I0", (1, 1), RelationGraphNodeType.STR),
            RelationGraphNode("I0", (3, 1), RelationGraphNodeType.STR),
            RelationGraphNode("I0", (4, 1), RelationGraphNodeType.STR),
        ]
        seattle_nodes_out = [
            RelationGraphNode("O0", (0, 1), RelationGraphNodeType.STR),
            RelationGraphNode("O1", (0, 1), RelationGraphNodeType.STR),
            RelationGraphNode("O2", (1, 1), RelationGraphNodeType.STR)
        ]
        portland_nodes_in = [
            RelationGraphNode("I0", (2, 1), RelationGraphNodeType.STR),
            RelationGraphNode("I0", (5, 1), RelationGraphNodeType.STR)
        ]
        portland_nodes_out = [
            RelationGraphNode("O2", (0, 1), RelationGraphNodeType.STR),
            RelationGraphNode("O2", (2, 1), RelationGraphNodeType.STR)
        ]

        def check_edges(in_nodes, out_nodes):
            for in_node in in_nodes:
                for out_node in out_nodes:
                    edge = RelationGraphEdge(in_node, out_node, RelationGraphEdgeType.EQUALITY)
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
        options = RelationGraphOptions()
        options.NODE_TYPES = True
        options.ADJACENCY_EDGES = False
        rel_graph: RelationGraph = RelationGraph.build_relation_graph([input_], output, options)
        rel_graph_edges = rel_graph.edges
        alice_nodes_in = [
            RelationGraphNode("I0", (0, 0), RelationGraphNodeType.STR)
        ]

        alice_nodes_out = [
            RelationGraphNode("O0", (0, 0), RelationGraphNodeType.STR)
        ]

        bob_nodes_in = [
            RelationGraphNode("I1", (0, 0), RelationGraphNodeType.STR),
            RelationGraphNode("I1", (1, 0), RelationGraphNodeType.STR)
        ]
        bob_nodes_out = [
            RelationGraphNode("O0", (1, 0), RelationGraphNodeType.STR)
        ]

        mallory_nodes_in = [
            RelationGraphNode("I2", (0, 0), RelationGraphNodeType.STR),
            RelationGraphNode("I2", (1, 0), RelationGraphNodeType.STR),
            RelationGraphNode("I2", (2, 0), RelationGraphNodeType.STR)
        ]
        mallory_nodes_out = [
            RelationGraphNode("O0", (2, 0), RelationGraphNodeType.STR)
        ]

        def check_edges(in_nodes, out_nodes):
            for in_node in in_nodes:
                for out_node in out_nodes:
                    edge = RelationGraphEdge(in_node, out_node, RelationGraphEdgeType.EQUALITY)
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
        options = RelationGraphOptions()
        options.COLUMN_NODES = True
        options.INDEX_NODES = True
        options.ADJACENCY_EDGES = True
        options.EQUALITY_EDGES = True
        options.NODE_TYPES = True
        options.INDEX_EDGES = True
        rel_graph: RelationGraph = RelationGraph.build_relation_graph([input_df], output_df, options)
        rel_graph_edges = rel_graph.edges

        bar_in_0 = RelationGraphNode("I0", (0, -2), RelationGraphNodeType.INDEX)
        bar_in_1 = RelationGraphNode("I0", (1, -2), RelationGraphNodeType.INDEX)
        bar_out = RelationGraphNode("O0", (0, -1), RelationGraphNodeType.INDEX)

        one_in = RelationGraphNode("I0", (0, -1), RelationGraphNodeType.INDEX)
        two_in = RelationGraphNode("I0", (1, -1), RelationGraphNodeType.INDEX)

        one_out = RelationGraphNode("O0", (-1, 0), RelationGraphNodeType.COLUMN)
        two_out = RelationGraphNode("O0", (-1, 1), RelationGraphNodeType.COLUMN)

        in_0 = RelationGraphNode("I0", (0, 0), RelationGraphNodeType.INT)
        in_1 = RelationGraphNode("I0", (1, 0), RelationGraphNodeType.INT)

        out_0 = RelationGraphNode("O0", (0, 0), RelationGraphNodeType.INT)
        out_1 = RelationGraphNode("O0", (0, 1), RelationGraphNodeType.INT)

        adjacency_edges = [
            RelationGraphEdge(bar_in_0, bar_in_1, RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(bar_in_0, one_in, RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(bar_in_1, two_in, RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(one_in, two_in, RelationGraphEdgeType.ADJACENCY)
        ]

        for edge in adjacency_edges:
            self.assertTrue(edge in rel_graph_edges,
                            "Could not find edge %s in set of edges:\n%s" % (edge, rel_graph_edges))
        indexing_edges = [
            RelationGraphEdge(bar_in_0, in_0, RelationGraphEdgeType.INDEX),
            RelationGraphEdge(one_in, in_0, RelationGraphEdgeType.INDEX),
            RelationGraphEdge(bar_in_1, in_1, RelationGraphEdgeType.INDEX),
            RelationGraphEdge(two_in, in_1, RelationGraphEdgeType.INDEX),
            RelationGraphEdge(bar_out, out_0, RelationGraphEdgeType.INDEX),
            RelationGraphEdge(bar_out, out_1, RelationGraphEdgeType.INDEX)
        ]

        for edge in indexing_edges:
            self.assertTrue(edge in rel_graph_edges,
                            "Could not find edge %s in set of edges:\n%s" % (edge, rel_graph_edges))

        equality_edges = [
            RelationGraphEdge(bar_in_0, bar_out, RelationGraphEdgeType.EQUALITY),
            RelationGraphEdge(bar_in_1, bar_out, RelationGraphEdgeType.EQUALITY),
            RelationGraphEdge(one_in, one_out, RelationGraphEdgeType.EQUALITY),
            RelationGraphEdge(two_in, two_out, RelationGraphEdgeType.EQUALITY)
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

        options = RelationGraphOptions()
        options.COLUMN_NODES = True
        options.ADJACENCY_EDGES = True
        options.EQUALITY_EDGES = True
        options.NODE_TYPES = True
        options.INDEX_EDGES = True
        rel_graph: RelationGraph = RelationGraph.build_relation_graph([input_df], output_df, options)
        rel_graph_edges = rel_graph.edges

        col_nodes = [[RelationGraphNode("I0", (-2, 0), RelationGraphNodeType.COLUMN),
                      RelationGraphNode("I0", (-2, 1), RelationGraphNodeType.COLUMN),
                      RelationGraphNode("I0", (-2, 2), RelationGraphNodeType.COLUMN),
                      RelationGraphNode("I0", (-2, 3), RelationGraphNodeType.COLUMN)],
                     [RelationGraphNode("I0", (-1, 0), RelationGraphNodeType.COLUMN),
                      RelationGraphNode("I0", (-1, 1), RelationGraphNodeType.COLUMN),
                      RelationGraphNode("I0", (-1, 2), RelationGraphNodeType.COLUMN),
                      RelationGraphNode("I0", (-1, 3), RelationGraphNodeType.COLUMN)],
                     ]

        adjacency_edges = [
            RelationGraphEdge(col_nodes[0][0], col_nodes[1][0], RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(col_nodes[0][0], col_nodes[0][1], RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(col_nodes[1][0], col_nodes[1][1], RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(col_nodes[1][1], col_nodes[1][2], RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(col_nodes[0][1], col_nodes[1][1], RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(col_nodes[0][1], col_nodes[0][2], RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(col_nodes[0][2], col_nodes[1][2], RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(col_nodes[0][2], col_nodes[0][3], RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(col_nodes[1][2], col_nodes[1][3], RelationGraphEdgeType.ADJACENCY),
            RelationGraphEdge(col_nodes[0][3], col_nodes[1][3], RelationGraphEdgeType.ADJACENCY)
        ]

        for edge in adjacency_edges:
            self.assertTrue(edge in rel_graph_edges,
                            "Could not find edge %s in set of edges:\n%s" % (edge, rel_graph_edges))

        # indexing edges
        input_coli_elems = [
            [RelationGraphNode("I0", (0, 0), RelationGraphNodeType.INT),
             RelationGraphNode("I0", (1, 0), RelationGraphNodeType.INT)],
            [RelationGraphNode("I0", (0, 1), RelationGraphNodeType.INT),
             RelationGraphNode("I0", (1, 1), RelationGraphNodeType.INT)],
            [RelationGraphNode("I0", (0, 2), RelationGraphNodeType.INT),
             RelationGraphNode("I0", (1, 2), RelationGraphNodeType.INT)],
            [RelationGraphNode("I0", (0, 3), RelationGraphNodeType.INT),
             RelationGraphNode("I0", (1, 3), RelationGraphNodeType.INT)]
        ]

        def check_edges(in_nodes, out_nodes, edge_type):
            for in_node in in_nodes:
                for out_node in out_nodes:
                    edge = RelationGraphEdge(in_node, out_node, edge_type)
                    self.assertTrue(edge in rel_graph_edges,
                                    "Could not find edge %s in set of edges:\n%s" % (edge, rel_graph_edges))

        for i in range(4):
            in_nodes = [col_nodes[0][i], col_nodes[1][i]]
            out_nodes = input_coli_elems[i]
            check_edges(in_nodes, out_nodes, RelationGraphEdgeType.INDEX)

        # equality_edges
        bars = [col_nodes[0][0], col_nodes[0][1]]
        bazs = [col_nodes[0][2], col_nodes[0][3]]
        ones = [col_nodes[1][0], col_nodes[1][2]]
        twos = [col_nodes[1][1], col_nodes[1][3]]

        out_01 = RelationGraphNode("O0", (0, 1), RelationGraphNodeType.STR)
        out_11 = RelationGraphNode("O0", (1, 1), RelationGraphNodeType.STR)
        out_21 = RelationGraphNode("O0", (2, 1), RelationGraphNodeType.STR)
        out_31 = RelationGraphNode("O0", (3, 1), RelationGraphNodeType.STR)

        out_col_2 = RelationGraphNode("O0", (-1, 2), RelationGraphNodeType.COLUMN)
        out_col_3 = RelationGraphNode("O0", (-1, 3), RelationGraphNodeType.COLUMN)

        check_edges(bars, [out_col_2], RelationGraphEdgeType.EQUALITY)
        check_edges(bazs, [out_col_3], RelationGraphEdgeType.EQUALITY)

        check_edges(ones, [out_01, out_21], RelationGraphEdgeType.EQUALITY)
        check_edges(twos, [out_11, out_31], RelationGraphEdgeType.EQUALITY)

    def test_no_spurious_for_idx_arg(self):
        df = pd.DataFrame([[5, 2], [2, 3], [2, 0]], columns = ["A", "B"])

        options = RelationGraphOptions()
        options.COLUMN_NODES = True
        options.INDEX_NODES = True
        options.ADJACENCY_EDGES = True
        options.EQUALITY_EDGES = True
        options.NODE_TYPES = True
        options.INDEX_EDGES = True
        options.INFLUENCE_EDGES = False

        rel_graph: RelationGraph = RelationGraph.build_relation_graph([df, df.columns], df, options)

        index_type_nodes = [node for node in rel_graph.nodes if node._type == RelationGraphNodeType.INDEX]
        column_type_nodes = [node for node in rel_graph.nodes if node._type == RelationGraphNodeType.COLUMN]

        self.assertEqual(len(index_type_nodes), 6)
        self.assertEqual(len(column_type_nodes), 4)

    def test_no_spurious_for_list_arg(self):
        df = pd.DataFrame([[5, 2], [2, 3], [2, 0]], columns = ["A", "B"])

        options = RelationGraphOptions()
        options.COLUMN_NODES = True
        options.INDEX_NODES = True
        options.ADJACENCY_EDGES = True
        options.EQUALITY_EDGES = True
        options.NODE_TYPES = True
        options.INDEX_EDGES = True

        rel_graph: RelationGraph = RelationGraph.build_relation_graph([df, [1,3,4]], df, options)

        index_type_nodes = [node for node in rel_graph.nodes if node._type == RelationGraphNodeType.INDEX]
        column_type_nodes = [node for node in rel_graph.nodes if node._type == RelationGraphNodeType.COLUMN]

        self.assertEqual(len(index_type_nodes), 6)
        self.assertEqual(len(column_type_nodes), 4)

    def test_series_has_idx_and_cols(self):
        df = pd.DataFrame([[5, 2], [2, 3], [2, 0]], columns = ["A", "B"])

        options = RelationGraphOptions()
        options.COLUMN_NODES = True
        options.INDEX_NODES = True
        options.ADJACENCY_EDGES = True
        options.EQUALITY_EDGES = True
        options.NODE_TYPES = True
        options.INDEX_EDGES = True

        rel_graph: RelationGraph = RelationGraph.build_relation_graph([df], df["A"], options)

        index_type_nodes = [node for node in rel_graph.nodes if node._type == RelationGraphNodeType.INDEX]
        column_type_nodes = [node for node in rel_graph.nodes if node._type == RelationGraphNodeType.COLUMN]

        self.assertEqual(len(index_type_nodes), 6)
        self.assertEqual(len(column_type_nodes), 3)

    def test_groupby_has_artifacts(self):
        df = pd.DataFrame([[5, 2], [2, 3], [2, 0]], columns = ["A", "B"])
        output = df.groupby(by="A")

        options = RelationGraphOptions()
        options.COLUMN_NODES = True
        options.INDEX_NODES = True
        options.ADJACENCY_EDGES = True
        options.EQUALITY_EDGES = True
        options.NODE_TYPES = True
        options.INDEX_EDGES = True

        rel_graph: RelationGraph = RelationGraph.build_relation_graph([df], output, options)

        index_type_nodes = [node for node in rel_graph.nodes if node._type == RelationGraphNodeType.INDEX]
        column_type_nodes = [node for node in rel_graph.nodes if node._type == RelationGraphNodeType.COLUMN]

        self.assertEqual(len(index_type_nodes), 6)
        self.assertEqual(len(column_type_nodes), 6)

    def test_index_name_nodes(self):
        df = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two', 'two'],
                           'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
                           'baz': [1, 2, 3, 4, 5, 6]})
        output = df.pivot(index='foo', columns='bar', values='baz')

        options = RelationGraphOptions()
        options.COLUMN_NODES = True
        options.INDEX_NODES = True
        options.INDEX_NAME_NODES = True
        options.ADJACENCY_EDGES = True
        options.EQUALITY_EDGES = True
        options.NODE_TYPES = True
        options.INDEX_EDGES = False

        rel_graph: RelationGraph = RelationGraph.build_relation_graph([df], output, options)
        index_name_nodes = [node for node in rel_graph.nodes if node._type == RelationGraphNodeType.INDEX_NAME]
        column_name_nodes = [node for node in rel_graph.nodes if node._type == RelationGraphNodeType.COL_INDEX_NAME]

        self.assertEqual(len(index_name_nodes), 1)
        self.assertEqual(len(column_name_nodes), 1)

    def test_index_name_nodes_multiindex(self):
        df = pd.DataFrame([(389.0, 'fly'), (24.0, 'fly'), (80.5, 'run'), (np.nan, 'jump')],
                          index=pd.MultiIndex.from_tuples(
                              [('bird', 'falcon'), ('bird', 'parrot'), ('mammal', 'lion'),
                               ('mammal', 'monkey')], names=['class', 'name']),
                          columns=pd.MultiIndex.from_tuples([('speed', 'max'), ('species', 'type')]))
        df.columns.names = ['name1', 'name2']

        options = RelationGraphOptions()
        options.COLUMN_NODES = True
        options.INDEX_NODES = True
        options.INDEX_NAME_NODES = True
        options.ADJACENCY_EDGES = True
        options.EQUALITY_EDGES = True
        options.NODE_TYPES = True
        options.INDEX_EDGES = False

        rel_graph: RelationGraph = RelationGraph.build_relation_graph([df], df, options)
        index_name_nodes = [node for node in rel_graph.nodes if node._type == RelationGraphNodeType.INDEX_NAME]
        column_name_nodes = [node for node in rel_graph.nodes if node._type == RelationGraphNodeType.COL_INDEX_NAME]

        self.assertEqual(len(index_name_nodes), 4)  # Both in the input and output, so x2
        self.assertEqual(len(column_name_nodes), 4)  # Both in the input and output, so x2

    def test_index_name_equality_edges(self):
        df = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two', 'two'],
                           'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
                           'baz': [1, 2, 3, 4, 5, 6]})
        output = df.pivot(index='foo', columns='bar', values='baz')

        options = RelationGraphOptions()
        options.COLUMN_NODES = True
        options.INDEX_NODES = True
        options.INDEX_NAME_NODES = True
        options.ADJACENCY_EDGES = False
        options.EQUALITY_EDGES = True
        options.NODE_TYPES = True
        options.INDEX_EDGES = False
        rel_graph: RelationGraph = RelationGraph.build_relation_graph([df], output, options)
        inp_col_nodes = [node for node in rel_graph.nodes if node._type == RelationGraphNodeType.COLUMN
                         and node.dfindex.startswith("I")]
        out_idx_name_nodes = [node for node in rel_graph.nodes if node._type == RelationGraphNodeType.INDEX_NAME
                              and node.dfindex.startswith("O")]
        out_col_idx_name_nodes = [node for node in rel_graph.nodes if node._type == RelationGraphNodeType.COL_INDEX_NAME
                                  and node.dfindex.startswith("O")]

        def check_edge_exists(in_node: RelationGraphNode, out_node: RelationGraphNode, graph: RelationGraph):
            for e in graph.edges:
                if (e.node1 == in_node and e.node2 == out_node) or (e.node1 == out_node and e.node2 == in_node):
                    return True

            return False

        inp_foo_node = [i for i in inp_col_nodes if i.pos == (-1, 0)][0]
        inp_bar_node = [i for i in inp_col_nodes if i.pos == (-1, 1)][0]
        out_foo_node = [i for i in out_idx_name_nodes if i.pos == (-1, -1)][0]
        out_bar_node = [i for i in out_col_idx_name_nodes if i.pos == (-1, -1)][0]

        self.assertTrue(check_edge_exists(inp_foo_node, out_foo_node, rel_graph))
        self.assertTrue(check_edge_exists(inp_bar_node, out_bar_node, rel_graph))
