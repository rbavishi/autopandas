import collections
import copy
import itertools
import logging
from typing import Set, Any, Dict, List

import numpy as np
import pandas as pd

from autopandas_v2.ml.featurization.edge_collections import EdgeCollection
from autopandas_v2.ml.featurization.graph import GraphNode, GraphEdgeType, GraphNodeType
from autopandas_v2.ml.featurization.options import GraphOptions
from autopandas_v2.utils import logger


class GraphNodeCollection:
    def __init__(self, obj: Any, type_str: str, source: str, options: GraphOptions, **kwargs):
        self.obj = obj
        self.type_str = type_str
        self.source = source
        self.options = options
        self.kwargs = kwargs

        self.nodes: Set[GraphNode] = set()
        self.representor: GraphNode = None

        #  For equality edges
        self.value_map: Dict[Any, Set[GraphNode]] = collections.defaultdict(set)

        #  Fire in the hole!
        self.init()

    def init(self):
        self.add_nodes()

    def get_node(self, val: Any, identifier='', options=None):
        if options is None:
            options = self.options

        return GraphNode.from_obj(val, source=self.source, identifier=identifier, options=options)

    def add_node(self, node: GraphNode):
        self.nodes.add(node)

    def add_value(self, val: Any, node: GraphNode):
        if self.options.EQUALITY_EDGES:
            self.value_map[val].add(node)

    def add_nodes(self):
        node = self.get_node(self.obj)
        self.add_node(node)
        self.add_value(self.obj, node)

    def setup_representor(self, collector: EdgeCollection):
        if self.representor is not None:
            return self.representor

        self.representor = GraphNode(self.source, '', GraphNodeType.REPRESENTOR)
        for n in self.nodes:
            collector.add_edge(self.representor, n, GraphEdgeType.REPRESENTOR)
            collector.add_edge(n, self.representor, GraphEdgeType.REPRESENTED)

        self.add_node(self.representor)
        return self.representor

    def add_internal_edges(self, collector: EdgeCollection):
        pass

    def add_external_edges(self, other: 'GraphNodeCollection', collector: EdgeCollection, is_reverse=False):
        if is_reverse:
            return

        if self.options.EQUALITY_EDGES and self.source[0] != other.source[0]:
            #  We only add equality edges between collections from different kinds of sources
            #  I don't see much point in having equality edges between say the groupby groups produced
            #  We want to capture relationships between the input and output, not amongst the outputs themselves
            for val1, nodes1 in self.value_map.items():
                if val1 in other.value_map:
                    val2 = val1
                    nodes2 = other.value_map[val2]
                    try:
                        #  This can fail for NaNs etc.
                        if val1 == val2:
                            for n1, n2 in itertools.product(nodes1, nodes2):
                                collector.add_edge(n1, n2, GraphEdgeType.EQUALITY, directed=False)

                    except Exception as e:
                        logger.err("Error comparing {} and {}".format(val1, val2))
                        logging.exception(e)

        if self.options.SUBSTR_EDGES and self.source[0] != other.source[0]:
            #  We only add substr edges between collections from different kinds of sources.
            #  The reasoning is the same as in the equality edges case
            for val1, nodes1 in self.value_map.items():
                for val2, nodes2 in other.value_map.items():
                    if isinstance(val1, str) or isinstance(val2, str):
                        # if (str(val1) in str(val2)) or (str(val2) in str(val1)):
                        if str(val1) in str(val2):
                            for n1, n2 in itertools.product(nodes1, nodes2):
                                collector.add_edge(n1, n2, GraphEdgeType.SUBSTR)
                                collector.add_edge(n2, n1, GraphEdgeType.SUPSTR)

                        elif str(val2) in str(val1):
                            for n1, n2 in itertools.product(nodes1, nodes2):
                                collector.add_edge(n2, n1, GraphEdgeType.SUBSTR)
                                collector.add_edge(n1, n2, GraphEdgeType.SUPSTR)


class DataFrameNodeCollection(GraphNodeCollection):
    def __init__(self, obj: pd.DataFrame, type_str: str, source: str, options: GraphOptions, **kwargs):
        self.index_nodes: List[List[GraphNode]] = []  # Shape : num_rows x num_levels (highest level first)
        self.column_nodes: List[List[GraphNode]] = []  # Shape : num_cols x num_levels (highest level first)
        self.cell_nodes: List[List[GraphNode]] = []  # Row-major
        self.index_name_nodes: List[GraphNode] = []  # Shape : num_levels (highest level first)
        self.column_name_nodes: List[GraphNode] = []  # Shape : num_levels (highest level first)

        self.index_name_nodes: List[GraphNode] = []
        self.col_name_nodes: List[GraphNode] = []

        super().__init__(obj, type_str, source, options, **kwargs)

    def get_index_node(self, val, level: int, idx: int, num_levels: int):
        identifier = '[{},{}]'.format(idx, level - num_levels)
        node = self.get_node(val, identifier=identifier)
        node.ntype = GraphNodeType.INDEX
        return node

    def get_column_node(self, val, level: int, idx: int, num_levels: int):
        identifier = '[{},{}]'.format(level - num_levels, idx)
        node = self.get_node(val, identifier=identifier)
        node.ntype = GraphNodeType.COLUMN
        return node

    def get_index_name_node(self, val, level: int, num_levels: int):
        identifier = '[{},{}]'.format(-1, level - num_levels)
        node = self.get_node(val, identifier=identifier)
        node.ntype = GraphNodeType.INDEX_NAME
        return node

    def get_column_name_node(self, val, level: int, num_levels: int):
        identifier = '[{},{}]'.format(level - num_levels, -1)
        node = self.get_node(val, identifier=identifier)
        node.ntype = GraphNodeType.COL_INDEX_NAME
        return node

    def add_index_nodes(self, index: pd.Index, mode='df.index'):
        if isinstance(index, pd.MultiIndex):
            index_nodes = []

            for idx, vals in enumerate(index):
                index_nodes.append([])
                for level, val in enumerate(vals):
                    if mode == 'df.index':
                        node = self.get_index_node(val, level=level, idx=idx, num_levels=index.nlevels)
                    else:
                        node = self.get_column_node(val, level=level, idx=idx, num_levels=index.nlevels)

                    self.add_node(node)
                    self.add_value(val, node)
                    index_nodes[-1].append(node)

            return index_nodes

        else:
            index_nodes = []
            for idx, val in enumerate(index):
                if mode == 'df.index':
                    node = self.get_index_node(val, level=0, idx=idx, num_levels=1)
                else:
                    node = self.get_column_node(val, level=0, idx=idx, num_levels=1)

                self.add_node(node)
                self.add_value(val, node)
                index_nodes.append(node)

            return [index_nodes]

    def add_nodes(self):
        cells = self.obj.values
        for r_idx, row in enumerate(cells):
            self.cell_nodes.append([])
            for c_idx, val in enumerate(row):
                node = self.get_node(val, identifier='[{},{}]'.format(r_idx, c_idx))
                self.add_node(node)
                self.add_value(val, node)
                self.cell_nodes[-1].append(node)

        if self.options.INDEX_NODES:
            self.index_nodes = self.add_index_nodes(self.obj.index, mode='df.index')

        if self.options.COLUMN_NODES:
            self.column_nodes = self.add_index_nodes(self.obj.columns, mode='df.columns')

        if self.options.INDEX_NAME_NODES:
            num_index_levels = len(self.obj.index.names)
            num_column_levels = len(self.obj.columns.names)
            for level, name in enumerate(self.obj.index.names):
                if name is None:
                    continue

                node = self.get_index_name_node(name, level=level, num_levels=num_index_levels)
                self.add_node(node)
                self.add_value(name, node)
                self.index_name_nodes.append(node)

            for level, name in enumerate(self.obj.columns.names):
                if name is None:
                    continue

                node = self.get_column_name_node(name, level=level, num_levels=num_column_levels)
                self.add_node(node)
                self.add_value(name, node)
                self.column_name_nodes.append(node)

    def add_internal_edges(self, collector: EdgeCollection):
        super().add_internal_edges(collector)
        if self.options.INDEX_EDGES:
            for index_nodes, cell_nodes in zip(self.index_nodes, self.cell_nodes):
                for n1, n2 in itertools.product(index_nodes, cell_nodes):
                    collector.add_edge(n1, n2, GraphEdgeType.INDEX)
                    collector.add_edge(n2, n1, GraphEdgeType.INDEXED_FOR)

            for col_nodes, cell_nodes in zip(self.column_nodes, np.transpose(self.cell_nodes)):
                for n1, n2 in itertools.product(col_nodes, cell_nodes):
                    collector.add_edge(n1, n2, GraphEdgeType.INDEX)
                    collector.add_edge(n2, n1, GraphEdgeType.INDEXED_FOR)

        if self.options.INDEX_NAME_EDGES:
            for index_name_node, index_nodes in zip(self.index_name_nodes, np.transpose(self.index_nodes)):
                for n in index_nodes:
                    collector.add_edge(index_name_node, n, GraphEdgeType.INDEX_NAME)
                    collector.add_edge(n, index_name_node, GraphEdgeType.INDEX_NAME_FOR)

            for col_name_node, col_nodes in zip(self.col_name_nodes, np.transpose(self.column_nodes)):
                for n in col_nodes:
                    collector.add_edge(col_name_node, n, GraphEdgeType.INDEX_NAME)
                    collector.add_edge(n, col_name_node, GraphEdgeType.INDEX_NAME_FOR)

        if self.options.ADJACENCY_EDGES:
            def adjacency_to_the_right(vals):
                for row_vals in vals:
                    for n1, n2 in zip(row_vals, row_vals[1:]):
                        collector.add_edge(n1, n2, GraphEdgeType.ADJ_RIGHT)
                        collector.add_edge(n2, n1, GraphEdgeType.ADJ_LEFT)

            def adjacency_below(vals):
                for col_vals in np.transpose(vals):
                    for n1, n2 in zip(col_vals, col_vals[1:]):
                        collector.add_edge(n1, n2, GraphEdgeType.ADJ_BELOW)
                        collector.add_edge(n2, n1, GraphEdgeType.ADJ_ABOVE)

            for node_set in [self.index_nodes, np.transpose(self.column_nodes), self.cell_nodes]:
                adjacency_to_the_right(node_set)
                adjacency_below(node_set)


class SizedIterableNodeCollection(DataFrameNodeCollection):
    """
    Approximated as a dataframe without index/column nodes and only adjacency edges
    Note that this preserves the 2-D structure if any, no harm in keeping this information alive
    """

    def __init__(self, obj: pd.DataFrame, type_str: str, source: str, options: GraphOptions, **kwargs):
        options = copy.copy(options)
        options.INDEX_NODES = False
        options.COLUMN_NODES = False
        options.INDEX_NAME_NODES = False
        options.INDEX_EDGES = False
        options.INDEX_NAME_EDGES = False
        super().__init__(obj, type_str, source, options, **kwargs)


class SingleNodeCollection(GraphNodeCollection):
    def __init__(self, obj: Any, type_str: str, source: str, options: GraphOptions, **kwargs):
        self.node = None
        super().__init__(obj, type_str, source, options, **kwargs)

    def add_node(self, node: GraphNode):
        super().add_node(node)
        self.node = node

    def setup_representor(self, collector: EdgeCollection):
        return self.node

    def add_custom_edge(self, other: 'SingleNodeCollection', edge_type: GraphEdgeType, collector: EdgeCollection):
        collector.add_edge(node1=self.node, node2=other.node, etype=edge_type)


class ScalarNodeCollection(SingleNodeCollection):
    pass
