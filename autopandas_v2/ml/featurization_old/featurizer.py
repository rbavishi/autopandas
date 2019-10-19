import itertools

import pandas as pd
import numpy as np
from typing import List, Any, Set, Dict, Tuple, Union, Callable

from autopandas_v2.ml.featurization_old.graph import GraphNode, GraphEdgeType
from autopandas_v2.ml.featurization_old.node_collections import GraphNodeCollection, \
    DataFrameNodeCollection, SizedIterableNodeCollection, ScalarNodeCollection, SingleNodeCollection
from autopandas_v2.ml.featurization_old.edge_collections import EdgeCollection
from autopandas_v2.ml.featurization_old.options import GraphOptions
from autopandas_v2.utils.exceptions import SilentException
from autopandas_v2.utils.types import Lambda


class RelationGraph:
    def __init__(self, options: GraphOptions):
        self.node_collections: List[GraphNodeCollection] = []
        self.edge_collection: EdgeCollection = EdgeCollection()
        self.nodes: Set[GraphNode] = set()
        self.edges = self.edge_collection.edges
        self.options = options

    def convert_obj_to_collections(self, obj: Any, source: str,
                                   options: GraphOptions = None) -> List[GraphNodeCollection]:

        if options is None:
            options = self.options

        def convert(val, collection, type_str: str, source: str = source, options: GraphOptions = options):
            return collection(val, type_str=type_str, source=source, options=options)

        if isinstance(obj, pd.DataFrame):
            return [convert(obj, DataFrameNodeCollection, type_str='dataframe')]

        elif isinstance(obj, pd.Series):
            return [convert(pd.DataFrame(obj), DataFrameNodeCollection, type_str='series')]

        elif isinstance(obj, np.ndarray):
            if all(map(np.isscalar, obj)):
                return [convert(pd.DataFrame(obj), SizedIterableNodeCollection, type_str='ndarray')]

            return sum(
                map(lambda x: self.convert_obj_to_collections(x[1], source=source + '_' + str(x[0]), options=options),
                    enumerate(obj)), [])

        elif isinstance(obj, dict):
            try:
                return [convert(pd.DataFrame(obj), SizedIterableNodeCollection, type_str='dict')]
            except ValueError:
                return [convert(pd.DataFrame(obj, index=[0]), SizedIterableNodeCollection, type_str='dict')]

        elif isinstance(obj, pd.MultiIndex):
            #  Not quite sure how to handle this properly
            return [convert(pd.DataFrame(list(map(list, obj))), DataFrameNodeCollection, type_str='multiindex')]

        elif isinstance(obj, pd.Index):
            return [convert(pd.DataFrame(obj), SizedIterableNodeCollection, type_str='index')]

        elif np.isscalar(obj) or obj is None:
            return [convert(obj, ScalarNodeCollection, type_str='scalar')]

        elif isinstance(obj, Lambda):
            return [convert(obj.fn, SingleNodeCollection, type_str='lambda')]

        elif isinstance(obj, Callable):
            try:
                if obj.__name__ == "<lambda>":
                    return [convert(obj, SingleNodeCollection, type_str='lambda')]
            except:
                pass

        elif isinstance(obj, list):
            if all(map(np.isscalar, obj)):
                return [convert(pd.DataFrame(obj), SizedIterableNodeCollection, type_str='list')]

            return sum(
                map(lambda x: self.convert_obj_to_collections(x[1], source=source + '_' + str(x[0]), options=options),
                    enumerate(obj)), [])

        elif isinstance(obj, tuple):
            if all(map(np.isscalar, obj)):
                return [convert(pd.DataFrame(list(obj)), SizedIterableNodeCollection, type_str='tuple')]

            return sum(
                map(lambda x: self.convert_obj_to_collections(x[1], source=source + '_' + str(x[0]), options=options),
                    enumerate(obj)), [])

        elif isinstance(obj, pd.core.groupby.GroupBy):
            #  Pandas cracks if there are nans in the group keys, so trying our best to circumvent that
            def custom_index(key):
                if key in obj.indices:
                    return obj.indices[key]

                if isinstance(key, tuple) and any(pd.isnull(x) for x in key):
                    for k, v in obj.indices.items():
                        if (not isinstance(k, tuple)) or len(k) != len(key):
                            continue

                        for i, j in zip(k, key):
                            if pd.isnull(j):
                                continue

                            if i != j:
                                break
                        else:
                            return v

                    if key in obj.groups:
                        index = np.array(obj.groups[key])
                        for i in obj.indices.values():
                            if np.array_equal(i, index):
                                return i

                #  Force the error
                obj.get_group(key)
                return obj._get_index(key)

            try:
                return [convert(pd.DataFrame(obj._selected_obj._take(custom_index(group), axis=obj.axis)),
                                DataFrameNodeCollection, type_str='dataframe', source=source + '_' + str(i))
                        for i, group in enumerate(obj.groups)]

            except (IndexError, KeyError):
                raise SilentException("Same ol' groupby error")

        raise NotImplementedError("Cannot convert object into a node-collection : {}".format(obj))

    def add_obj(self, obj: Any, source: str, options: GraphOptions = None):
        """
        Convert obj into a GraphNodeCollection and add it to the graph (along with any edges resulting from it).
        Source captures the role the object plays. For example, I0 denotes that it's the first input in the
        IO Spec that is provided to AutoPandas. Similarly O0 implies an output
        """

        new_collections: List[GraphNodeCollection] = self.convert_obj_to_collections(obj, source, options=options)

        #  Add edges within the collection
        for c in new_collections:
            c.add_internal_edges(collector=self.edge_collection)

        #  Add edges amongst themselves
        for c1, c2 in itertools.combinations(new_collections, 2):
            c1.add_external_edges(c2, collector=self.edge_collection)
            c2.add_external_edges(c1, is_reverse=True, collector=self.edge_collection)

        #  Add edges between the new collections and the old collections
        for c1, c2 in itertools.product(self.node_collections, new_collections):
            c1.add_external_edges(c2, collector=self.edge_collection)
            c2.add_external_edges(c1, is_reverse=True, collector=self.edge_collection)

        for c in new_collections:
            self.nodes.update(c.nodes)

        #  Assimilate the new collections
        self.node_collections.extend(new_collections)

    def from_input_output(self, inputs: List[Any], output: Any):
        for idx, inp in enumerate(inputs):
            self.add_obj(inp, source='I{}'.format(idx))

        self.add_obj(output, source='O0')

    @classmethod
    def from_parent_graph(cls, graph: 'RelationGraph') -> 'RelationGraph':
        result = cls(graph.options)
        result.node_collections = graph.node_collections[:]
        result.edge_collection.edges = graph.edge_collection.edges.copy()
        result.nodes = set(graph.nodes)
        result.edges = result.edge_collection.edges
        return result

    def get_encoding(self, get_mapping=False) -> Union[Dict, Tuple[Dict, Dict]]:
        """
        The format of the dict is as follows
        {
            'edges': [
                List[src, edge-type, dst]
            ]

            'node_features': [
                int
            ]
        }

        There should be no more than 1 edge of the same type between two nodes
        Also, each node is assumed to have a single type associated with it
        """

        node_to_int: Dict[GraphNode, int] = {n: idx for idx, n in enumerate(sorted(self.nodes, key=str))}
        edges: List[List] = []
        for edge in self.edges:
            src, dst = map(node_to_int.get, (edge.node1, edge.node2))
            edges.append([src, edge.get_encoding(), dst])

        node_features = [n.get_encoding() for n in node_to_int]
        result = {
            'edges': edges,
            'node_features': node_features
        }

        if get_mapping:
            return result, node_to_int

        return result
