import itertools
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Tuple

from autopandas_v2.generators.dsl.values import Value, Default
from autopandas_v2.ml.featurization.featurizer import RelationGraph
from autopandas_v2.ml.featurization.graph import GraphNode, GraphEdgeType, GraphNodeType
from autopandas_v2.ml.featurization.node_collections import SingleNodeCollection, GraphNodeCollection
from autopandas_v2.ml.featurization.options import GraphOptions
from autopandas_v2.utils.exceptions import AutoPandasException


class GraphCacheIOExample:
    class Key:
        def __init__(self, inputs: List[Any], output: Any):
            #  The cache is only concerned with object references
            #  Therefore a input-output example in the cache is matched only if it uses EXACTLY the same objects
            #  as the query This makes sense because a generator uses the same I/O spec in enumeration
            self.input_ids = [id(inp) for inp in inputs]
            self.output_id = id(output)

        def __hash__(self):
            return 31 * sum(hash(inp) for inp in self.input_ids) + hash(self.output_id)

        def __eq__(self, other):
            if not isinstance(other, GraphCacheIOExample.Key):
                return False

            return self.input_ids == other.input_ids and self.output_id == other.output_id

        def __ne__(self, other):
            return not (self.__eq__(other))

    cache: Dict[Key, RelationGraph] = {}

    @classmethod
    def get_key(cls, inputs: List[Any], output: Any) -> Key:
        return GraphCacheIOExample.Key(inputs, output)

    @classmethod
    def is_present(cls, inputs: List[Any], output: Any) -> Tuple[bool, Key]:
        key = GraphCacheIOExample.Key(inputs, output)
        return key in cls.cache, key

    @classmethod
    def get_graph(cls, key: Key) -> RelationGraph:
        return cls.cache[key]

    @classmethod
    def add_graph(cls, key: Key, graph: RelationGraph):
        cls.cache[key] = graph


class BaseRelationGraphOp(RelationGraph, ABC):
    @classmethod
    @abstractmethod
    def get_options(cls) -> GraphOptions:
        pass

    @classmethod
    def init(cls, inputs: List[Any], output: Any):
        is_cached, key = GraphCacheIOExample.is_present(inputs, output)
        if is_cached:
            return cls.from_parent_graph(GraphCacheIOExample.get_graph(key))

        else:
            graph = RelationGraph(cls.get_options())
            graph.from_input_output(inputs, output)
            GraphCacheIOExample.add_graph(key, graph)
            return cls.init(inputs, output)

    def __init__(self, options: GraphOptions):
        super().__init__(options)

    def convert_obj_to_collections(self, obj: Any, source: str,
                                   options: GraphOptions = None):
        try:
            return super().convert_obj_to_collections(obj, source, options)
        except NotImplementedError:
            pass

        if isinstance(obj, Value):
            if isinstance(obj, Default):
                result = self.convert_obj_to_collections(obj.val, source, options)
                #  TODO : Fix this
                for c in result:
                    for n in c.nodes:
                        n.ntype = GraphNodeType.DEFAULT

                return result

            return self.convert_obj_to_collections(obj.val, source, options)


class RelationGraphChoice(BaseRelationGraphOp):
    class GraphOptionsChoice(GraphOptions):
        pass

    @classmethod
    def get_options(cls):
        options = RelationGraphChoice.GraphOptionsChoice()
        options.INDEX_NODES = True
        options.COLUMN_NODES = True
        options.INDEX_NAME_NODES = True
        options.INDEX_EDGES = True
        options.EQUALITY_EDGES = True
        options.ADJACENCY_EDGES = True
        options.NODE_TYPES = True
        options.EDGE_TYPES = True
        options.SUBSTR_EDGES = True

        return options

    def __init__(self, options: GraphOptions):
        super().__init__(options)
        self.num_choices: int = None
        self.chosen: int = None

    def add_choices(self, num_choices: int, chosen: int = None, query: bool = False):
        if (not query) and (chosen is None):
            raise AutoPandasException("One of query and chosen needs to be supplied to Choice")

        self.num_choices = num_choices
        self.chosen = chosen

    def get_encoding(self, get_mapping=False):
        encoding, node_to_int = super().get_encoding(get_mapping=True)
        encoding['operator'] = 'Choice'
        encoding['num_choices'] = self.num_choices
        if self.chosen is not None:
            encoding['chosen'] = self.chosen

        if get_mapping:
            return encoding, node_to_int

        return encoding


class RelationGraphChain(BaseRelationGraphOp):
    class GraphOptionsChain(GraphOptions):
        pass

    @classmethod
    def get_options(cls):
        options = RelationGraphChain.GraphOptionsChain()
        options.INDEX_NODES = True
        options.COLUMN_NODES = True
        options.INDEX_NAME_NODES = True
        options.INDEX_EDGES = True
        options.EQUALITY_EDGES = True
        options.ADJACENCY_EDGES = True
        options.NODE_TYPES = True
        options.EDGE_TYPES = True
        options.SUBSTR_EDGES = True

        return options

    def __init__(self, options: GraphOptions):
        super().__init__(options)
        self.num_options: int = None
        self.picked: int = None

    def add_options(self, num_options: int, picked: int = None, query: bool = False):
        if (not query) and (picked is None):
            raise AutoPandasException("One of query and picked needs to be supplied to Chain")

        self.num_options = num_options
        self.picked = picked

    def get_encoding(self, get_mapping=False):
        encoding, node_to_int = super().get_encoding(get_mapping=True)
        encoding['operator'] = 'Chain'
        encoding['num_options'] = self.num_options
        if self.picked is not None:
            encoding['picked'] = self.picked

        if get_mapping:
            return encoding, node_to_int

        return encoding


class RelationGraphSelect(BaseRelationGraphOp):
    class GraphOptionsSelect(GraphOptions):
        pass

    @classmethod
    def get_options(cls):
        options = RelationGraphSelect.GraphOptionsSelect()
        options.INDEX_NODES = True
        options.COLUMN_NODES = True
        options.INDEX_NAME_NODES = True
        options.INDEX_EDGES = True
        options.EQUALITY_EDGES = True
        options.ADJACENCY_EDGES = True
        options.NODE_TYPES = True
        options.SUBSTR_EDGES = True
        options.EDGE_TYPES = True

        return options

    def __init__(self, options: GraphOptions):
        super().__init__(options)
        self.domain_nodes: List[GraphNode] = []
        self.selected_node: GraphNode = None

    def get_domain_node_collection(self, val: Any, idx: int):
        return self.convert_obj_to_collections(val, source='E' + str(idx))[0]

    def add_domain(self, domain: List[Any], selected_idx: int = None, query: bool = False):
        if (not query) and (selected_idx is None):
            raise AutoPandasException("One of query and selected_idx needs to be supplied to Select")

        domain_node_collections: List[GraphNodeCollection] = [self.get_domain_node_collection(v, idx)
                                                              for idx, v in enumerate(domain)]

        for c in domain_node_collections:
            self.domain_nodes.append(c.setup_representor(self.edge_collection))

        #  Add the adjacency edges amongst the elements
        #  UPDATE : It's probably more appropriate to NOT have adjacency edges
        #  There is no implicit order in the set of values passed to Select at any point,
        #  it's just a pool of candidates to choose from
        # for c1, c2 in zip(domain_node_collections, domain_node_collections[1:]):
        #     c1.add_custom_edge(c2, GraphEdgeType.ADJACENCY, self.edge_collection)

        #  Add any internal edges
        for c in domain_node_collections:
            c.add_internal_edges(self.edge_collection)

        #  Add edges from these to the existing nodes
        for c1, c2 in itertools.product(domain_node_collections, self.node_collections):
            c1.add_external_edges(c2, collector=self.edge_collection)

        for c in domain_node_collections:
            self.nodes.update(c.nodes)

        if selected_idx is not None:
            self.selected_node = self.domain_nodes[selected_idx]

    def get_encoding(self, get_mapping=False, get_reverse_mapping=False):
        encoding, node_to_int = super().get_encoding(get_mapping=True)
        encoding['operator'] = 'Select'
        encoding['candidates'] = [node_to_int[x] for x in self.domain_nodes]
        reverse_mapping = {node_to_int[x]: idx for idx, x in enumerate(self.domain_nodes)}

        if self.selected_node is not None:
            encoding['selected'] = node_to_int[self.selected_node]

        if get_mapping:
            if get_reverse_mapping:
                return encoding, reverse_mapping, node_to_int

            return encoding, node_to_int

        if get_reverse_mapping:
            return encoding, reverse_mapping

        return encoding


class RelationGraphProduct(BaseRelationGraphOp):
    class GraphOptionsProduct(GraphOptions):
        pass

    @classmethod
    def get_options(cls):
        options = RelationGraphProduct.GraphOptionsProduct()
        options.INDEX_NODES = True
        options.COLUMN_NODES = True
        options.INDEX_NAME_NODES = True
        options.INDEX_EDGES = True
        options.EQUALITY_EDGES = True
        options.ADJACENCY_EDGES = True
        options.NODE_TYPES = True
        options.EDGE_TYPES = True
        options.SUBSTR_EDGES = True

        return options

    def __init__(self, options: GraphOptions):
        super().__init__(options)
        self.iterables: List[List[GraphNode]] = []
        self.selected_nodes: List[GraphNode] = None

    def get_iterable_node_collection(self, val: Any, idx: int):
        return self.convert_obj_to_collections(val, source='E' + str(idx))[0]

    def add_iterables(self, iterables: List[List[Any]], selected_indices: List[int] = None, query: bool = False):
        if (not query) and (selected_indices is None):
            raise AutoPandasException("One of query and selected_indices needs to be supplied to Product")

        iterable_node_collections: List[List[GraphNodeCollection]] = []
        for iter_num, iterable in enumerate(iterables):
            iterable_node_collections.append([self.get_iterable_node_collection(v, idx + iter_num * len(iterables))
                                              for idx, v in enumerate(iterable)])
            self.iterables.append([])
            for c in iterable_node_collections[-1]:
                self.iterables[-1].append(c.setup_representor(self.edge_collection))
                c.add_internal_edges(self.edge_collection)

        for iterable_collections in iterable_node_collections:
            for c1, c2 in itertools.product(iterable_collections, self.node_collections):
                c1.add_external_edges(c2, collector=self.edge_collection)

        for iterable_collections in iterable_node_collections:
            for c in iterable_collections:
                self.nodes.update(c.nodes)

        if selected_indices is not None:
            self.selected_nodes = [self.iterables[iter_num][idx]
                                   for iter_num, idx in enumerate(selected_indices)]

    def get_encoding(self, get_mapping=False, get_reverse_mapping=False):
        encoding, node_to_int = super().get_encoding(get_mapping=True)
        encoding['operator'] = 'Product'
        encoding['iterables'] = [[node_to_int[x] for x in iterable] for iterable in self.iterables]
        reverse_mapping = {node_to_int[x]: idx for iterable in self.iterables
                           for idx, x in enumerate(iterable)}

        if self.selected_nodes is not None:
            encoding['selected'] = [node_to_int[x] for x in self.selected_nodes]

        if get_mapping:
            if get_reverse_mapping:
                return encoding, reverse_mapping, node_to_int

            return encoding, node_to_int

        if get_reverse_mapping:
            return encoding, reverse_mapping

        return encoding


class RelationGraphSubsets(BaseRelationGraphOp):
    class GraphOptionsSubsets(GraphOptions):
        pass

    @classmethod
    def get_options(cls):
        options = RelationGraphSubsets.GraphOptionsSubsets()
        options.INDEX_NODES = True
        options.COLUMN_NODES = True
        options.INDEX_NAME_NODES = True
        options.INDEX_EDGES = True
        options.EQUALITY_EDGES = True
        options.ADJACENCY_EDGES = True
        options.NODE_TYPES = True
        options.EDGE_TYPES = True
        options.SUBSTR_EDGES = True

        return options

    def __init__(self, options: GraphOptions):
        super().__init__(options)
        self.elements: List[GraphNode] = []
        self.selected_nodes: List[GraphNode] = None

    def get_element_node_collection(self, val: Any, idx: int):
        return self.convert_obj_to_collections(val, source='E' + str(idx))[0]

    def add_set(self, vals: List[Any], selected_indices: List[int] = None, query: bool = False):
        if (not query) and (selected_indices is None):
            raise AutoPandasException("One of query and selected_indices needs to be supplied to Subsets")

        element_node_collections: List[GraphNodeCollection] = [self.get_element_node_collection(v, idx)
                                                               for idx, v in enumerate(vals)]

        for c in element_node_collections:
            self.elements.append(c.setup_representor(self.edge_collection))

        #  Add any internal edges
        for c in element_node_collections:
            c.add_internal_edges(self.edge_collection)

        #  Add edges from these to the existing nodes
        for c1, c2 in itertools.product(element_node_collections, self.node_collections):
            c1.add_external_edges(c2, collector=self.edge_collection)

        for c in element_node_collections:
            self.nodes.update(c.nodes)

        if selected_indices is not None:
            self.selected_nodes = [self.elements[idx] for idx in selected_indices]

    def get_encoding(self, get_mapping=False, get_reverse_mapping=False):
        encoding, node_to_int = super().get_encoding(get_mapping=True)
        encoding['operator'] = 'Subsets'
        encoding['elements'] = [node_to_int[x] for x in self.elements]
        reverse_mapping = {node_to_int[x]: idx for idx, x in enumerate(self.elements)}
        if self.selected_nodes is not None:
            encoding['selected'] = [node_to_int[x] for x in self.selected_nodes]

        if get_mapping:
            if get_reverse_mapping:
                return encoding, reverse_mapping, node_to_int

            return encoding, node_to_int

        if get_reverse_mapping:
            return encoding, reverse_mapping

        return encoding


class RelationGraphOrderedSubsets(BaseRelationGraphOp):
    class GraphOptionsOrderedSubsets(GraphOptions):
        pass

    @classmethod
    def get_options(cls):
        options = RelationGraphOrderedSubsets.GraphOptionsOrderedSubsets()
        options.INDEX_NODES = True
        options.COLUMN_NODES = True
        options.INDEX_NAME_NODES = True
        options.INDEX_EDGES = True
        options.EQUALITY_EDGES = True
        options.ADJACENCY_EDGES = True
        options.NODE_TYPES = True
        options.EDGE_TYPES = True
        options.SUBSTR_EDGES = True

        return options

    def __init__(self, options: GraphOptions):
        super().__init__(options)
        self.elements: List[GraphNode] = []
        self.selected_nodes: List[GraphNode] = None

    def get_element_node_collection(self, val: Any, idx: int):
        return self.convert_obj_to_collections(val, source='E' + str(idx))[0]

    def add_set(self, vals: List[Any], selected_indices: List[int] = None, query: bool = False):
        if (not query) and (selected_indices is None):
            raise AutoPandasException("One of query and selected_indices needs to be supplied to OrderedSubsets")

        element_node_collections: List[GraphNodeCollection] = [self.get_element_node_collection(v, idx)
                                                               for idx, v in enumerate(vals)]

        for c in element_node_collections:
            self.elements.append(c.setup_representor(self.edge_collection))

        #  Add any internal edges
        for c in element_node_collections:
            c.add_internal_edges(self.edge_collection)

        #  Add edges from these to the existing nodes
        for c1, c2 in itertools.product(element_node_collections, self.node_collections):
            c1.add_external_edges(c2, collector=self.edge_collection)

        for c in element_node_collections:
            self.nodes.update(c.nodes)

        if selected_indices is not None:
            self.selected_nodes = [self.elements[idx] for idx in selected_indices]

    def get_encoding(self, get_mapping=False, get_reverse_mapping=False):
        encoding, node_to_int = super().get_encoding(get_mapping=True)
        encoding['operator'] = 'OrderedSubsets'
        encoding['elements'] = [node_to_int[x] for x in self.elements]
        reverse_mapping = {node_to_int[x]: idx for idx, x in enumerate(self.elements)}
        if self.selected_nodes is not None:
            encoding['selected'] = [node_to_int[x] for x in self.selected_nodes]

        #  Create a terminal token and add it
        terminal_node = GraphNode(source='T', identifier='', ntype=GraphNodeType.TERMINAL)
        node_to_int[terminal_node] = max(node_to_int.values()) + 1
        encoding['elements'].append(node_to_int[terminal_node])
        encoding['terminal'] = node_to_int[terminal_node]
        encoding['node_features'].append(terminal_node.get_encoding())
        reverse_mapping[node_to_int[terminal_node]] = len(self.elements)

        if self.selected_nodes is not None:
            encoding['selected'].append(node_to_int[terminal_node])

        if get_mapping:
            if get_reverse_mapping:
                return encoding, reverse_mapping, node_to_int

            return encoding, node_to_int

        if get_reverse_mapping:
            return encoding, reverse_mapping

        return encoding
