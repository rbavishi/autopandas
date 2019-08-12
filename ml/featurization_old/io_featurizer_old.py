import json
import logging
from enum import Enum
import pandas as pd
import numpy as np
from typing import Any, List, Dict, Tuple, Set, Callable, Union
from collections import defaultdict

from autopandas_v2.generators.dsl.values import Value, Default
from autopandas_v2.utils import logger
from autopandas_v2.utils.cli import ArgNamespace
from autopandas_v2.utils.types import Lambda


class RelationGraphNodeType(Enum):
    COLUMN = 0
    INDEX = 1
    OBJECT = 2
    FLOAT = 3
    INT = 4
    STR = 5
    DATE = 6
    DELTA = 7
    BOOL = 8
    NAN = 9
    NONE = 10
    COL_INDEX_NAME = 11
    INDEX_NAME = 12
    DEFAULT = 13

    @classmethod
    def get_node_type(cls, value):
        # Gets the `RelationGraphNodeType` for the given value. Does not return column name or index
        # type --- those should be encoded separately.
        val_type = type(value)
        if np.issubdtype(val_type, np.floating):
            if pd.isnull(value):
                val_type = cls.NAN
            else:
                val_type = cls.FLOAT
        elif np.issubdtype(val_type, np.signedinteger) or np.issubdtype(val_type, np.unsignedinteger):
            val_type = cls.INT
        elif np.issubdtype(val_type, np.str_):
            val_type = cls.STR
        elif np.issubdtype(val_type, np.bool_):
            val_type = cls.BOOL
        elif isinstance(value, pd.datetime):
            val_type = cls.DATE
        elif isinstance(value, pd.Timedelta):
            val_type = cls.DELTA
        elif value is None:
            val_type = cls.NONE
        elif isinstance(value, Value):
            if isinstance(value, Default):
                val_type = cls.DEFAULT
            else:
                return cls.get_node_type(value.val)
        else:
            val_type = cls.OBJECT
        return val_type


class DfTypeWrapper(object):
    def __init__(self, df, true_type=None):
        self.df = df
        self.true_type = true_type

    def is_df(self):
        # A series is functionally equivalent since it can have a name and indices.
        return self.true_type in [None, 'series']


class RelationGraphNode:
    def __init__(self, dfindex: str, pos: (int, int), type_: RelationGraphNodeType):
        self.dfindex = dfindex
        self.pos = pos
        #  TODO : Support multiple node types for generality
        self._type = type_

    def __hash__(self):
        res = 17
        res = 31 * res + hash(self.dfindex)
        res = 31 * res + hash(self.pos)
        res = 31 * res + hash(self._type)
        return res

    def __eq__(self, other):
        if not (isinstance(other, RelationGraphNode)):
            return False
        else:
            return (self.dfindex == other.dfindex) and (self.pos == other.pos) \
                   and (self._type == other._type)

    def __str__(self):
        return "%s[%d,%d](%s)" % (self.dfindex, self.pos[0], self.pos[1], self._type)

    def __repr__(self):
        return self.__str__()

    def is_extra(self):
        return "E" in self.dfindex

    def get_feature_encoding(self) -> int:
        is_output = "O" in self.dfindex
        if self._type is not None:
            type_encoding: int = self._type.value
            return type_encoding + is_output * len(RelationGraphNodeType)
        else:
            return is_output * 1


class RelationGraphEdgeType(Enum):
    ADJACENCY = 0
    EQUALITY = 1
    INDEX = 2
    GENERIC = 4
    INFLUENCE = 3


class RelationGraphEdge:
    def __init__(self, node1: RelationGraphNode, node2: RelationGraphNode, type_: RelationGraphEdgeType):
        self.node1 = node1
        self.node2 = node2
        self._type = type_

    def __hash__(self):
        res = 17
        res = 31 * res + hash(self.node1) + hash(self.node2)
        res = 31 * res + hash(self._type)
        return res

    def __eq__(self, other):
        if not (isinstance(other, RelationGraphEdge)):
            return False
        else:
            return (self.node1 in other.get_nodes()) and (self.node2 in other.get_nodes()) \
                   and self._type == other.get_type()

    def __str__(self):
        return "(%s, %s)[%s]" % (self.node1, self.node2, self._type)

    def __repr__(self):
        return self.__str__()

    def get_nodes(self):
        return [self.node1, self.node2]

    def get_type(self):
        return self._type

    def get_feature_encoding(self) -> int:
        if self._type == RelationGraphEdgeType.GENERIC:
            #  Only possible if edge features are turned off
            #  TODO : This is ugly nevertheless, refactor
            return 0

        type_encoding: int = self._type.value
        return type_encoding


class RelationGraphOptions:
    NODE_TYPES = True
    EDGE_TYPES = True
    INDEX_NODES = True
    COLUMN_NODES = True
    INDEX_NAME_NODES = False
    ADJACENCY_EDGES = True
    EQUALITY_EDGES = True
    INDEX_EDGES = True
    INFLUENCE_EDGES = False

    @staticmethod
    def from_args(args: ArgNamespace):
        res = RelationGraphOptions()
        feats = args.get('use_features', [])
        no_feats = args.get('ignore_features', [])
        if feats:
            allowed = set()
            for k, v in vars(RelationGraphOptions).items():
                if not k.isupper():
                    continue

                setattr(res, k, False)
                allowed.add(k)

            for k in feats:
                if k in allowed:
                    setattr(res, k, True)

            return res

        elif no_feats:
            allowed = set()
            for k, v in vars(RelationGraphOptions).items():
                if not k.isupper():
                    continue

                setattr(res, k, True)
                allowed.add(k)

            for k in no_feats:
                if k in allowed:
                    setattr(res, k, False)

            return res

        else:
            return res


class RelationGraph:
    def __init__(self, options=RelationGraphOptions()):
        self.nodes: Set[RelationGraphNode] = set()
        self.edges: Set[RelationGraphEdge] = set()
        self.options = options
        self.input_dfs = []
        self.output_dfs = []
        self.version = "1.0.0"

    def add_node(self, node: RelationGraphNode):
        self.nodes.add(node)

    def remove_node(self, node: RelationGraphNode):
        self.nodes.remove(node)
        for edge in list(self.edges):
            endpoints = edge.get_nodes()
            if node in endpoints:
                self.edges.remove(edge)

    def add_edge(self, node1: RelationGraphNode, node2: RelationGraphNode, type_: RelationGraphEdgeType):
        if node1 not in self.nodes:
            self.add_node(node1)

        if node2 not in self.nodes:
            self.add_node(node2)

        #  Filtering things out here because we don't want to miss out on any nodes
        #  Ideally there should be a separate method that adds all desired nodes, but the code
        #  is not structured that way currently. So we add edges assuming everything is enabled
        #  and disable everything at one place - here.

        if type_ == RelationGraphEdgeType.INDEX and not self.options.INDEX_EDGES:
            return
        if type_ == RelationGraphEdgeType.ADJACENCY and not self.options.ADJACENCY_EDGES:
            return
        if type_ == RelationGraphEdgeType.EQUALITY and not self.options.EQUALITY_EDGES:
            return

        if self.options.EDGE_TYPES:
            self.edges.add(RelationGraphEdge(node1, node2, type_))
        else:
            self.edges.add(RelationGraphEdge(node1, node2, RelationGraphEdgeType.GENERIC))

    def get_node_type(self, value=None, is_column=False, is_index=False, is_col_index_name=False, is_index_name=False):
        if self.options.NODE_TYPES:
            if is_column:
                return RelationGraphNodeType.COLUMN
            if is_index:
                return RelationGraphNodeType.INDEX
            if is_col_index_name:
                return RelationGraphNodeType.COL_INDEX_NAME
            if is_index_name:
                return RelationGraphNodeType.INDEX_NAME

            return RelationGraphNodeType.get_node_type(value)
        else:
            return None

    def add_indexing_edges(self, wrapped_df: DfTypeWrapper, df_idx: str):
        df = wrapped_df.df
        if self.options.COLUMN_NODES and wrapped_df.is_df():
            for col_idx in range(len(df.columns)):
                if isinstance(df.columns, pd.MultiIndex):
                    num_col_levels: int = len(df.columns.levels)
                else:
                    num_col_levels: int = 1
                for df_col_row_idx in range(-num_col_levels, 0):
                    column_node = RelationGraphNode(df_idx, (df_col_row_idx, col_idx),
                                                    self.get_node_type(is_column=True))
                    for row_idx in range(len(df)):
                        inner_node = RelationGraphNode(df_idx, (row_idx, col_idx),
                                                       self.get_node_type(df.iloc[row_idx, col_idx]))
                        self.add_edge(column_node, inner_node, RelationGraphEdgeType.INDEX)

        if self.options.INDEX_NODES and wrapped_df.is_df():
            for row_idx in range(len(df)):
                if isinstance(df.index, pd.MultiIndex):
                    num_idx_levels: int = len(df.index.levels)
                else:
                    num_idx_levels: int = 1
                for df_idx_row_idx in range(-num_idx_levels, 0):
                    row_node = RelationGraphNode(df_idx, (row_idx, df_idx_row_idx),
                                                 self.get_node_type(is_index=True))
                    for col_idx in range(len(df.columns)):
                        inner_node = RelationGraphNode(df_idx, (row_idx, col_idx),
                                                       self.get_node_type(df.iloc[row_idx, col_idx]))
                        self.add_edge(row_node, inner_node, RelationGraphEdgeType.INDEX)

    def add_adjacency_edges(self, wrapped_df: DfTypeWrapper, df_idx: str):
        df = wrapped_df.df
        # check to make sure that this df is in fact a df by checking its true type
        if self.options.COLUMN_NODES and wrapped_df.is_df():
            if isinstance(df.columns, pd.MultiIndex):
                num_col_levels: int = len(df.columns.levels)
            else:
                num_col_levels: int = 1
            for df_col_row_idx in range(-num_col_levels, 0):
                for df_col_col_idx in range(len(df.columns)):
                    current_node = RelationGraphNode(df_idx, (df_col_row_idx, df_col_col_idx),
                                                     self.get_node_type(is_column=True))
                    if df_col_row_idx < -1:
                        below_node = RelationGraphNode(df_idx, (df_col_row_idx + 1, df_col_col_idx),
                                                       self.get_node_type(is_column=True))
                        self.add_edge(current_node, below_node, RelationGraphEdgeType.ADJACENCY)
                    if df_col_col_idx < len(df.columns) - 1:
                        beside_node = RelationGraphNode(df_idx, (df_col_row_idx, df_col_col_idx + 1),
                                                        self.get_node_type(is_column=True))
                        self.add_edge(current_node, beside_node, RelationGraphEdgeType.ADJACENCY)

        if self.options.INDEX_NODES and wrapped_df.is_df():
            if isinstance(df.index, pd.MultiIndex):
                num_idx_levels: int = len(df.index.levels)
            else:
                num_idx_levels: int = 1

            for df_idx_row_idx in range(len(df)):
                for df_idx_col_idx in range(-num_idx_levels, 0):
                    current_node = RelationGraphNode(df_idx, (df_idx_row_idx, df_idx_col_idx),
                                                     self.get_node_type(is_index=True))
                    if df_idx_row_idx < len(df) - 1:
                        below_node = RelationGraphNode(df_idx, (df_idx_row_idx + 1, df_idx_col_idx),
                                                       self.get_node_type(is_index=True))
                        self.add_edge(current_node, below_node, RelationGraphEdgeType.ADJACENCY)
                    if df_idx_col_idx < -1:
                        beside_node = RelationGraphNode(df_idx, (df_idx_row_idx, df_idx_col_idx + 1),
                                                        self.get_node_type(is_index=True))
                        self.add_edge(current_node, beside_node, RelationGraphEdgeType.ADJACENCY)

        for row_num in range(len(df)):
            for col_num in range(len(df.columns)):
                current_node = RelationGraphNode(df_idx, (row_num, col_num),
                                                 self.get_node_type(df.iloc[row_num, col_num]))
                if row_num < len(df) - 1:
                    below_node = RelationGraphNode(df_idx, (row_num + 1, col_num),
                                                   self.get_node_type(df.iloc[row_num + 1, col_num]))
                    self.add_edge(current_node, below_node, RelationGraphEdgeType.ADJACENCY)
                if col_num < len(df.columns) - 1:
                    beside_node = RelationGraphNode(df_idx, (row_num, col_num + 1),
                                                    self.get_node_type(df.iloc[row_num, col_num + 1]))
                    self.add_edge(current_node, beside_node, RelationGraphEdgeType.ADJACENCY)

    def value_to_node_map(self, wrapped_df: DfTypeWrapper, dfindex: str) -> Dict[Any, List[RelationGraphNode]]:
        df_values = defaultdict(list)
        df = wrapped_df.df
        # There are no columns for non-df objects
        if self.options.COLUMN_NODES and wrapped_df.is_df():
            if isinstance(df.columns, pd.MultiIndex):
                df_num_col_levels: int = len(df.columns.levels)
            else:
                df_num_col_levels: int = 1
            for df_col_row_idx in range(-df_num_col_levels, 0):
                for df_col_col_idx in range(len(df.columns)):
                    df_value = df.columns[df_col_col_idx][df_col_row_idx + df_num_col_levels] \
                        if isinstance(df.columns, pd.MultiIndex) else df.columns[df_col_col_idx]
                    df_node = RelationGraphNode(dfindex, (df_col_row_idx, df_col_col_idx),
                                                self.get_node_type(is_column=True))
                    df_values[df_value].append(df_node)

        # There are no values for type pointers
        if self.options.INDEX_NODES and wrapped_df.is_df():
            if isinstance(df.index, pd.MultiIndex):
                df_num_idx_levels: int = len(df.index.levels)
            else:
                df_num_idx_levels: int = 1
            for df_idx_row_idx in range(len(df)):
                for df_idx_col_idx in range(-df_num_idx_levels, 0):
                    df_value = df.index[df_idx_row_idx][df_idx_col_idx + df_num_idx_levels] \
                        if isinstance(df.index, pd.MultiIndex) else df.index[df_idx_row_idx]
                    df_node = RelationGraphNode(dfindex, (df_idx_row_idx, df_idx_col_idx),
                                                self.get_node_type(is_index=True))
                    df_values[df_value].append(df_node)

        if self.options.INDEX_NAME_NODES and wrapped_df.is_df():
            df_num_col_levels: int = len(df.columns.names)
            for df_col_idx_name_idx in range(-df_num_col_levels, 0):
                name_val = df.columns.names[df_num_col_levels + df_col_idx_name_idx]
                if name_val is None:
                    continue

                df_name_node = RelationGraphNode(dfindex, (df_col_idx_name_idx, -1),
                                                 self.get_node_type(is_col_index_name=True))
                df_values[name_val].append(df_name_node)

            df_num_idx_levels: int = len(df.index.names)
            for df_idx_name_idx in range(-df_num_idx_levels, 0):
                name_val = df.index.names[df_num_idx_levels + df_idx_name_idx]
                if name_val is None:
                    continue

                df_name_node = RelationGraphNode(dfindex, (-1, df_idx_name_idx),
                                                 self.get_node_type(is_index_name=True))
                df_values[name_val].append(df_name_node)

        for df_row_num in range(len(df)):
            for df_col_num in range(len(df.columns)):
                df_value = df.iloc[df_row_num, df_col_num]
                df_node = RelationGraphNode(dfindex, (df_row_num, df_col_num),
                                            self.get_node_type(df_value))

                if dfindex.startswith("E"):
                    self.add_node(df_node)

                df_values[df_value].append(df_node)

        return df_values

    def add_equality_edges(self, wrapped_df1: DfTypeWrapper, df1_idx: str, wrapped_df2: DfTypeWrapper, df2_idx: str):
        df1_values_to_nodes = self.value_to_node_map(wrapped_df1, df1_idx)
        df2_values_to_nodes = self.value_to_node_map(wrapped_df2, df2_idx)
        for df1_value in df1_values_to_nodes.keys():
            for df2_value in df2_values_to_nodes.keys():
                try:
                    if df1_value == df2_value:
                        for df1_node in df1_values_to_nodes[df1_value]:
                            for df2_node in df2_values_to_nodes[df2_value]:
                                self.add_edge(df1_node, df2_node, RelationGraphEdgeType.EQUALITY)
                except TypeError:
                    pass
                except ValueError:
                    pass
                except SyntaxError:
                    pass
                except Exception as e:
                    logger.err("Error comparing {} and {}".format(df1_value, df2_value))
                    logging.exception(e)

    def node_for_position(self, position: Tuple[int, int], df: pd.DataFrame, df_name: str):
        row = position[0]
        column = position[1]
        if row < 0:
            return RelationGraphNode(df_name, position, self.get_node_type(is_column=True))
        elif column < 0:
            return RelationGraphNode(df_name, position, self.get_node_type(is_index=True))
        else:
            return RelationGraphNode(df_name, position, self.get_node_type(df.iloc[row, column]))

    def add_dfs(self, input_dfs: List[DfTypeWrapper], output_dfs: List[DfTypeWrapper], extra_dfs: List[DfTypeWrapper]):
        #  Add Indexing Edges
        for input_df_idx in range(len(input_dfs)):
            input_df = input_dfs[input_df_idx]
            self.add_indexing_edges(input_df, "I%d" % input_df_idx)
        for output_df_idx in range(len(output_dfs)):
            output_df = output_dfs[output_df_idx]
            self.add_indexing_edges(output_df, "O%d" % output_df_idx)

        #  Add Adjacency Edges
        for input_df_idx in range(len(input_dfs)):
            input_df = input_dfs[input_df_idx]
            self.add_adjacency_edges(input_df, "I%d" % input_df_idx)
        for output_df_idx in range(len(output_dfs)):
            output_df = output_dfs[output_df_idx]
            self.add_adjacency_edges(output_df, "O%d" % output_df_idx)

        #  Add Equality Edges
        for input_df_idx in range(len(input_dfs)):
            for output_df_idx in range(len(output_dfs)):
                input_df = input_dfs[input_df_idx]
                output_df = output_dfs[output_df_idx]
                self.add_equality_edges(input_df, "I%d" % input_df_idx, output_df, "O%d" % output_df_idx)

        # Extras only get equality edges for now
        if extra_dfs:
            for input_df_idx in range(len(input_dfs)):
                for extra_df_idx in range(len(extra_dfs)):
                    input_df = input_dfs[input_df_idx]
                    extra_df = extra_dfs[extra_df_idx]
                    self.add_equality_edges(input_df, "I%d" % input_df_idx, extra_df, "E%d" % extra_df_idx)

            for output_df_idx in range(len(output_dfs)):
                for extra_df_idx in range(len(extra_dfs)):
                    output_df = output_dfs[output_df_idx]
                    extra_df = extra_dfs[extra_df_idx]
                    self.add_equality_edges(output_df, "O%d" % output_df_idx, extra_df, "E%d" % extra_df_idx)

    @classmethod
    def get_df(cls, val, mode='input'):
        if mode == 'extra':
            #  Quick sanity check.
            #  TODO : Handle more cases
            #  This is done as we only support singular extra nodes i.e. each extra node should
            #  correspond to exactly one node in the graph encoding
            if not (np.isscalar(val) or (val is None) or isinstance(val, Value)):
                raise NotImplementedError("Cannot handle {} in mode=extra".format(val))

        if isinstance(val, pd.DataFrame):
            return [DfTypeWrapper(pd.DataFrame(val))]

        elif isinstance(val, pd.Series):
            return [DfTypeWrapper(pd.DataFrame(val), 'series')]

        elif isinstance(val, np.ndarray):
            return [DfTypeWrapper(pd.DataFrame(val), 'ndarray')]

        elif isinstance(val, dict):
            try:
                return [DfTypeWrapper(pd.DataFrame(val), 'dict')]
            except ValueError:
                try:
                    return [DfTypeWrapper(pd.DataFrame(val, index=[0]), 'dict')]
                except:
                    raise NotImplementedError("Cannot handle {} dict : {}".format(mode, val))
            except:
                raise NotImplementedError("Cannot handle {} dict : {}".format(mode, val))

        elif isinstance(val, pd.MultiIndex):
            # not sure how to handle this so this is still going to be a DataFrame
            return [DfTypeWrapper(pd.DataFrame(list(map(list, list(val)))))]

        elif isinstance(val, pd.Index):
            return [DfTypeWrapper(pd.DataFrame(val), 'index')]

        elif np.isscalar(val):
            return [DfTypeWrapper(pd.DataFrame([val]), 'scalar')]

        elif val is None:
            return [DfTypeWrapper(pd.DataFrame([None]), 'scalar')]

        elif isinstance(val, Value):
            if isinstance(val, Default):
                result = cls.get_df(val.val, mode=mode)
                result[0].df = result[0].df.applymap(Default)
                return result

            return cls.get_df(val.val, mode=mode)

        elif isinstance(val, list):
            if all(map(np.isscalar, val)):
                return [DfTypeWrapper(pd.DataFrame(val), 'list')]

            return sum(map(lambda x: cls.get_df(x), val), [])

        elif isinstance(val, tuple):
            if all(map(np.isscalar, val)):
                return [DfTypeWrapper(pd.DataFrame(list(val)), 'tuple')]

            return sum(map(lambda x: cls.get_df(x), val), [])

        elif isinstance(val, Callable):
            try:
                if val.__name__ == "<lambda>":
                    return [DfTypeWrapper(pd.DataFrame([val]), 'lambda')]
            except:
                pass

            if isinstance(val, Lambda):
                return [DfTypeWrapper(pd.DataFrame([eval(val.fn)]), 'lambda')]

        elif isinstance(val, pd.core.groupby.groupby.GroupBy):
            # I think this should be a dataframe
            return [DfTypeWrapper(pd.DataFrame(val.get_group(group))) for group in val.groups]

        raise NotImplementedError("Cannot handle {} : {}".format(mode, val))

    @classmethod
    def convert_input_output(cls, inputs, output, options: RelationGraphOptions, extras=None):
        input_dfs = []
        for input_ in inputs:
            try:
                input_dfs += cls.get_df(input_)
            except NotImplementedError:
                raise
            except Exception as e:
                logger.err("Error while getting df for input : {}".format(input_))
                print(e)
                raise NotImplementedError("Caught exception for input : {}".format(input_))

        for _inp in input_dfs:
            _inp = _inp.df
            if len(_inp) > 100 or len(_inp.columns) > 100:
                raise NotImplementedError("Cannot handle inputs with >100 rows or columns")

        try:
            output_dfs = cls.get_df(output, mode='output')
        except NotImplementedError:
            raise
        except Exception as e:
            logger.err("Error while getting df for output : {}".format(output))
            print(e)
            raise NotImplementedError("Caught exception for output : {}".format(output))

        extra_dfs = []
        if extras:
            for extra_ in extras:
                try:
                    extra_dfs += cls.get_df(extra_, mode='extra')
                except NotImplementedError:
                    raise
                except Exception as e:
                    logger.err("Error while getting df for extra : {}".format(extra_))
                    print(e)
                    raise NotImplementedError("Caught exception for input : {}".format(extra_))

        return input_dfs, output_dfs, extra_dfs

    @classmethod
    def build_relation_graph(cls, inputs, output, options=RelationGraphOptions(), program=None,
                             extras=None) -> 'RelationGraph':

        relation_graph: RelationGraph = RelationGraph(options)
        input_dfs, output_dfs, extra_dfs = cls.convert_input_output(inputs, output, options, extras)

        for _out in output_dfs:
            _out = _out.df
            if len(_out) > 100 or len(_out.columns) > 100:
                raise NotImplementedError("Cannot handle outputs with >100 rows or columns")

        try:
            relation_graph.add_dfs(input_dfs, output_dfs, extra_dfs)
        except Exception as e:
            logger.err("Error while adding dfs")
            print(e)
            raise NotImplementedError("Caught exception while adding dfs")

        relation_graph.input_dfs = input_dfs
        relation_graph.output_dfs = output_dfs

        return relation_graph

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def to_dict(self, get_mapping=False) -> Union[Dict, Tuple[Dict, Dict]]:
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

        #  Create a node to int mapping
        num = 0
        node_to_int: Dict[RelationGraphNode, int] = {}
        for node in self.nodes:
            node_to_int[node] = num
            num += 1

        edges: List[List] = []
        for edge in self.edges:
            src = node_to_int[edge.node1]
            dst = node_to_int[edge.node2]
            edges.append([src, edge.get_feature_encoding(), dst])

        node_features = [n.get_feature_encoding() for n in node_to_int]

        result = {
            'edges': edges,
            'node_features': node_features
        }

        if get_mapping:
            return result, node_to_int

        return result

    def to_dict_old(self) -> Dict:
        """
        The format of the dict is as follows
        {
            'nodes': [
                {
                    'node_feat': node_feat,
                    'neighbors': [(n, edge_feat) for n in neighbors]
                }
                {
                    'node_feat': ...,
                    'neighbors': ...
                }
            ]
        }

        As of now, we assume that there are NO redundant edges
        Also, each node is assumed to have a single type associated with it
        """

        #  First create a node to edges mapping
        node_to_edges: Dict[RelationGraphNode,
                            List[Tuple[RelationGraphNode, RelationGraphEdge]]] = defaultdict(list)

        for edge in self.edges:
            node_to_edges[edge.node1].append((edge.node2, edge))
            node_to_edges[edge.node2].append((edge.node1, edge))

        #  Create a node to int mapping
        num = 0
        node_to_int: Dict[RelationGraphNode, int] = {}
        for node in node_to_edges.keys():
            node_to_int[node] = num
            num += 1

        nodes_dicts: List[Dict] = []
        for node, edges in node_to_edges.items():
            node_feat = node.get_feature_encoding()
            neighbours = []
            for n_node, edge in edges:
                neighbours.append((node_to_int[n_node], edge.get_feature_encoding()))

            d = {
                'node_feat': node_feat,
                'neighbors': neighbours  # British vs American :|
            }

            nodes_dicts.append(d)

        result = {
            'nodes': nodes_dicts
        }

        return result
