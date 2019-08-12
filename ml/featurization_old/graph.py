from enum import Enum
from typing import Any
import pandas as pd
import numpy as np

from autopandas_v2.ml.featurization_old.options import GraphOptions


class GraphNodeType(Enum):
    COLUMN = 0
    INDEX = 1
    OBJECT = 2
    FLOAT = 3
    INT = 4
    STR = 5
    DATE = 6
    DELTA = 7
    BOOL = 8

    @classmethod
    def get_node_type(cls, value):
        # Gets the `RelationGraphNodeType` for the given value. Does not return column name or index
        # type --- those should be encoded separately.
        val_type = type(value)
        if np.issubdtype(val_type, np.floating):
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
        else:
            val_type = cls.OBJECT
        return val_type


class GraphNode:
    def __init__(self, source: str, identifier: str, ntype: GraphNodeType):
        """
        Source is the obj from which this node created (such as the dataframe which this node comes from
        Identifier is the identifier of this node within the collection representing the aforementioned object
        NType is the node type
        """
        self.source = source
        self.identifier = identifier
        self.ntype = ntype

    @classmethod
    def from_obj(cls, obj: Any, source: str, identifier: str, options: GraphOptions):
        return GraphNode(source, identifier, GraphNodeType.get_node_type(obj))

    def __hash__(self):
        res = 17
        res = 31 * res + hash(self.source)
        res = 31 * res + hash(self.identifier)
        res = 31 * res + hash(self.ntype.value)
        return res

    def __eq__(self, other):
        if not isinstance(other, GraphNode):
            return False

        return self.source == other.source and self.identifier == other.identifier and self.ntype == other.ntype

    def __ne__(self, other):
        return not (self.__eq__(other))

    def __repr__(self):
        if self.identifier != '':
            return self.source + ':' + self.identifier
        else:
            return self.source

    def get_encoding(self):
        source_id = {
            'I': 0,
            'O': 1,
            'E': 2,
            'T': 2,
        }[self.source[0]]

        if self.ntype is not None:
            type_encoding: int = self.ntype.value
            return type_encoding + source_id * len(GraphNodeType)
        else:
            return source_id


class GraphEdgeType(Enum):
    ADJACENCY = 0
    EQUALITY = 1
    INDEX = 2

    @classmethod
    def get_encoding(cls, val):
        return val.value


class GraphEdge:
    def __init__(self, node1: GraphNode, node2: GraphNode, etype: GraphEdgeType):
        """
        We avoid using the names src and dst as this is supposed to be a general edge that can be interpreted
        a directed or an undirected edge as well. The equality subroutine would have to be overridden though
        """
        self.node1 = node1
        self.node2 = node2
        self.etype = etype

    def __hash__(self):
        res = 17
        res = 31 * res + hash(self.node1) + hash(self.node2)
        res = 31 * res + hash(self.etype)
        return res

    def __eq__(self, other):
        if not isinstance(other, GraphEdge):
            return False

        return self.etype == other.etype and ((self.node1 == other.node1 and self.node2 == other.node2) or
                                              (self.node1 == other.node2 and self.node2 == other.node1))

    def __ne__(self, other):
        return not (self.__eq__(other))

    def __repr__(self):
        return '{} -> {}'.format(self.node1, self.node2)

    def get_encoding(self):
        return GraphEdgeType.get_encoding(self.etype)
