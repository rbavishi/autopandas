from typing import Set

from autopandas_v2.ml.featurization_old.graph import GraphEdge, GraphNode, GraphEdgeType


class EdgeCollection:
    def __init__(self):
        self.edges: Set[GraphEdge] = set()

    def add_edge(self, node1: GraphNode, node2: GraphNode, etype: GraphEdgeType):
        # print("Adding edge {} between {} and {}".format(etype, node1, node2))
        self.edges.add(GraphEdge(node1, node2, etype))
