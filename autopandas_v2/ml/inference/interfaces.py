import pickle
from abc import ABC, abstractmethod
from typing import Tuple, List, Any, Dict

from autopandas_v2.ml.featurization.featurizer import RelationGraph
from autopandas_v2.ml.featurization.options import GraphOptions


class GenericInterface(ABC):
    @abstractmethod
    def init(self, model_dir):
        pass

    @abstractmethod
    def predict(self, dpoints: List[Tuple[List[Any], Any]], with_confidence=True) -> List[List[Tuple[str, float]]]:
        pass

    @abstractmethod
    def debug(self, dpoint: Tuple[List[Any], Any]):
        pass

    @staticmethod
    def from_model_dir(model_dir: str) -> 'GenericInterface':
        with open('{}/interface.pkl'.format(model_dir), 'rb') as f:
            interface: GenericInterface = pickle.load(f)

        interface.init(model_dir)
        return interface

    @abstractmethod
    def close(self):
        pass


class RelGraphInterface(GenericInterface, ABC):
    def predict(self, dpoints: List[Tuple[List[Any], Any]], with_confidence=True) -> List[List[Tuple[str, float]]]:
        relgraphs = []
        for dpoint in dpoints:
            inputs, output = dpoint
            graph = RelationGraph(GraphOptions())
            graph.from_input_output(inputs, output)
            relgraphs.append(graph.get_encoding(get_mapping=False))

        return self.predict_graphs(relgraphs, with_confidence=with_confidence)

    @staticmethod
    def from_model_dir(model_dir: str) -> 'RelGraphInterface':
        with open('{}/interface.pkl'.format(model_dir), 'rb') as f:
            interface: RelGraphInterface = pickle.load(f)

        interface.init(model_dir)
        return interface

    @abstractmethod
    def predict_graphs(self, graphs: List[Dict], with_confidence=True, **kwargs) -> List[List[Tuple[str, float]]]:
        pass

    def debug(self, dpoint: Tuple[List[Any], Any]):
        inputs, output = dpoint
        graph = RelationGraph(GraphOptions())
        graph.from_input_output(inputs, output)

        self.debug_graph(graph.get_encoding(get_mapping=False))

    @abstractmethod
    def debug_graph(self, graph: Dict):
        pass
