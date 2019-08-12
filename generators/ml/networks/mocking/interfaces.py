import pickle
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

from autopandas_v2.generators.ml.networks.mocking.models import MockSelectModel, MockChoiceModel, MockChainModel, \
    MockSubsetsModel, MockOrderedSubsetsModel
from autopandas_v2.ml.inference.interfaces import RelGraphInterface


class BaseMockInterface(RelGraphInterface, ABC):
    @abstractmethod
    def get_model(self, model_dir: str):
        pass

    def init(self, model_dir):
        self.model = self.get_model(model_dir)

    def predict_graphs(self, graphs: List[Dict], with_confidence=True, **kwargs) -> List[List[Tuple[str, float]]]:
        return [self.model.get_mocked_inference(graph['op_label'], graph, **kwargs) for graph in graphs]

    def debug_graph(self, graph: Dict):
        pass


class MockChoiceInterface(BaseMockInterface):
    def get_model(self, model_dir: str):
        with open(model_dir + '/behavior.pkl', 'rb') as f:
            behavior = pickle.load(f)

        return MockChoiceModel(behavior)


class MockChainInterface(BaseMockInterface):
    def get_model(self, model_dir: str):
        with open(model_dir + '/behavior.pkl', 'rb') as f:
            behavior = pickle.load(f)

        return MockChainModel(behavior)


class MockSelectInterface(BaseMockInterface):
    def get_model(self, model_dir: str):
        with open(model_dir + '/behavior.pkl', 'rb') as f:
            behavior = pickle.load(f)

        return MockSelectModel(behavior)


class MockSubsetsInterface(BaseMockInterface):
    def get_model(self, model_dir: str):
        with open(model_dir + '/behavior.pkl', 'rb') as f:
            behavior = pickle.load(f)

        return MockSubsetsModel(behavior)


class MockOrderedSubsetsInterface(BaseMockInterface):
    def get_model(self, model_dir: str):
        with open(model_dir + '/behavior.pkl', 'rb') as f:
            behavior = pickle.load(f)

        return MockOrderedSubsetsModel(behavior)
