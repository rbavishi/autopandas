from abc import abstractmethod, ABC
from typing import List, Tuple, Dict

from autopandas_v2.ml.featurization.featurizer import RelationGraph
from autopandas_v2.ml.inference.interfaces import RelGraphInterface
from autopandas_v2.ml.networks.ggnn.models.base import BaseGGNN
from autopandas_v2.ml.networks.ggnn.models.sparse.base import SparseGGNN
from autopandas_v2.ml.networks.ggnn.models.sparse.seq.static_rnn import GGNNSeqStaticRNN


class GGNNInterface(RelGraphInterface, ABC):
    @abstractmethod
    def get_model(self):
        pass

    def init(self, model_dir):
        self.model: BaseGGNN = self.get_model()
        self.model.setup_inference(model_dir)

    def predict_graphs(self, graphs: List[Dict], with_confidence=True, **kwargs) -> List[List[Tuple[str, float]]]:
        return self.model.infer(graphs, **kwargs)

    def close(self):
        self.model.close()


class SparseGGNNInterface(GGNNInterface):
    def get_model(self):
        return SparseGGNN()

    def debug_graph(self, graph: RelationGraph):
        pass


class GGNNSeqStaticRNNInterface(GGNNInterface):
    def get_model(self):
        return GGNNSeqStaticRNN()

    def debug_graph(self, graph: RelationGraph):
        pass
