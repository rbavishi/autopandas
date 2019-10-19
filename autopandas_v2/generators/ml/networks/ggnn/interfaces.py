from autopandas_v2.generators.ml.networks.ggnn.ops.chain import ModelChain
from autopandas_v2.generators.ml.networks.ggnn.ops.choice import ModelChoice
from autopandas_v2.generators.ml.networks.ggnn.ops.select import ModelSelect
from autopandas_v2.generators.ml.networks.ggnn.ops.subsets import ModelSubsets
from autopandas_v2.generators.ml.networks.ggnn.ops.orderedsubsets import ModelOrderedSubsets
from autopandas_v2.ml.featurization.featurizer import RelationGraph
from autopandas_v2.ml.networks.ggnn.inference.interfaces import GGNNInterface


class ModelSelectInterface(GGNNInterface):
    def get_model(self):
        return ModelSelect()

    def debug_graph(self, graph: RelationGraph):
        pass


class ModelChoiceInterface(GGNNInterface):
    def get_model(self):
        return ModelChoice()

    def debug_graph(self, graph: RelationGraph):
        pass


class ModelChainInterface(GGNNInterface):
    def get_model(self):
        return ModelChain()

    def debug_graph(self, graph: RelationGraph):
        pass


class ModelSubsetsInterface(GGNNInterface):
    def get_model(self):
        return ModelSubsets()

    def debug_graph(self, graph: RelationGraph):
        pass


class ModelOrderedSubsetsInterface(GGNNInterface):
    def get_model(self):
        return ModelOrderedSubsets()

    def debug_graph(self, graph: RelationGraph):
        pass
