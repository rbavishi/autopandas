import os
import pickle
from abc import abstractmethod, ABC
from typing import Any, Dict, Tuple, List

from autopandas_v2.utils.checker import Checker
from autopandas_v2.utils.exceptions import AutoPandasException


class MockBasicModel(ABC):
    def __init__(self, behavior: Dict[str, Any]):
        """
        Behavior should contain for each label corresponding to a DSL op,
        the values a neural network is supposed to return.
        """

        self.behavior = behavior

    @abstractmethod
    def get_mocked_inference(self, label: str, graph, **kwargs):
        return self.behavior[label]

    @abstractmethod
    def save_interface(self, model_dir: str):
        pass

    def save(self, model_dir: str):
        if not os.path.exists(model_dir):
            os.system('mkdir -p ' + model_dir)

        with open(model_dir + '/behavior.pkl', 'wb') as f:
            pickle.dump(self.behavior, f)

        self.save_interface(model_dir)


class MockChoiceModel(MockBasicModel):
    def save_interface(self, model_dir: str):
        from autopandas_v2.generators.ml.networks.mocking.interfaces import MockChoiceInterface
        interface = MockChoiceInterface()
        with open(model_dir + '/interface.pkl', 'wb') as f:
            pickle.dump(interface, f)

    def get_mocked_inference(self, label: str, graph, **kwargs):
        choices_raw = graph['choices_raw']
        # This should be an ordering containing the probabilities and raw choice values
        ordering = self.behavior[label]
        result: List[Tuple[float, int]] = []
        for prob, val in ordering:
            for idx, raw_val in enumerate(choices_raw):
                if Checker.check(val, raw_val):
                    result.append((prob, idx))
                    break
            else:
                return []
                # raise AutoPandasException("Mocker behavior does not match query")

        return result


class MockChainModel(MockBasicModel):
    def save_interface(self, model_dir: str):
        from autopandas_v2.generators.ml.networks.mocking.interfaces import MockChainInterface
        interface = MockChainInterface()
        with open(model_dir + '/interface.pkl', 'wb') as f:
            pickle.dump(interface, f)

    def get_mocked_inference(self, label: str, graph, **kwargs):
        # This should be an ordering containing the probabilities and index values
        return self.behavior[label]


class MockSelectModel(MockBasicModel):
    def save_interface(self, model_dir: str):
        from autopandas_v2.generators.ml.networks.mocking.interfaces import MockSelectInterface
        interface = MockSelectInterface()
        with open(model_dir + '/interface.pkl', 'wb') as f:
            pickle.dump(interface, f)

    def get_mocked_inference(self, label: str, graph, **kwargs):
        domain_raw = graph['domain_raw']
        # This should be an ordering containing the probabilities and  raw domain values
        ordering = self.behavior[label]
        result: List[Tuple[float, int]] = []
        for prob, val in ordering:
            for idx, raw_val in enumerate(domain_raw):
                if Checker.check(val, raw_val):
                    result.append((prob, idx))
                    break
            else:
                return []
                # raise AutoPandasException("Mocker behavior does not match query")

        return result


class MockSubsetsModel(MockBasicModel):
    def save_interface(self, model_dir: str):
        from autopandas_v2.generators.ml.networks.mocking.interfaces import MockSubsetsInterface
        interface = MockSubsetsInterface()
        with open(model_dir + '/interface.pkl', 'wb') as f:
            pickle.dump(interface, f)

    def get_mocked_inference(self, label: str, graph, **kwargs):
        # This should be a list of tuples (discard_prob, keep_prob, val)
        vals_with_probs = self.behavior[label]
        result: List[Tuple[float, float, int]] = []
        for discard_prob, keep_prob, val in vals_with_probs:
            for idx, raw_val in enumerate(graph['raw_vals']):
                if Checker.check(val, raw_val):
                    result.append((discard_prob, keep_prob, idx))
                    break
            else:
                return []

        return result


class MockOrderedSubsetsModel(MockBasicModel):
    def save_interface(self, model_dir: str):
        from autopandas_v2.generators.ml.networks.mocking.interfaces import MockOrderedSubsetsInterface
        interface = MockOrderedSubsetsInterface()
        with open(model_dir + '/interface.pkl', 'wb') as f:
            pickle.dump(interface, f)

    def get_mocked_inference(self, label: str, graph, beam_search_k: int = None, **kwargs):
        # This should be an ordering containing the probabilities and index values
        return self.behavior[label]
