from typing import List, Any

from autopandas_v2.generators.trackers import OpTracker
from autopandas_v2.iospecs import EngineSpec, SearchSpec
from autopandas_v2.synthesis.search.results.programs import Program


class ExplorationSpec(EngineSpec):
    def __init__(self, inputs: List[Any], output: Any, max_depth: int):
        super().__init__(inputs, output, max_depth)
        self.tracking: List[OpTracker] = [None] * max_depth
        self.program: Program = None


class GeneratorInversionSpec(SearchSpec):
    def __init__(self, inputs: List[Any], output: Any, intermediates: List[Any], trackers: List[OpTracker]):
        super().__init__(inputs, output, intermediates=intermediates)
        self.trackers = trackers
