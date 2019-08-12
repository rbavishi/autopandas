from typing import Any, List, Dict


class IOSpec:
    def __init__(self, inputs: List[Any], output: Any):
        self.inputs = inputs
        self.output = output
        self.funcs: List[str] = None
        self.seqs: List[List[int]] = None


class SearchSpec(IOSpec):
    def __init__(self, inputs: List[Any], output: Any, intermediates: List[Any] = None, max_depth: int = None,
                 depth: int = 1):
        super().__init__(inputs, output)

        if intermediates is None and max_depth is None:
            raise Exception("One of intermediates and max_depth is required")

        self.intermediates = intermediates if intermediates is not None else [None] * (max_depth - 1)
        self.max_depth = max_depth or (len(intermediates) + 1)
        self.depth = depth


class ArgTrainingSpec(SearchSpec):
    def __init__(self, inputs: List[Any], output: Any, args: Dict[str, Any], intermediates: List[Any] = None,
                 max_depth: int = None, depth: int = 1):
        super().__init__(inputs, output, intermediates, max_depth, depth)
        self.args = args


class EngineSpec(SearchSpec):
    def __init__(self, inputs: List[Any], output: Any, max_depth: int):
        super().__init__(inputs, output, max_depth=max_depth)
