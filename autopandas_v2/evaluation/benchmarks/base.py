from typing import Any, List


class Benchmark:
    def __init__(self):
        self.inputs: List[Any] = None
        self.output: Any = None
        self.funcs: List[str] = None
        self.seqs: List[List[int]] = None

    def unwrap(self):
        return self.inputs, self.output, self.funcs, self.seqs
