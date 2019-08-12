import functools
import operator
from typing import Any, Dict, List

from autopandas_v2.generators.base import BaseGenerator
from autopandas_v2.synthesis.search.results.computers import ArgComputer


class FunctionCall:
    """
    This represents a single call in the overall program sequence
    """

    def __init__(self, gen: BaseGenerator, arg_vals: Dict[str, Any], arg_annotations: Dict[str, Dict[str, Any]]):
        self.fname = gen.name
        self.arg_vals = arg_vals
        self.arg_annotations = arg_annotations
        self.computers: Dict[str, ArgComputer] = {k: ArgComputer(k, v, arg_annotations[k]) for k, v in arg_vals.items()}
        self.representation: str = gen.representation.format(**{k: v.repr for k, v in self.computers.items()})

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return self.representation

    def get_used_intermediates(self):
        return functools.reduce(operator.ior,
                                (computer.get_used_intermediates() for computer in self.computers.values()))

    def get_used_inputs(self):
        return functools.reduce(operator.ior,
                                (computer.get_used_inputs() for computer in self.computers.values()))


class Program:
    """
    This represents a full program as a sequence of function calls
    """
    def __init__(self, call_seq: List[FunctionCall]):
        self.call_seq = call_seq
        self.representation: str = None

        self.init()

    def init(self):
        self.representation = ""
        for idx, call in enumerate(self.call_seq[:-1]):
            self.representation += "v{} = {}\n".format(idx, call)

        self.representation += "output = {}".format(self.call_seq[-1])

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return self.representation
