from typing import Generator, List, Set

from autopandas_v2.generators.base import BaseGenerator
from autopandas_v2.generators.ml.traindata.exploration.engines.arguments import RandArgEngine
from autopandas_v2.generators.ml.traindata.exploration.iospecs import ExplorationSpec
from autopandas_v2.iospecs import IOSpec
from autopandas_v2.synthesis.search.engines.functions import BaseEngine
from autopandas_v2.synthesis.search.results.programs import FunctionCall, Program
from autopandas_v2.utils.cli import ArgNamespace


class RandProgEngine(BaseEngine):
    """
    This engine, given a pre-defined function sequence, produces a random I/O spec + program using those functions
    """

    def __init__(self, func_seq: List[BaseGenerator], cmd_args: ArgNamespace = None):
        iospec: IOSpec = IOSpec(inputs=[], output=None)
        super().__init__(iospec, cmd_args)

        self.func_seq = func_seq
        self.max_depth = len(func_seq)
        self.stats = None

    def iter_func_seqs(self) -> Generator[List[BaseGenerator], None, None]:
        yield self.func_seq

    def report_example(self, spec: ExplorationSpec, programs: List[Set[FunctionCall]]):
        #  We know that each set only contains one call
        call_seq = [list(i)[0] for i in programs]
        program = Program(list(call_seq))
        spec.program = program
        return spec

    def generate(self):
        #  Making a copy of iospec.inputs is important because we modify the spec while exploring random programs
        self.engine_spec = ExplorationSpec(self.iospec.inputs[:], self.iospec.output, max_depth=self.max_depth)

        for func_seq in self.iter_func_seqs():
            arg_engine = RandArgEngine(func_seq, self.cmd_args, stats=None)
            for result, programs in arg_engine.run(self.engine_spec):
                self.engine_spec.output = result
                return self.report_example(self.engine_spec, programs)
