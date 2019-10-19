"""
This file contains engines that systematically explore the space of function sequences.
The exploration of the consequent argument space is done using argument engines.
These engines therefore are responsible for choosing which function sequences to explore,
instantiating the argument engines and choosing how much of that argument engine to explore.

Because of the way the argument engines are structured, these function sequence engines can also
use the pausing/restarting of argument engines to revisit a function sequence later on.
"""
import itertools
from abc import ABC, abstractmethod
from typing import Generator, List, Type, Set, Dict

import autopandas_v2.synthesis.search.engines.arguments as arg_engines
from autopandas_v2.generators.base import BaseGenerator
from autopandas_v2.generators.utils import load_generators
from autopandas_v2.iospecs import IOSpec, EngineSpec
from autopandas_v2.ml.inference.interfaces import RelGraphInterface
from autopandas_v2.synthesis.search.results.programs import Program, FunctionCall
from autopandas_v2.synthesis.search.stats.collectors import StatsCollector
from autopandas_v2.utils.checker import Checker
from autopandas_v2.utils.cli import ArgNamespace


class BaseEngine(ABC):
    def __init__(self, iospec: IOSpec, cmd_args: ArgNamespace = None):
        self.iospec = iospec
        if cmd_args is None:
            cmd_args = ArgNamespace()

        self.cmd_args = cmd_args
        self.max_depth = cmd_args.get('max_depth', 1)
        self.engine_spec: EngineSpec = None
        self.solutions: List[Program] = None

        self.argument_engine = cmd_args.get('arg_engine', 'bfs')
        self.arg_model_dir = cmd_args.get('arg_model_dir', None)
        self.arg_top_k = cmd_args.get('top_k_args', None)

        #  Knobs
        self.collect_stats = cmd_args.get('collect_stats', True)
        self.silent = cmd_args.get('engine_silent', False)
        self.stop_first_solution = cmd_args.get('stop_first_solution', False)

        self.stats = None if not self.collect_stats else StatsCollector()

    def get_arg_engine(self, func_seq: List[BaseGenerator]) -> arg_engines.BaseArgEngine:
        if self.argument_engine == 'bfs':
            return arg_engines.BreadthFirstEngine(func_seq, self.cmd_args, stats=self.stats)

        elif self.argument_engine == 'beam-search':
            return arg_engines.BeamSearchEngine(func_seq, model_path=self.arg_model_dir,
                                                cmd_args=self.cmd_args, stats=self.stats,
                                                k=self.arg_top_k)

    def report_solution(self, programs: List[Set[FunctionCall]]):
        for call_seq in itertools.product(*programs):
            program = Program(list(call_seq))
            self.solutions.append(program)
            if not self.silent:
                print(program)

    @abstractmethod
    def iter_func_seqs(self) -> Generator[List[BaseGenerator], None, None]:
        """
        This method controls the order in which function sequences are added to the worklist
        """
        pass

    def search(self) -> bool:
        """
        This method dictates how the sequences are processed.
        The basic version here either processes a sequence fully or permanently discard it
        """

        target_output = self.iospec.output
        checker = Checker.get_checker(target_output)
        self.solutions = []
        self.engine_spec = EngineSpec(self.iospec.inputs, self.iospec.output, max_depth=self.max_depth)

        for func_seq in self.iter_func_seqs():
            arg_engine = self.get_arg_engine(func_seq)
            for result, programs in arg_engine.run(self.engine_spec):
                if checker(target_output, result):
                    self.report_solution(programs)
                    if self.stop_first_solution:
                        return True

            arg_engine.close()

        return len(self.solutions) > 0


class BFSEngine(BaseEngine):
    def __init__(self, iospec: IOSpec, cmd_args: ArgNamespace = None):
        super().__init__(iospec, cmd_args)
        self.use_spec_seqs = self.cmd_args.get('use_spec_seqs', False)
        self.use_spec_funcs = self.cmd_args.get('use_spec_funcs', False)

    def iter_func_seqs(self) -> Generator[List[BaseGenerator], None, None]:
        generators: Dict[str, BaseGenerator] = load_generators()
        if self.use_spec_funcs and self.iospec.funcs is not None:
            generators = {k: v for k, v in generators.items() if k in self.iospec.funcs}

        if self.use_spec_seqs and self.iospec.seqs is not None:
            for seq in self.iospec.seqs:
                yield [generators[self.iospec.funcs[i]] for i in seq]

            return

        raise NotImplementedError("Iteration of function sequences not defined yet")


class NeuralEngine(BaseEngine):
    def __init__(self, iospec: IOSpec, model_path: str, cmd_args: ArgNamespace = None, top_k: int = 10):
        super().__init__(iospec, cmd_args)
        self.model_path = model_path
        self.top_k = top_k
        self.use_old_featurization = False

    def iter_func_seqs(self) -> Generator[List[BaseGenerator], None, None]:
        generators: Dict[str, BaseGenerator] = load_generators()
        model = RelGraphInterface.from_model_dir(self.model_path)
        if self.use_old_featurization:
            from autopandas_v2.ml.featurization_old.featurizer import RelationGraph
            from autopandas_v2.ml.featurization_old.options import GraphOptions
        else:
            from autopandas_v2.ml.featurization.featurizer import RelationGraph
            from autopandas_v2.ml.featurization.options import GraphOptions

        options = GraphOptions()
        graph: RelationGraph = RelationGraph(options)
        graph.from_input_output(self.iospec.inputs, self.iospec.output)
        encoding = graph.get_encoding(get_mapping=False)

        str_seqs, probs = list(zip(*model.predict_graphs([encoding], top_k=self.top_k)[0]))
        str_seqs = [i.split(':') for i in str_seqs]
        model.close()

        for str_seq in str_seqs:
            yield [generators[i] for i in str_seq]

    def search(self) -> bool:
        """
        This method dictates how the sequences are processed.
        The basic version here either processes a sequence fully or permanently discard it
        """

        target_output = self.iospec.output
        checker = Checker.get_checker(target_output)
        self.solutions = []

        for func_seq in self.iter_func_seqs():
            if self.stats is not None:
                self.stats.num_seqs_explored += 1

            self.engine_spec = EngineSpec(self.iospec.inputs, self.iospec.output, max_depth=len(func_seq))
            arg_engine = self.get_arg_engine(func_seq)
            for result, programs in arg_engine.run(self.engine_spec):
                if checker(target_output, result):
                    self.report_solution(programs)
                    if self.stop_first_solution:
                        return True

            arg_engine.close()

        return len(self.solutions) > 0
