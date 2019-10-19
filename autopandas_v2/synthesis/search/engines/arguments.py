"""
This file contains engines that explore the argument space.
Given a function or a sequence of functions, these engines systematically explore
the argument combinations (or programs when combined with the function) and check if they
satisfy the specification at hand

These engines fundamentally are Python generators (iterators that accept a next() call).
So at a high-level they can be paused and resumed later. These can be really useful for writing
sophisticated function-sequence level engines, which can choose to process a sequence or come back to it
later on without exploring it fully
"""
import collections
import functools
import glob
import itertools
import logging
import operator
import os
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Set

from autopandas_v2.generators.base import BaseGenerator
from autopandas_v2.iospecs import EngineSpec
from autopandas_v2.ml.inference.interfaces import RelGraphInterface
from autopandas_v2.synthesis.search.results.programs import FunctionCall
from autopandas_v2.synthesis.search.stats.collectors import StatsCollector
from autopandas_v2.utils.checker import Checker
from autopandas_v2.utils.cli import ArgNamespace
from autopandas_v2.utils.hasher import Hasher


class BaseArgEngine(ABC):
    def __init__(self, func_sequence: List[BaseGenerator], cmd_args: ArgNamespace = None, stats: StatsCollector = None):
        self.func_sequence = func_sequence
        self.cmd_args = cmd_args
        self.stats = stats

    def execute(self, func: BaseGenerator, args: Dict[str, Any], arg_annotations: Dict[str, Dict[str, Any]],
                debug=False):
        try:
            if '' in arg_annotations.values():
                return func.target(**{k: args[k] for k, v in arg_annotations.items()
                                      if v != ''})
            else:
                return func.target(**args)

        except Exception as e:
            #  None is a safe choice because we don't care about programs producing None anyway
            if debug:
                print(args)
                logging.exception(e)
            return None

    def get_used_intermediates(self, programs: List[Set[FunctionCall]]) -> Set[int]:
        used = set()
        for calls in programs:
            if calls is None:
                break

            for call in calls:
                used |= functools.reduce(operator.ior,
                                         (computer.get_used_intermediates() for computer in call.computers.values()))

        return used

    def iter_args_wrapper(self, spec: EngineSpec, depth: int, current_programs: List[Set[FunctionCall]]):
        #  If this is the final function, get a list of all the intermediates that have been consumed till now.
        #  We can then use this information to avoid generating candidates that contain no-ops.
        #  For example, a program such as v0 = inps[0].some_func1(), output = inps[0].some_func2()
        #  where some_func2() doesn't use v0 should be considered a spurious solution
        used_intermediates = self.get_used_intermediates(current_programs)
        max_consumed_later = max([1] + [i.arity for i in self.func_sequence[depth:]])

        for candidate in self.iter_args(spec, depth, current_programs):
            #  Enumerators can choose to return stuff other than vals and annotations (for eg. trackers)
            arg_vals, arg_annotations = candidate[:2]
            new_used = set()
            for annotation in arg_annotations.values():
                if (not annotation) or 'sources' not in annotation:
                    continue

                new_used |= {idx for src, idx in zip(annotation['sources'], annotation['indices'])
                             if src == 'intermediates'}

            # Includes the intermediate produced at this depth
            num_unused = depth - len(used_intermediates | new_used)
            if num_unused > max_consumed_later:
                continue

            yield candidate

    @abstractmethod
    def iter_args(self, spec: EngineSpec, depth: int, current_programs: List[Set[FunctionCall]]):
        """
        Given a spec and the the depth (consequently the function to be explored), explore the
        argument combinations systematically. This function should be redefined in the subclasses
        for implementing variants such as naive-exhaustive, smart etc.

        The current_programs argument contains the set of programs that produces the EngineSpec at this
        depth. This can be useful for pruning candidates.
        """
        pass

    @abstractmethod
    def iter_specs(self, inp_spec: EngineSpec, depth: int, programs: List[Set[FunctionCall]] = None):
        """
        Given a depth and an input spec, this goes over all the argument combinations,
        computes the result of executing the function with those arguments, and outputs a
        new EngineSpec which contains this result as an additional intermediate.

        The programs argument is supposed to keep track of all the programs generating the intermediates/outputs
        These will be represented as a list of sets of function-calls, where each set corresponds to a particular
        depth. The set of programs will then be defined as the cross-product of all these sets
        """
        pass

    def run(self, spec: EngineSpec):
        yield from self.iter_specs(spec, depth=1, programs=None)

    def close(self):
        pass


class BreadthFirstEngine(BaseArgEngine):
    """
    This engine is called breadth-first because it first computes all the intermediates generated by
    the first function in the sequence, and then only proceeds to the next and so on.
    This can be advantageous for saving exploration times by clubbing intermediates that can be produced
    by multiple argument combinations, but can also be disadvantageous as it would spend time exploring
    a function completely before moving on.
    """

    def iter_args(self, spec: EngineSpec, depth: int, current_programs: List[Set[FunctionCall]]):
        """
        The naive version simply goes over everything exhaustively
        """

        yield from self.func_sequence[depth - 1].enumerate_exhaustive(spec, depth=depth)

    def iter_specs(self, inp_spec: EngineSpec, depth: int, programs: List[Set[FunctionCall]] = None):
        """
        The basic version evaluates all argument combinations, and maintains a cache
        of results which clubs together combinations that produce the same result.
        This cache is then iterated over at the end to produce the new specs for the next
        depth level. If this is the last function in the overall sequence, then the values
        are returned along with programs as final results.

        The checking of the result against the output is NOT done here. This will be the responsibility
        of the function-sequence engine. This is to afford the capability of pausing/restarting
        this argument engine.
        """
        func: BaseGenerator = self.func_sequence[depth - 1]
        if programs is None:
            programs = [None] * len(self.func_sequence)

        result_cache: ResultCache = ResultCache()
        for arg_vals, arg_annotations in self.iter_args_wrapper(inp_spec, depth, programs):
            result = self.execute(func, arg_vals, arg_annotations)
            if self.stats is not None:
                self.stats.num_cands_generated[depth] += 1

            if result is None:
                if self.stats is not None:
                    self.stats.num_cands_error[depth] += 1

                continue

            call: FunctionCall = FunctionCall(func, arg_vals, arg_annotations)
            result_cache.insert(result, call)

        if depth == len(self.func_sequence):
            for result, calls in result_cache.iter_results():
                programs[depth - 1] = calls
                if self.stats is not None:
                    self.stats.num_cands_propagated[depth] += 1

                yield result, programs
                programs[depth - 1] = None

        else:
            for result, calls in result_cache.iter_results():
                programs[depth - 1] = calls
                inp_spec.intermediates[depth - 1] = result
                inp_spec.depth = depth + 1
                if self.stats is not None:
                    self.stats.num_cands_propagated[depth] += 1

                yield from self.iter_specs(inp_spec, depth + 1, programs)
                programs[depth - 1] = None

            inp_spec.depth = depth


class BeamSearchEngine(BaseArgEngine):
    """
    This engine uses the neural-backed generators to predict only the most likely argument combinations.
    It does this by invoking the generator-wise beam-search enumeration rather than the vanilla exhaustive one
    (see generators/base.py for details on that.

    This is more similar to depth-first rather than breadth-first, as it does not wait to evaluate all the argument
    combinations before proceeding. The intuition being that if the neural network is good enough to get it right
    in the first few attempts, it's better to explore them fully.

    k is the beam-search parameter i.e. the width of the beam. We need to choose it wisely
    """

    def __init__(self, func_sequence: List[BaseGenerator], model_path: str, k: int = 10000,
                 cmd_args: ArgNamespace = None, stats: StatsCollector = None):
        super().__init__(func_sequence, cmd_args, stats)
        self.model_path = model_path
        self.beam_search_k = k

        self.model_store: Dict[str, Dict[str, RelGraphInterface]] = collections.defaultdict(dict)
        for func in self.func_sequence:
            func_model_path = model_path + '/' + func.qual_name

            for dsl_op_model_path in glob.glob(func_model_path + '/*'):
                if not os.path.exists(dsl_op_model_path + '/model_best.pickle'):
                    continue

                label = os.path.basename(dsl_op_model_path)
                self.model_store[func.qual_name][label] = RelGraphInterface.from_model_dir(dsl_op_model_path)

    def iter_args(self, spec: EngineSpec, depth: int, current_programs: List[Set[FunctionCall]]):
        """
        Use infer instead of enumerate_exhaustive
        """
        func_name = self.func_sequence[depth - 1].qual_name
        yield from itertools.islice(self.func_sequence[depth - 1].infer(spec,
                                                                        model_store=self.model_store[func_name],
                                                                        depth=depth, k=self.beam_search_k),
                                    self.beam_search_k)

    def iter_specs(self, inp_spec: EngineSpec, depth: int, programs: List[Set[FunctionCall]] = None):
        """
        The checking of the result against the output is NOT done here. This will be the responsibility
        of the function-sequence engine. This is to afford the capability of pausing/restarting
        this argument engine.
        """
        func: BaseGenerator = self.func_sequence[depth - 1]
        if programs is None:
            programs = [None] * len(self.func_sequence)

        for arg_vals, arg_annotations in self.iter_args_wrapper(inp_spec, depth, programs):
            result = self.execute(func, arg_vals, arg_annotations)
            if self.stats is not None:
                self.stats.num_cands_generated[depth] += 1

            if result is None:
                if self.stats is not None:
                    self.stats.num_cands_error[depth] += 1

                continue

            call: FunctionCall = FunctionCall(func, arg_vals, arg_annotations)

            if depth == len(self.func_sequence):
                programs[depth - 1] = [call]
                if self.stats is not None:
                    self.stats.num_cands_propagated[depth] += 1

                yield result, programs
                programs[depth - 1] = None

            else:
                programs[depth - 1] = [call]
                inp_spec.intermediates[depth - 1] = result
                inp_spec.depth = depth + 1
                if self.stats is not None:
                    self.stats.num_cands_propagated[depth] += 1

                yield from self.iter_specs(inp_spec, depth + 1, programs)
                programs[depth - 1] = None
                inp_spec.intermediates[depth - 1] = None

                inp_spec.depth = depth

    def close(self):
        for model_store in self.model_store.values():
            for model in model_store.values():
                model.close()


class ResultCache:
    class Result:
        def __init__(self, val: Any):
            self.val = val
            self.checker = Checker.get_checker(self.val)
            self.hash_val = Hasher.hash(self.val)

        def __eq__(self, o: 'ResultCache.Result') -> bool:
            return self.checker(self.val, o.val)

        def __ne__(self, o: object) -> bool:
            return not (self == o)

        def __hash__(self) -> int:
            return self.hash_val

    def __init__(self):
        self.cache: Dict[ResultCache.Result, Set[FunctionCall]] = collections.defaultdict(set)

    def insert(self, key: Any, val: FunctionCall):
        self.cache[ResultCache.Result(key)].add(val)

    def iter_results(self):
        for k, v in self.cache.items():
            yield k.val, v
