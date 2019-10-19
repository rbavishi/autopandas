import itertools
import random
from typing import Set, List, Any, Dict

import pandas as pd
from autopandas_v2.generators.base import BaseGenerator
from autopandas_v2.generators.dsl.values import Fetcher
from autopandas_v2.generators.ml.traindata.exploration.iospecs import ExplorationSpec
from autopandas_v2.generators.trackers import OpTracker
from autopandas_v2.synthesis.search.engines.arguments import BreadthFirstEngine
from autopandas_v2.synthesis.search.results.programs import FunctionCall
from autopandas_v2.synthesis.search.stats.collectors import StatsCollector
from autopandas_v2.utils.checker import Checker
from autopandas_v2.utils.cli import ArgNamespace


class RandArgEngine(BreadthFirstEngine):
    """
    This engine randomly picks an argument combination to use
    """

    def __init__(self, func_sequence: List[BaseGenerator], cmd_args: ArgNamespace = None, stats: StatsCollector = None):
        super().__init__(func_sequence, cmd_args, stats)
        self.inp_cache = []

    def push_arg_combination(self, func: BaseGenerator, inp_spec: ExplorationSpec, arg_vals: Dict[str, Any],
                             arg_annotations: Dict[str, Dict[str, Any]], tracker: OpTracker):
        self.inp_cache.append(inp_spec.inputs[:])
        new_tracker: OpTracker = OpTracker()
        for k, v in tracker.record.items():
            if k.startswith("rext_"):
                new_inp = v['val']
                new_tracker.record[k[1:]] = {'source': 'inps', 'idx': len(inp_spec.inputs)}
                arg_name = v['arg_name']
                fetcher = Fetcher(new_inp, source='inps', idx=len(inp_spec.inputs))
                _, new_annotation = func.process_val(fetcher)
                annotation = arg_annotations[arg_name]
                annotation.update(new_annotation)
                inp_spec.inputs.append(new_inp)

            else:
                new_tracker.record[k] = v

        tracker.record.clear()
        tracker.record.update(new_tracker.record)

    def pop_arg_combination(self, inp_spec: ExplorationSpec):
        inp_spec.inputs = self.inp_cache.pop()

    def iter_args(self, spec: ExplorationSpec, depth: int, current_programs: List[Set[FunctionCall]]):
        yield from self.func_sequence[depth - 1].generate_training_data(spec, depth=depth)

    def iter_specs(self, inp_spec: ExplorationSpec, depth: int, programs: List[Set[FunctionCall]] = None):
        func: BaseGenerator = self.func_sequence[depth - 1]
        if programs is None:
            programs = [None] * len(self.func_sequence)

        max_exploration = self.cmd_args.get('max_exploration', 500)
        max_arg_trials = self.cmd_args.get('max_arg_trials', 500)

        arg_cands = []
        for arg_vals, arg_annotations, tracker in itertools.islice(self.iter_args_wrapper(inp_spec, depth, programs),
                                                                   max_exploration):
            arg_cands.append((arg_vals.copy(), arg_annotations.copy(), tracker))

        #  Since the ops already try to return candidates in a uniform manner across multiple invocations,
        #  shuffling here would actually be harmful as it can introduce class imbalance, especially when
        #  dsl operators like Subsets and OrderedSubsets are involved
        # random.shuffle(arg_cands)

        for arg_vals, arg_annotations, tracker in itertools.islice(arg_cands, max_arg_trials):
            result = self.execute(func, arg_vals, arg_annotations)

            if result is None:
                continue

            self.push_arg_combination(func, inp_spec, arg_vals, arg_annotations, tracker)
            #  We only consider results that are not equal to an already provided input/intermediate
            for inp in itertools.chain(inp_spec.inputs, inp_spec.intermediates):
                if inp is None:
                    continue

                if Checker.check(inp, result):
                    break

            else:
                #  We also don't want extremely large dataframes or empty dataframes
                if isinstance(result, pd.DataFrame):
                    if 0 in result.shape:
                        self.pop_arg_combination(inp_spec)
                        continue

                    if result.shape[0] > 25 or result.shape[1] > 25:
                        self.pop_arg_combination(inp_spec)
                        continue

                #  No checks were falsified, so we're good
                call: FunctionCall = FunctionCall(func, arg_vals, arg_annotations)
                programs[depth - 1] = {call}
                inp_spec.tracking[depth - 1] = tracker

                if depth == len(self.func_sequence):
                    yield result, programs
                    return

                else:
                    inp_spec.intermediates[depth - 1] = result
                    inp_spec.depth = depth + 1
                    yield from self.iter_specs(inp_spec, depth + 1, programs)
                    inp_spec.depth = depth
                    inp_spec.intermediates[depth - 1] = None

                programs[depth - 1] = None
                inp_spec.tracking[depth - 1] = None

            self.pop_arg_combination(inp_spec)
