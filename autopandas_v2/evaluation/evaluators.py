import time

from autopandas_v2.evaluation.benchmarks.base import Benchmark
from autopandas_v2.iospecs import IOSpec
from autopandas_v2.ml.inference.interfaces import RelGraphInterface
from autopandas_v2.synthesis.search.engines.functions import BFSEngine, NeuralEngine
from autopandas_v2.utils.cli import ArgNamespace


class GeneratorModelEvaluator:
    def __init__(self, benchmark: Benchmark, cmd_args: ArgNamespace):
        self.benchmark = benchmark
        self.cmd_args = cmd_args

        self.model_dir = self.cmd_args.model_dir

    def run(self, qual_name: str):
        inputs, output, funcs, seqs = self.benchmark.unwrap()

        iospec: IOSpec = IOSpec(inputs, output)
        iospec.funcs = funcs
        iospec.seqs = seqs

        engine: BFSEngine = BFSEngine(iospec)
        engine.max_depth = max(len(i) for i in seqs)
        engine.silent = True
        engine.stop_first_solution = True
        engine.use_spec_funcs = True
        engine.use_spec_seqs = True
        engine.argument_engine = 'beam-search'
        engine.arg_model_dir = self.model_dir

        solution_found = engine.search()
        return {
            'benchmark': qual_name,
            'num_candidates_generated': dict(engine.stats.num_cands_generated),
            'solution_found': solution_found
        }


class FunctionModelEvaluator:
    def __init__(self, benchmark: Benchmark, cmd_args: ArgNamespace):
        self.benchmark = benchmark
        self.cmd_args = cmd_args
        self.model_dir = self.cmd_args.model_dir
        self.top_k = self.cmd_args.top_k

    def run(self, qual_name: str):
        inputs, output, funcs, seqs = self.benchmark.unwrap()

        iospec: IOSpec = IOSpec(inputs, output)
        iospec.funcs = funcs
        iospec.seqs = seqs

        model = RelGraphInterface.from_model_dir(self.model_dir)
        if self.cmd_args.use_old_featurization:
            from autopandas_v2.ml.featurization_old.featurizer import RelationGraph
            from autopandas_v2.ml.featurization_old.options import GraphOptions
        else:
            from autopandas_v2.ml.featurization.featurizer import RelationGraph
            from autopandas_v2.ml.featurization.options import GraphOptions

        options = GraphOptions()
        graph: RelationGraph = RelationGraph(options)
        graph.from_input_output(inputs, output)
        encoding = graph.get_encoding(get_mapping=False)

        str_seqs, probs = list(zip(*model.predict_graphs([encoding], top_k=self.top_k)[0]))
        ground_truth = ':'.join(funcs[i] for i in seqs[0])

        pos = -1
        for idx, pred in enumerate(str_seqs, 1):
            if pred == ground_truth:
                pos = idx
                break

        return {
            'benchmark': qual_name,
            'ground_truth': ground_truth,
            'rank': pos
        }


class NeuralSynthesisEvaluator:
    def __init__(self, benchmark: Benchmark, cmd_args: ArgNamespace):
        self.benchmark = benchmark
        self.cmd_args = cmd_args
        self.function_model_dir = self.cmd_args.function_model_dir
        self.arg_model_dir = self.cmd_args.arg_model_dir

    def run(self, qual_name: str):
        inputs, output, funcs, seqs = self.benchmark.unwrap()

        iospec: IOSpec = IOSpec(inputs, output)
        iospec.funcs = funcs
        iospec.seqs = seqs

        engine: NeuralEngine = NeuralEngine(iospec, self.function_model_dir, top_k=self.cmd_args.top_k_function)
        engine.max_depth = max(len(i) for i in seqs)
        engine.stop_first_solution = True
        engine.use_spec_funcs = True
        engine.use_spec_seqs = True
        engine.argument_engine = 'beam-search'
        engine.arg_model_dir = self.arg_model_dir
        engine.arg_top_k = self.cmd_args.top_k_args
        engine.use_old_featurization = self.cmd_args.use_old_featurization

        start_time = time.time()
        solution_found = engine.search()
        return {
            'benchmark': qual_name,
            'num_seqs_explored': engine.stats.num_seqs_explored,
            'num_candidates_generated': dict(engine.stats.num_cands_generated),
            'solution_found': solution_found,
            'time': time.time() - start_time
        }
