import logging

import pandas as pd
from argparse import ArgumentParser
from typing import Pattern, Type, Dict
from concurrent.futures import TimeoutError

from autopandas_v2.evaluation.benchmarks.base import Benchmark
from autopandas_v2.evaluation.benchmarks.utils import discover_benchmarks
from autopandas_v2.evaluation.evaluators import GeneratorModelEvaluator, NeuralSynthesisEvaluator, \
    FunctionModelEvaluator
from autopandas_v2.utils import logger
from autopandas_v2.utils.cli import ArgNamespace, subcommand
import re

from autopandas_v2.utils.misc import call_with_timeout


def parse_args(parser: ArgumentParser):
    parser_common = ArgumentParser(add_help=False)
    parser_common.add_argument("path_regex", type=str,
                               help="Path to benchmark. Should be a qualified name such as 'FooBenchmarks.bar'. "
                                    "Supports regexes. For example '*.bar' will match all paths ending with .bar")

    @subcommand(parser, cmd='generator-models', help='Evaluate Generator Models', dest='eval_subcommand',
                inherit_from=[parser_common])
    def cmd_generator_models(parser: ArgumentParser):
        parser.add_argument("model_dir", type=str,
                            help="Path to model")
        parser.add_argument("outfile", type=str,
                            help="Path to CSV file to hold the results")

    @subcommand(parser, cmd='function-model', help='Evaluate Function Seq Models', dest='eval_subcommand',
                inherit_from=[parser_common])
    def cmd_generator_models(parser: ArgumentParser):
        parser.add_argument("--use-old-featurization", default=False,
                            action="store_true",
                            help="Use old featurization")
        parser.add_argument("--top-k", type=int, default=10,
                            help="Top-k")
        parser.add_argument("model_dir", type=str,
                            help="Path to model")
        parser.add_argument("outfile", type=str,
                            help="Path to CSV file to hold the results")

    @subcommand(parser, cmd='synthesis', help='Evaluate the Full Synthesis Engine', dest='eval_subcommand',
                inherit_from=[parser_common])
    def cmd_synthesis(parser: ArgumentParser):
        parser.add_argument("--use-old-featurization", default=False,
                            action="store_true",
                            help="Use old featurization")
        parser.add_argument("--top-k-function", type=int, default=10,
                            help="Top-k for functions")
        parser.add_argument("--top-k-args", type=int, default=10,
                            help="Top-k for arguments")
        parser.add_argument("--timeout", type=int, default=600,
                            help="Timeout in seconds")
        parser.add_argument("arg_model_dir", type=str,
                            help="Path to Arguments model")
        parser.add_argument("function_model_dir", type=str,
                            help="Path to Function-Seq model")
        parser.add_argument("outfile", type=str,
                            help="Path to CSV file to hold the results")


def run_generator_model_eval(cmd_args: ArgNamespace):
    benchmarks: Dict[str, Type[Benchmark]] = discover_benchmarks()
    path_matcher: Pattern = re.compile(cmd_args.path_regex)
    results = []
    for qual_name, benchmark_cls in benchmarks.items():
        if not path_matcher.match(qual_name):
            continue

        try:
            logger.info("Running benchmark {}".format(qual_name))
            benchmark = benchmark_cls()
            evaluator = GeneratorModelEvaluator(benchmark, cmd_args)
            results.append(evaluator.run(qual_name))
            logger.info("Result for {} : {}".format(qual_name, results[-1]))

        except Exception as e:
            logger.warn("Failed for {}".format(qual_name))
            logging.exception(e)

    results = pd.DataFrame(results)
    print(results)
    with open(cmd_args.outfile, 'w') as f:
        results.to_csv(f)


def run_function_model_eval(cmd_args: ArgNamespace):
    benchmarks: Dict[str, Type[Benchmark]] = discover_benchmarks()
    path_matcher: Pattern = re.compile(cmd_args.path_regex)
    results = []
    for qual_name, benchmark_cls in benchmarks.items():
        if not path_matcher.match(qual_name):
            continue

        try:
            logger.info("Running benchmark {}".format(qual_name))
            benchmark = benchmark_cls()
            evaluator = FunctionModelEvaluator(benchmark, cmd_args)
            results.append(evaluator.run(qual_name))
            logger.info("Result for {} : {}".format(qual_name, results[-1]))

        except Exception as e:
            logger.warn("Failed for {}".format(qual_name))
            logging.exception(e)

    results = pd.DataFrame(results)
    print(results)
    with open(cmd_args.outfile, 'w') as f:
        results.to_csv(f)


def run_synthesis_for_benchmark(qual_name: str, cmd_args: ArgNamespace):
    benchmarks: Dict[str, Type[Benchmark]] = discover_benchmarks()
    benchmark_cls = benchmarks[qual_name]
    benchmark = benchmark_cls()
    evaluator = NeuralSynthesisEvaluator(benchmark, cmd_args)
    return evaluator.run(qual_name)


def run_synthesis_eval(cmd_args):
    benchmarks: Dict[str, Type[Benchmark]] = discover_benchmarks()
    path_matcher: Pattern = re.compile(cmd_args.path_regex)
    results = []

    for qual_name, benchmark_cls in benchmarks.items():
        if not path_matcher.match(qual_name):
            continue

        try:
            logger.info("Running benchmark {}".format(qual_name))
            result = call_with_timeout(run_synthesis_for_benchmark,
                                       qual_name, cmd_args, timeout=cmd_args.timeout)
            results.append(result)
            logger.info("Result for {} : {}".format(qual_name, results[-1]))

        except TimeoutError:
            logger.info("Timed out for {}".format(qual_name))
            result = {
                'benchmark': qual_name,
                'num_seqs_explored': {},
                'num_candidates_generated': {},
                'solution_found': False,
                'time': cmd_args.timeout
            }

            results.append(result)

        except Exception as e:
            logger.warn("Failed for {}".format(qual_name))
            logging.exception(e)

    results = pd.DataFrame(results)
    print(results)
    with open(cmd_args.outfile, 'w') as f:
        results.to_csv(f)


def run(cmd_args: ArgNamespace):
    if cmd_args.eval_subcommand == 'generator-models':
        run_generator_model_eval(cmd_args)

    elif cmd_args.eval_subcommand == 'function-model':
        run_function_model_eval(cmd_args)

    elif cmd_args.eval_subcommand == 'synthesis':
        run_synthesis_eval(cmd_args)
