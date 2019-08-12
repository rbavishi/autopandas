import os
import warnings

import pandas as pd
from argparse import ArgumentParser

from autopandas_v2.generators.ml.traindata.generation import ArgDataGenerator, RawDataGenerator, \
    FunctionSeqDataGenerator, NextFunctionDataGenerator
from autopandas_v2.utils import logger
from autopandas_v2.utils.cli import subcommand, ArgNamespace


def parse_args(parser: ArgumentParser):
    @subcommand(parser, cmd='raw', help='Generate Raw Training Data',
                dest='train_data_subcommand')
    def cmd_training_data_raw(parser):
        parser.add_argument("--debug", default=False, action="store_true",
                            help="Debug-level logging")
        parser.add_argument("--processes", type=int, default=1,
                            help="Number of processes to use")
        parser.add_argument("--chunksize", type=int, default=1000,
                            help="Pebble Chunk Size. Only touch this if you understand the source")
        parser.add_argument("--task-timeout", type=int, default=10,
                            help="Timeout for a datapoint generation task (for multiprocessing). "
                                 "Useful for avoiding enumeration-gone-wrong cases, where something "
                                 "is taking a long time or is consuming too many resources")

        parser.add_argument("--max-exploration", type=int, default=20,
                            help="Maximum number of arg combinations to explore before moving on")
        parser.add_argument("--max-arg-trials", type=int, default=20,
                            help="Maximum number of argument trials to actually execute")
        parser.add_argument("--max-seq-trials", type=int, default=30,
                            help="Maximum number of trials to generate data for a single sequence")

        parser.add_argument("--blacklist-threshold", type=int, default=100,
                            help="Maximum number of trials for a sequence before giving up forever. "
                                 "Use -1 to have no threshold")

        parser.add_argument("--min-depth", type=int, default=1,
                            help="Minimum length of sequences allowed")
        parser.add_argument("--max-depth", type=int, default=1,
                            help="Maximum length of sequences allowed")

        parser.add_argument("--num-training-points", type=int, default=1,
                            help="Number of training examples to generate")
        parser.add_argument("--sequences", type=str, default=None, required=True,
                            help="Path to pickle file containing sequences that the generator "
                                 "can stick to while generating data. Helps in generating random data "
                                 "that mimics actual usage of the API in the wild. "
                                 "Can also be a comma plus colon-separated string containing functions to use. "
                                 "For example - df.pivot:df.index,df.columns:df.T allows the sequences "
                                 "(df.pivot, df.index) and (df.columns, df.T)")

        parser.add_argument("--no-repeat", default=False, action="store_true",
                            help="Produce only 1 training example for each sequence")
        parser.add_argument("--append", default=False, action="store_true",
                            help="Whether to append to an already existing dataset")
        parser.add_argument("outfile", help="Path to output file")

    @subcommand(parser, cmd='generators', help='Generate Training Data for Smart Generators',
                dest='train_data_subcommand')
    def cmd_training_data_generators(parser):
        parser.add_argument("--debug", default=False, action="store_true",
                            help="Debug-level logging")
        parser.add_argument("-f", "--force", action="store_true", default=False,
                            help="Force recreation of outdir if it exists")
        parser.add_argument("--append-arg-level", action="store_true", default=False,
                            help="Append training-data at argument-operator level "
                                 "instead of overwriting by default")

        parser.add_argument("--processes", type=int, default=1,
                            help="Number of processes to use")
        parser.add_argument("--chunksize", type=int, default=100,
                            help="Pebble Chunk Size. Only touch this if you understand the source")
        parser.add_argument("--task-timeout", type=int, default=10,
                            help="Timeout for a datapoint generation task (for multiprocessing). "
                                 "Useful for avoiding enumeration-gone-wrong cases, where something "
                                 "is taking a long time or is consuming too many resources")

        parser.add_argument("raw_data_path", type=str,
                            help="Path to pkl containing the raw I/O example data")
        parser.add_argument("outdir", type=str,
                            help="Path to output directory where the generated data is to be stored")

    @subcommand(parser, cmd='function-seq', help='Generate Training Data for Function Sequence Prediction',
                dest='train_data_subcommand')
    def cmd_training_data_generators(parser):
        parser.add_argument("--debug", default=False, action="store_true",
                            help="Debug-level logging")
        parser.add_argument("--append", action="store_true", default=False,
                            help="Append training-data to the existing dataset represented by outfile"
                                 "instead of overwriting by default")

        parser.add_argument("--processes", type=int, default=1,
                            help="Number of processes to use")
        parser.add_argument("--chunksize", type=int, default=100,
                            help="Pebble Chunk Size. Only touch this if you understand the source")
        parser.add_argument("--task-timeout", type=int, default=10,
                            help="Timeout for a datapoint generation task (for multiprocessing). "
                                 "Useful for avoiding enumeration-gone-wrong cases, where something "
                                 "is taking a long time or is consuming too many resources")

        parser.add_argument("raw_data_path", type=str,
                            help="Path to pkl containing the raw I/O example data")
        parser.add_argument("outfile", type=str,
                            help="Path to output file where the generated data is to be stored")

    @subcommand(parser, cmd='next-func', help='Generate Training Data for Next Function Prediction',
                dest='train_data_subcommand')
    def cmd_training_data_generators(parser):
        parser.add_argument("--debug", default=False, action="store_true",
                            help="Debug-level logging")
        parser.add_argument("--append", action="store_true", default=False,
                            help="Append training-data to the existing dataset represented by outfile"
                                 "instead of overwriting by default")

        parser.add_argument("--processes", type=int, default=1,
                            help="Number of processes to use")
        parser.add_argument("--chunksize", type=int, default=100,
                            help="Pebble Chunk Size. Only touch this if you understand the source")
        parser.add_argument("--task-timeout", type=int, default=10,
                            help="Timeout for a datapoint generation task (for multiprocessing). "
                                 "Useful for avoiding enumeration-gone-wrong cases, where something "
                                 "is taking a long time or is consuming too many resources")

        parser.add_argument("raw_data_path", type=str,
                            help="Path to pkl containing the raw I/O example data")
        parser.add_argument("outfile", type=str,
                            help="Path to output file where the generated data is to be stored")


def run_raw_training_data_generation(cmd_args: ArgNamespace):
    with warnings.catch_warnings():
        #  Just STFU
        warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
        warnings.simplefilter(action='ignore', category=UserWarning)
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        warnings.simplefilter(action='ignore', category=FutureWarning)

        RawDataGenerator(cmd_args).generate()


def run_smartgen_training_data_generation(cmd_args: ArgNamespace):
    ArgDataGenerator(cmd_args).generate()


def run_func_seq_training_data_generation(cmd_args: ArgNamespace):
    FunctionSeqDataGenerator(cmd_args).generate()


def run_next_func_training_data_generation(cmd_args: ArgNamespace):
    NextFunctionDataGenerator(cmd_args).generate()


def run(cmd_args: ArgNamespace):
    if cmd_args.train_data_subcommand == 'raw':
        run_raw_training_data_generation(cmd_args)

    elif cmd_args.train_data_subcommand == 'generators':
        run_smartgen_training_data_generation(cmd_args)

    elif cmd_args.train_data_subcommand == 'function-seq':
        run_func_seq_training_data_generation(cmd_args)

    elif cmd_args.train_data_subcommand == 'next-func':
        run_next_func_training_data_generation(cmd_args)
