import argparse
import json

from typing import List

from ggnn.models.sparse.base import SparseGGNN
from ggnn.models.sparse.seq.static_rnn import GGNNSeqStaticRNN
from ggnn.utils import ParamsNamespace


def parse_args(cmd: List[str] = None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="Available Commands")
    #  Hack needed to make subparsers work
    subparsers.required = True
    subparsers.dest = 'subcommand'

    parser_common = argparse.ArgumentParser(add_help=False)

    parser_common.add_argument("--config", type=str, default=None,
                               help="File containing hyper-parameter configuration (JSON format)")
    parser_common.add_argument("--outdir", type=str, default=None,
                               help="Directory containing output")

    # =====================================
    #  Sparse GGNN
    # =====================================

    parser_sparse = subparsers.add_parser("sparse", parents=[parser_common],
                                          help="Sparse GGNN")

    parser_sparse.add_argument("mode", choices=["train", "test", "finalize"],
                               help="Mode of operation")

    parser_sparse.add_argument("--rnn", default=False, action="store_true",
                               help="Use an RNN-based final layer")

    parser_sparse.add_argument("--use-memory", default=False, action="store_true",
                               help="Store all processed graphs in memory. Fastest processing, but can easily"
                                    "run out of memory")

    parser_sparse.add_argument("--use-disk", default=False, action="store_true",
                               help="Use disk for storing processed graphs as opposed to computing them every time"
                                    "Speeds things up a lot but can take a lot of space")

    #  Training + Testing
    parser_sparse.add_argument("--restore-file", default=None, type=str,
                               help="File to restore weights from")

    #  Training
    parser_sparse.add_argument("--train", default=None, type=str,
                               help="Path to train file")
    parser_sparse.add_argument("--valid", default=None, type=str,
                               help="Path to validation file")
    parser_sparse.add_argument("--freeze-graph-model", default=False, action="store_true",
                               help="Freeze graph model components")
    parser_sparse.add_argument("--load-shuffle", default=False, action="store_true",
                               help="Shuffle data when loading. Useful when passing num-training-points")
    parser_sparse.add_argument("--num-epochs", default=3000, type=int,
                               help="Maximum number of epochs to run training for")
    parser_sparse.add_argument("--num-training-points", default=-1, type=int,
                               help="Number of training points to use. Default : -1 (all)")

    #  Testing
    parser_sparse.add_argument("--test", default=None, type=str,
                               help="Path to testing file")
    parser_sparse.add_argument("--model", default=None, type=str,
                               help="Path to model directory")
    parser_sparse.add_argument("--analysis", default=False, action="store_true",
                               help="Enable analysis. Can be more expensive")

    #  Testing with Analysis enabled
    parser_sparse.add_argument("--top-k", default=1, type=int,
                               help="Calculate top-k accuracy")
    parser_sparse.add_argument("--label-mapping", default=None, type=str,
                               help="Integer to Function (seq) label mapping file (YAML)")

    if cmd is None:
        return parser.parse_args()
    else:
        return parser.parse_args(cmd)


def run(cmd):
    cmd_args = parse_args(cmd)

    if cmd_args.subcommand == "sparse":
        params = ParamsNamespace()
        if cmd_args.config is not None:
            with open(cmd_args.config, 'r') as f:
                params.update(json.load(f))

        params.args = ParamsNamespace()
        params.args.update(cmd_args)

        if cmd_args.rnn:
            model = GGNNSeqStaticRNN.from_params(params)
        else:
            model = SparseGGNN.from_params(params)

        model.run()

