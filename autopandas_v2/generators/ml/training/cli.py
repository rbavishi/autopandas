import glob
import json
import os
import pandas as pd
from argparse import ArgumentParser

from autopandas_v2.utils import logger
from autopandas_v2.utils.cli import ArgNamespace, subcommand


def parse_args(parser: ArgumentParser):
    parser_common = ArgumentParser(add_help=False)
    parser_common.add_argument("--device", type=str, default=None,
                               help="ID of Device (GPU) to use")

    @subcommand(parser, cmd='train-generators', help='Perform Training for Generators', dest='training_subcommand',
                inherit_from=[parser_common])
    def cmd_train_generators(parser):
        parser.add_argument("modeldir", type=str,
                            help="Path to the directory to save the model(s) in")

        parser.add_argument("--config", type=str, default=None,
                            help="File containing hyper-parameter configuration (JSON format)")
        parser.add_argument("--config-str", type=str, default=None,
                            help="String containing hyper-parameter configuration (JSON format)")
        parser.add_argument("--use-memory", default=False, action="store_true",
                            help="Store all processed graphs in memory. Fastest processing, but can easily"
                                 "run out of memory")

        parser.add_argument("--use-disk", default=False, action="store_true",
                            help="Use disk for storing processed graphs as opposed to computing them every time"
                                 "Speeds things up a lot but can take a lot of space")

        parser.add_argument("--train", default=None, type=str, required=True,
                            help="Path to train file")
        parser.add_argument("--valid", default=None, type=str, required=True,
                            help="Path to validation file")
        parser.add_argument("--restore-file", default=None, type=str,
                            help="File to restore weights from")
        parser.add_argument("--restore-params", default=None, type=str,
                            help="File to restore params from (pkl)")
        parser.add_argument("--freeze-graph-model", default=False, action="store_true",
                            help="Freeze graph model components")
        parser.add_argument("--load-shuffle", default=False, action="store_true",
                            help="Shuffle data when loading. Useful when passing num-training-points")
        parser.add_argument("--num-epochs", default=100, type=int,
                            help="Maximum number of epochs to run training for")
        parser.add_argument("--patience", default=25, type=int,
                            help="Maximum number of epochs to wait for validation accuracy to increase")
        parser.add_argument("--num-training-points", default=-1, type=int,
                            help="Number of training points to use. Default : -1 (all)")

        parser.add_argument("--include", nargs="+", type=str, default=None,
                            help="fn:identifier tuples to include in training list")
        parser.add_argument("--restore-if-exists", default=False, action='store_true',
                            help="If a model already exists, pick up training from there")
        parser.add_argument("--ignore-if-exists", default=False, action="store_true",
                            help="If the model exists, skip.")

    @subcommand(parser, cmd='train-functions', help='Perform Training for Predicting Functions',
                dest='training_subcommand', inherit_from=[parser_common])
    def cmd_train_functions(parser):
        parser.add_argument("modeldir", type=str,
                            help="Path to the directory to save the model in")

        parser.add_argument("--config", type=str, default=None,
                            help="File containing hyper-parameter configuration (JSON format)")
        parser.add_argument("--config-str", type=str, default=None,
                            help="String containing hyper-parameter configuration (JSON format)")
        parser.add_argument("--use-memory", default=False, action="store_true",
                            help="Store all processed graphs in memory. Fastest processing, but can easily"
                                 "run out of memory")

        parser.add_argument("--use-disk", default=False, action="store_true",
                            help="Use disk for storing processed graphs as opposed to computing them every time"
                                 "Speeds things up a lot but can take a lot of space")

        parser.add_argument("--train", default=None, type=str, required=True,
                            help="Path to train file")
        parser.add_argument("--valid", default=None, type=str, required=True,
                            help="Path to validation file")
        parser.add_argument("--restore-file", default=None, type=str,
                            help="File to restore weights from")
        parser.add_argument("--restore-params", default=None, type=str,
                            help="File to restore params from (pkl)")
        parser.add_argument("--freeze-graph-model", default=False, action="store_true",
                            help="Freeze graph model components")
        parser.add_argument("--load-shuffle", default=False, action="store_true",
                            help="Shuffle data when loading. Useful when passing num-training-points")
        parser.add_argument("--num-epochs", default=100, type=int,
                            help="Maximum number of epochs to run training for")
        parser.add_argument("--patience", default=25, type=int,
                            help="Maximum number of epochs to wait for validation accuracy to increase")
        parser.add_argument("--num-training-points", default=-1, type=int,
                            help="Number of training points to use. Default : -1 (all)")

    @subcommand(parser, cmd='train-next-func', help='Perform Training for Predicting Next Function',
                dest='training_subcommand', inherit_from=[parser_common])
    def cmd_train_functions(parser):
        parser.add_argument("modeldir", type=str,
                            help="Path to the directory to save the model in")

        parser.add_argument("--config", type=str, default=None,
                            help="File containing hyper-parameter configuration (JSON format)")
        parser.add_argument("--config-str", type=str, default=None,
                            help="String containing hyper-parameter configuration (JSON format)")
        parser.add_argument("--use-memory", default=False, action="store_true",
                            help="Store all processed graphs in memory. Fastest processing, but can easily"
                                 "run out of memory")

        parser.add_argument("--use-disk", default=False, action="store_true",
                            help="Use disk for storing processed graphs as opposed to computing them every time"
                                 "Speeds things up a lot but can take a lot of space")

        parser.add_argument("--train", default=None, type=str, required=True,
                            help="Path to train file")
        parser.add_argument("--valid", default=None, type=str, required=True,
                            help="Path to validation file")
        parser.add_argument("--restore-file", default=None, type=str,
                            help="File to restore weights from")
        parser.add_argument("--restore-params", default=None, type=str,
                            help="File to restore params from (pkl)")
        parser.add_argument("--freeze-graph-model", default=False, action="store_true",
                            help="Freeze graph model components")
        parser.add_argument("--load-shuffle", default=False, action="store_true",
                            help="Shuffle data when loading. Useful when passing num-training-points")
        parser.add_argument("--num-epochs", default=100, type=int,
                            help="Maximum number of epochs to run training for")
        parser.add_argument("--patience", default=25, type=int,
                            help="Maximum number of epochs to wait for validation accuracy to increase")
        parser.add_argument("--num-training-points", default=-1, type=int,
                            help="Number of training points to use. Default : -1 (all)")

    @subcommand(parser, cmd='analyze', help='Perform Analysis of Model', dest='training_subcommand')
    def cmd_analyze(parser):
        parser.add_argument("modeldir", type=str,
                            help="Path to the directory to save the model(s) in")
        parser.add_argument("outfile", type=str,
                            help="Path to output file")
        parser.add_argument("--config", type=str, default=None,
                            help="File containing hyper-parameter configuration (JSON format)")
        parser.add_argument("--test", default=None, type=str, required=True,
                            help="Path to test")
        parser.add_argument("--top-k", default=1, type=int, required=True,
                            help="Top-k")
        parser.add_argument("--include", nargs="+", type=str, default=None,
                            help="fn:identifier tuples to include in testing list")


def run_training_generators(cmd_args: ArgNamespace):
    #  Get the functions for which training data has been generated
    fnames = list(map(os.path.basename, glob.glob(cmd_args.train + '/*')))
    for fname in fnames:
        identifiers = list(map(os.path.basename, glob.glob(cmd_args.train + '/' + fname + '/*.pkl')))
        for identifier in identifiers:
            identifier = identifier[:-len(".pkl")]
            if cmd_args.include is not None and '{}:{}'.format(fname, identifier) not in cmd_args.include:
                continue

            logger.info("Performing training for {}:{}".format(fname, identifier))
            try:
                run_training_generators_helper(fname, identifier, cmd_args)
            except:
                continue


def run_training_generators_helper(fname: str, identifier: str, cmd_args: ArgNamespace):
    from autopandas_v2.generators.ml.networks.ggnn.ops.choice import ModelChoice
    from autopandas_v2.generators.ml.networks.ggnn.ops.chain import ModelChain
    from autopandas_v2.generators.ml.networks.ggnn.ops.select import ModelSelect
    from autopandas_v2.generators.ml.networks.ggnn.ops.subsets import ModelSubsets
    from autopandas_v2.generators.ml.networks.ggnn.ops.orderedsubsets import ModelOrderedSubsets
    from autopandas_v2.ml.networks.ggnn.utils import ParamsNamespace

    train_path = '{}/{}/{}.pkl'.format(cmd_args.train, fname, identifier)
    valid_path = '{}/{}/{}.pkl'.format(cmd_args.valid, fname, identifier)
    model_path = '{}/{}/{}'.format(cmd_args.modeldir, fname, identifier)
    if not os.path.exists(train_path):
        raise Exception("Training data path {} does not exist".format(train_path))

    if not os.path.exists(valid_path):
        raise Exception("Validation data path {} does not exist".format(valid_path))

    if cmd_args.ignore_if_exists and os.path.exists(model_path + '/model_best.pickle'):
        logger.info("Skipping training for {}:{} as model already exists".format(fname, identifier))
        return

    os.system('mkdir -p ' + model_path)
    ggnn_args = ArgNamespace.from_namespace(cmd_args)
    ggnn_args.train = train_path
    ggnn_args.valid = valid_path
    ggnn_args.outdir = model_path
    ggnn_args.mode = 'train'

    if cmd_args.restore_if_exists and os.path.exists(model_path + '/model_best.pickle'):
        ggnn_args.restore = model_path + '/model_best.pickle'

    params = ParamsNamespace()

    if cmd_args.config is not None:
        with open(cmd_args.config, 'r') as f:
            params.update(json.load(f))

    if cmd_args.config_str is not None:
        params.update(json.loads(cmd_args.config_str))

    params.args = ParamsNamespace()
    params.args.update(ggnn_args)
    params.use_directed_edges = True

    if identifier.startswith("choice"):
        model = ModelChoice.from_params(params)

    elif identifier.startswith("chain"):
        model = ModelChain.from_params(params)

    elif identifier.startswith("select"):
        model = ModelSelect.from_params(params)

    elif identifier.startswith("subsets"):
        model = ModelSubsets.from_params(params)

    elif identifier.startswith("orderedsubsets"):
        model = ModelOrderedSubsets.from_params(params)

    else:
        raise NotImplementedError("Model not defined for operator {}".format(identifier.split('_')[0]))

    model.run()


def run_training_functions(cmd_args: ArgNamespace):
    from autopandas_v2.ml.networks.ggnn.models.sparse.seq.static_rnn import GGNNSeqStaticRNN
    from autopandas_v2.ml.networks.ggnn.utils import ParamsNamespace

    train_path = cmd_args.train
    valid_path = cmd_args.valid
    model_path = cmd_args.modeldir

    if not os.path.exists(model_path):
        os.system('mkdir -p ' + model_path)

    ggnn_args = ArgNamespace.from_namespace(cmd_args)
    ggnn_args.train = train_path
    ggnn_args.valid = valid_path
    ggnn_args.outdir = model_path
    ggnn_args.mode = 'train'

    params = ParamsNamespace()

    if cmd_args.config is not None:
        with open(cmd_args.config, 'r') as f:
            params.update(json.load(f))

    if cmd_args.config_str is not None:
        params.update(json.loads(cmd_args.config_str))

    params.args = ParamsNamespace()
    params.args.update(ggnn_args)
    params.use_directed_edges = True

    model = GGNNSeqStaticRNN.from_params(params)
    model.run()


def run_training_next_function(cmd_args: ArgNamespace):
    from autopandas_v2.ml.networks.ggnn.models.sparse.base import SparseGGNN
    from autopandas_v2.ml.networks.ggnn.utils import ParamsNamespace

    train_path = cmd_args.train
    valid_path = cmd_args.valid
    model_path = cmd_args.modeldir

    if not os.path.exists(model_path):
        os.system('mkdir -p ' + model_path)

    ggnn_args = ArgNamespace.from_namespace(cmd_args)
    ggnn_args.train = train_path
    ggnn_args.valid = valid_path
    ggnn_args.outdir = model_path
    ggnn_args.mode = 'train'

    params = ParamsNamespace()

    if cmd_args.config is not None:
        with open(cmd_args.config, 'r') as f:
            params.update(json.load(f))

    if cmd_args.config_str is not None:
        params.update(json.loads(cmd_args.config_str))

    params.args = ParamsNamespace()
    params.args.update(ggnn_args)
    params.use_directed_edges = True

    model = SparseGGNN.from_params(params)
    model.run()


def run_analysis(cmd_args: ArgNamespace):
    #  Get the functions for which training data has been generated
    fnames = list(map(os.path.basename, glob.glob(cmd_args.test + '/*')))
    results = []
    for fname in fnames:
        identifiers = list(map(os.path.basename, glob.glob(cmd_args.test + '/' + fname + '/*.pkl')))
        for identifier in identifiers:
            identifier = identifier[:-len(".pkl")]
            if cmd_args.include is not None and '{}:{}'.format(fname, identifier) not in cmd_args.include:
                continue

            logger.info("Performing Analysis for {}:{}".format(fname, identifier))
            result = run_analysis_helper(fname, identifier, cmd_args)
            result['Name'] = '{}:{}'.format(fname, identifier)
            results.append(result)

    with open(cmd_args.outfile, 'w') as f:
        print(pd.DataFrame(results).to_csv(), file=f)


def run_analysis_helper(fname: str, identifier: str, cmd_args: ArgNamespace):
    from autopandas_v2.generators.ml.networks.ggnn.ops.choice import ModelChoice
    from autopandas_v2.generators.ml.networks.ggnn.ops.chain import ModelChain
    from autopandas_v2.generators.ml.networks.ggnn.ops.select import ModelSelect
    from autopandas_v2.generators.ml.networks.ggnn.ops.subsets import ModelSubsets
    from autopandas_v2.generators.ml.networks.ggnn.ops.orderedsubsets import ModelOrderedSubsets
    from autopandas_v2.ml.networks.ggnn.utils import ParamsNamespace

    test_path = '{}/{}/{}.pkl'.format(cmd_args.test, fname, identifier)
    model_path = '{}/{}/{}'.format(cmd_args.modeldir, fname, identifier)
    if not os.path.exists(test_path):
        raise Exception("Test data path {} does not exist".format(test_path))

    os.system('mkdir -p ' + model_path)
    ggnn_args = ArgNamespace.from_namespace(cmd_args)
    ggnn_args.test = test_path
    ggnn_args.outdir = model_path
    ggnn_args.mode = 'train'
    ggnn_args.model = model_path

    params = ParamsNamespace()

    if cmd_args.config is not None:
        with open(cmd_args.config, 'r') as f:
            params.update(json.load(f))

    params.args = ParamsNamespace()
    params.args.update(ggnn_args)
    params.use_directed_edges = True

    if identifier.startswith("choice"):
        model = ModelChoice.from_params(params)

    elif identifier.startswith("chain"):
        model = ModelChain.from_params(params)

    elif identifier.startswith("select"):
        model = ModelSelect.from_params(params)

    elif identifier.startswith("subsets"):
        model = ModelSubsets.from_params(params)

    elif identifier.startswith("orderedsubsets"):
        model = ModelOrderedSubsets.from_params(params)

    else:
        raise NotImplementedError("Model not defined for operator {}".format(identifier.split('_')[0]))

    return model.run_analysis(test_path)


def run(cmd_args: ArgNamespace):
    if cmd_args.training_subcommand == 'train-generators':
        run_training_generators(cmd_args)

    elif cmd_args.training_subcommand == 'train-functions':
        run_training_functions(cmd_args)

    elif cmd_args.training_subcommand == 'train-next-func':
        run_training_next_function(cmd_args)

    elif cmd_args.training_subcommand == 'analyze':
        run_analysis(cmd_args)
