import ast
import inspect
from argparse import ArgumentParser

import autopandas_v2.generators.ml.traindata.cli
import autopandas_v2.generators.ml.training.cli
from autopandas_v2.generators.compilation.compiler import compile_gens_from_module, compile_gens_to_module
from autopandas_v2.generators.ml.traindata.compiler import compile_randomized_gens
from autopandas_v2.utils import astutils
from autopandas_v2.utils.cli import subcommand, ArgNamespace


def parse_args(parser: ArgumentParser):
    @subcommand(parser, cmd='compile', help='Compile Generators', dest='gen_subcommand')
    def cmd_compile(parser):
        pass

    @subcommand(parser, cmd='compile-randomized', help='Compile Generators', dest='gen_subcommand')
    def cmd_compile_randomized(parser):
        pass

    @subcommand(parser, cmd='training-data', help='Generate Training Data', dest='gen_subcommand')
    def cmd_training_data(parser):
        autopandas_v2.generators.ml.traindata.cli.parse_args(parser)

    @subcommand(parser, cmd='training', help='Perform Training', dest='gen_subcommand')
    def cmd_train_generators(parser):
        autopandas_v2.generators.ml.training.cli.parse_args(parser)


def run_compile(cmd_args: ArgNamespace):
    from autopandas_v2.generators import specs
    path = inspect.getfile(specs)
    spec_ast: ast.Module = astutils.parse_file(path)
    spec_ast = compile_gens_to_module(target_ast=spec_ast,
                                      gens=compile_gens_from_module(spec_ast, cmd_args), cmd_args=cmd_args)

    new_path = "/".join(path.split('/')[:-1]) + '/specs_compiled.py'
    with open(new_path, 'w') as f:
        print(astutils.to_source(spec_ast), file=f)


def run_compile_randomized(cmd_args: ArgNamespace):
    from autopandas_v2.generators import specs as orig_specs
    from autopandas_v2.generators.ml.traindata import specs
    orig_path = inspect.getfile(orig_specs)
    path = inspect.getfile(specs)
    orig_spec_ast: ast.Module = astutils.parse_file(orig_path)
    spec_ast: ast.Module = astutils.parse_file(path)
    spec_ast = compile_gens_to_module(target_ast=spec_ast,
                                      gens=compile_randomized_gens(spec_ast, orig_spec_ast, cmd_args),
                                      cmd_args=cmd_args)

    new_path = "/".join(path.split('/')[:-1]) + '/specs_compiled.py'
    with open(new_path, 'w') as f:
        print(astutils.to_source(spec_ast), file=f)


def run(cmd_args: ArgNamespace):
    if cmd_args.gen_subcommand == 'compile':
        run_compile(cmd_args)

    elif cmd_args.gen_subcommand == 'compile-randomized':
        run_compile_randomized(cmd_args)

    elif cmd_args.gen_subcommand == 'training-data':
        autopandas_v2.generators.ml.traindata.cli.run(cmd_args)

    elif cmd_args.gen_subcommand == 'training':
        autopandas_v2.generators.ml.training.cli.run(cmd_args)
