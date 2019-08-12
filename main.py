from argparse import ArgumentParser

import autopandas_v2.generators.cli
import autopandas_v2.evaluation.cli
import autopandas_v2.cloud.cli
from autopandas_v2.utils.cli import subcommand, ArgNamespace


def parse_args(root: ArgumentParser) -> ArgNamespace:
    @subcommand(root, cmd='synthesize', help='Run AutoPandas Synthesis', dest='subcommand')
    def cmd_synth(parser):
        parser.add_argument("path", help="Path to file")

    @subcommand(root, cmd='generators', help='Generator Utilities', dest='subcommand')
    def cmd_generators(parser):
        autopandas_v2.generators.cli.parse_args(parser)

    @subcommand(root, cmd='evaluate', help='Evaluation Utilities (Research-only)', dest='subcommand')
    def cmd_evaluation(parser):
        autopandas_v2.evaluation.cli.parse_args(parser)

    @subcommand(root, cmd='cloud', help='Cloud Utilities (Backup, Download etc.)', dest='subcommand')
    def cmd_cloud(parser):
        autopandas_v2.cloud.cli.parse_args(parser)

    return ArgNamespace.from_namespace(root.parse_args())


def run_console():
    parser = ArgumentParser()
    args = parse_args(parser)
    if args.subcommand == 'generators':
        autopandas_v2.generators.cli.run(args)

    elif args.subcommand == 'evaluate':
        autopandas_v2.evaluation.cli.run(args)

    elif args.subcommand == 'cloud':
        autopandas_v2.cloud.cli.run(args)
