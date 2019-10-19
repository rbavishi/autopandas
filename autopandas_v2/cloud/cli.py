import os
import pandas as pd
from argparse import ArgumentParser

from autopandas_v2.cloud.utils import GDriveRunner
from autopandas_v2.utils import logger
from autopandas_v2.utils.cli import subcommand, ArgNamespace


def parse_args(parser: ArgumentParser):
    parser_common = ArgumentParser(add_help=False)
    parser_common.add_argument("--max-gdrive-retries", type=int, default=20,
                               help="Maximum number of attempts to get through the rate-limiting")

    @subcommand(parser, cmd='upload', help='Upload data to cloud backup',
                dest='train_data_subcommand', inherit_from=[parser_common])
    def cmd_training_data_upload_data(parser):
        parser.add_argument("--parent", default=None, type=str,
                            help='A qualified path such as /Data/Raw. '
                                 'Only one of parent and parent-id should be provided')
        parser.add_argument("--parent-id", default=None, type=str,
                            help="Parent ID like 1hlg3OcR3uPiqJQRVPuLqJeeeB6R4ESyY. "
                                 "Only one of parent and parent-id should be provided")
        parser.add_argument("path", type=str, help="Path to file")
        parser.add_argument("--desc", type=str, default=None,
                            help="Comments if any")

    @subcommand(parser, cmd='download', help='Download data from cloud backup',
                dest='train_data_subcommand', inherit_from=[parser_common])
    def cmd_training_data_download_data(parser):
        parser.add_argument("--path", default=None, type=str, help="Path to file. Can be qualified.")
        parser.add_argument("--path-id", default=None, type=str,
                            help="The ID of the file like 1hlg3OcR3uPiqJQRVPuLqJeeeB6R4ESyY. "
                                 "Only one of parent and parent-id should be provided")
        parser.add_argument("--outdir", type=str, default='.',
                            help="Directory to store the output in")


def run_upload(cmd_args: ArgNamespace):
    home_dir = os.path.expanduser("~")
    gdrive_bin = home_dir + '/gdrive'
    if not os.path.exists(gdrive_bin):
        logger.err("Could not find gdrive at {home_dir}. "
                   "Please download binary from https://github.com/gdrive-org/gdrive \n"
                   "WARNING : Delete gdrive and {home_dir}/.gdrive after use".format(home_dir=home_dir))
        return

    runner = GDriveRunner(home_dir, cmd_args)
    if cmd_args.parent_id is None and cmd_args.parent is None:
        raise Exception("One of --parent-id and --parent should be provided")

    if cmd_args.parent_id is not None:
        parent = cmd_args.parent_id
    else:
        parent = runner.get_id(cmd_args.parent)

    cmd = '{gdrive} upload -p {data_url} {path}'

    if cmd_args.desc is not None:
        cmd += ' --description ' + cmd_args.desc

    paths = [cmd_args.path]
    if os.path.exists(cmd_args.path + ".index"):
        paths.append(cmd_args.path + ".index")

    for path in paths:
        p_cmd = cmd.format(gdrive=gdrive_bin, data_url=parent, path=path)
        runner.run(p_cmd)


def run_download(cmd_args: ArgNamespace):
    home_dir = os.path.expanduser("~")
    gdrive_bin = home_dir + '/gdrive'
    if not os.path.exists(gdrive_bin):
        logger.err("Could not find gdrive at {home_dir}. "
                   "Please download binary from https://github.com/gdrive-org/gdrive \n"
                   "WARNING : Delete gdrive and {home_dir}/.gdrive after use".format(home_dir=home_dir))
        return

    if cmd_args.path is None and cmd_args.path_id is None:
        raise Exception("One of --path and --path-id should be provided")

    runner = GDriveRunner(home_dir, cmd_args)

    if cmd_args.path_id is not None:
        path = cmd_args.path_id
    else:
        path = runner.get_id(cmd_args.path)

    cmd = '{gdrive} download {path} --force --path {outdir}'.format(gdrive=gdrive_bin, path=path,
                                                                    outdir=cmd_args.outdir)
    runner.run(cmd)


def run(cmd_args: ArgNamespace):
    if cmd_args.train_data_subcommand == 'upload':
        run_upload(cmd_args)

    elif cmd_args.train_data_subcommand == 'download':
        run_download(cmd_args)
