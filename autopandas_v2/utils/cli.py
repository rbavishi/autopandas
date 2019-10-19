from argparse import ArgumentParser, Namespace
from typing import Callable, List


def subcommand(parent: ArgumentParser, cmd: str = None, help: str = None, dest: str = None,
               inherit_from: List[ArgumentParser] = None) -> Callable:
    if not hasattr(subcommand, '_subparsers_cache'):
        subcommand._subparsers_cache = {}

    if parent not in subcommand._subparsers_cache:
        subparsers = parent.add_subparsers(help='Available Commands')
        subparsers.required = True
        subparsers.dest = dest or cmd + '_subcommand'
        subcommand._subparsers_cache[parent] = subparsers

    subparsers = subcommand._subparsers_cache[parent]

    if inherit_from is None:
        child_parser = subparsers.add_parser(cmd, help=help)
    else:
        child_parser = subparsers.add_parser(cmd, help=help, parents=inherit_from)

    def wrapper_parser_def(func: Callable[[ArgumentParser], None]):
        func(child_parser)

    return wrapper_parser_def


class ArgNamespace(Namespace):
    @staticmethod
    def from_namespace(namespace: Namespace):
        res = ArgNamespace()
        for k, v in namespace.__dict__.items():
            setattr(res, k, v)

        return res

    def get(self, key, default):
        return self.__dict__.get(key, default)

    def __contains__(self, item):
        return item in self.__dict__
