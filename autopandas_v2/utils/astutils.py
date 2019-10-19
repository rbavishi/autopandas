import ast
import inspect
from typing import Union, List

import astor
import astunparse
import copy


def parse(code, wrap_module=False):
    if wrap_module:
        result = ast.parse(code)
    else:
        result = ast.parse(code).body[0]

    return result


def parse_obj(obj):
    src = inspect.getsource(obj).strip()
    return parse(src)


def parse_file(fname: str) -> ast.Module:
    with open(fname, 'r') as f:
        return ast.parse(f.read().strip())


def to_source(node: ast.AST) -> str:
    return astunparse.unparse(node)
    # if pretty:
    #     return astor.to_source(node).strip()
    # else:
    #     return astor.to_source(node, pretty_source=lambda x: ''.join(x)).strip()


def get_op(op: ast.BinOp):
    return astor.op_util.get_op_symbol(op)


def copy_asts(asts: Union[ast.AST, List[ast.AST]]):
    if isinstance(asts, list):
        return [copy_asts(i) for i in asts]

    return copy.deepcopy(asts)


def attr_to_qual_name(node: ast.Attribute):
    accesses = [node.attr]
    while isinstance(node.value, ast.Attribute):
        node = node.value
        accesses.append(node.attr)

    accesses.append(node.value.id)
    qual_name = '.'.join(reversed(accesses))
    return qual_name
