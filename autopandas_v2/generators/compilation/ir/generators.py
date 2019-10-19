from typing import Dict, List, Callable

from autopandas_v2.generators.compilation.ir.arguments import IArg
from autopandas_v2.generators.compilation.ir.signatures import ISignature


class IGenerator:
    def __init__(self, namespace: str, name: str, sig: ISignature,
                 arg_irs: Dict[str, IArg], enum_order: List[str], target: Callable, arity: int,
                 representation: str):

        self.sig = sig
        self.namespace = namespace
        self.name = name
        self.sig = sig
        self.arg_irs = arg_irs
        self.enum_order = enum_order
        self.target = target
        self.arity = arity
        self.representation = representation
