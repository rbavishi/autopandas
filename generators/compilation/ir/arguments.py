import ast
from typing import List, Set

from autopandas_v2.utils import astutils


class IArg:
    def __init__(self, arg_name: str, expr: ast.AST, defs: List[ast.AST], dependencies: Set[str]):
        """
        defs are the supporting definitions (functions/assignments) required to generate the domain
        expr is the RHS of the final assignment to the argument in the generator definition i.e. where the LHS
        is of the form _{arg_name}
        """
        self.arg_name = arg_name
        self.expr = expr
        self.defs = defs
        self.dependencies = dependencies

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = "IArg for " + self.arg_name + ":\n"
        s += "-" * 20 + "\n"
        for d in self.defs:
            s += astutils.to_source(d).strip() + "\n"

        s += "return " + astutils.to_source(self.expr).strip() + "\n"
        s += "-" * 20 + "\n"
        s += "Dependencies : " + str(self.dependencies) + "\n"
        s += "-" * 20 + "\n"
        return s
