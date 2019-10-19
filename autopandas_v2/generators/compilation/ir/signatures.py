import ast
from numpy import nan
from typing import List, Tuple, Any, Set


class ISignature:
    def __init__(self, sig: str):
        self.sig: str = sig
        self.fname: str = None
        self.pos_args: List[str] = None
        self.kw_args: List[Tuple[str, Any]] = None  # Keyword arguments have both a name and a default value
        self.arg_names: Set[str] = None
        self.type: str = None

        self.init()

    def init(self):
        root = ast.parse(self.sig.lstrip().rstrip())
        for node in ast.walk(root):
            if isinstance(node, ast.Call):
                self.pos_args = list(map(lambda x: x.id, node.args))
                self.kw_args = list(map(lambda x: (x.arg, self.eval_node(x.value)), node.keywords))

                if isinstance(node.func, ast.Attribute):
                    #  This is a method
                    fname = [node.func.attr]
                    node = node.func
                    #  Sometimes things can be complicated, the following is to process
                    #  cases like 'DataFrame.at.__getitem__'
                    while isinstance(node.value, ast.Attribute):
                        node = node.value
                        fname.append(node.attr)

                    fname.append(node.value.id)
                    self.fname = ".".join(reversed(fname))
                    self.type = 'method'

                else:
                    #  This is a general function not belonging to any particular class
                    #  Ideally this should never happen
                    self.fname = node.func.id

                self.arg_names = set(self.pos_args) | set(map(lambda x: x[0], self.kw_args))

                return

            elif isinstance(node, ast.Attribute):
                self.pos_args = ['self']
                self.kw_args = []
                fname = [node.attr]

                while isinstance(node.value, ast.Attribute):
                    node = node.value
                    fname.append(node.attr)

                fname.append(node.value.id)
                self.fname = ".".join(reversed(fname))

                self.arg_names = set(self.pos_args) | set(map(lambda x: x[0], self.kw_args))
                self.type = 'attribute'
                return

        raise Exception("Malformed Signature!")

    @staticmethod
    def eval_node(node):
        return eval(compile(ast.Expression(node), 'dummy', 'eval'), {'nan': nan})
