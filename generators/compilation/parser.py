import ast
import functools
import itertools
import operator
from typing import Dict, List, Set, Callable, Optional, Tuple

from autopandas_v2.generators.compilation.ir.arguments import IArg
from autopandas_v2.generators.compilation.ir.generators import IGenerator
from autopandas_v2.generators.compilation.ir.signatures import ISignature
from autopandas_v2.utils import astutils
from autopandas_v2.utils.cli import ArgNamespace
from autopandas_v2.utils.graphs import topological_ordering


def parse_gen_from_ast(gen_ast: ast.FunctionDef, namespace: str, gen_id: str,
                       parse_cache: Dict[str, IGenerator], cmd_args: ArgNamespace) -> IGenerator:
    #  First go over the ast and assign ids to nodes using a pre-order traversal
    idx = 0
    for n in ast.walk(gen_ast):
        n._parser_idx = idx
        idx += 1

    sym_to_ast_def_map: Dict[str, ast.AST] = get_sym_to_ast_def_map(gen_ast)

    #  First check if this generator is to be extended from another
    extension: Optional[str] = get_extension_of(gen_ast)
    if extension is not None:
        parent: IGenerator = parse_cache[extension]
        signature: ISignature = parent.sig
        target: Callable = parent.target
        arity: int = parent.arity
        arg_irs: Dict[str, IArg] = parent.arg_irs.copy()
        arg_irs.update(get_arg_irs(gen_ast, signature, sym_to_ast_def_map))
        representation: str = parent.representation

    else:
        signature: ISignature = get_signature(gen_ast)
        target: Callable = get_target(gen_ast)
        arg_irs: Dict[str, IArg] = get_arg_irs(gen_ast, signature, sym_to_ast_def_map)

        inheritance: Optional[str] = get_inheritance(gen_ast)
        if inheritance is not None:
            arg_irs.update({k: v for k, v in parse_cache[inheritance].arg_irs.items()
                            if (k not in arg_irs) and (k in signature.arg_names)})

        arity: int = get_arity(gen_ast, arg_irs)
        representation: str = get_representation(gen_id, gen_ast, signature, active_args=list(arg_irs.keys()))

    #  Compute the enumeration order as the topological ordering of nodes in the dependency graph of the IArgs
    adjacency_list: Dict[str, Set[str]] = {i.arg_name: set(i.dependencies) for i in arg_irs.values()}
    tiebreaker = get_tiebreaker(arg_irs)
    enum_order: List[str] = topological_ordering(adjacency_list, tiebreak_ranking=tiebreaker)

    #  Construct the IGenerator object
    return IGenerator(namespace=namespace, name=gen_id, sig=signature,
                      arg_irs=arg_irs, enum_order=enum_order, target=target, arity=arity,
                      representation=representation)


def get_signature(gen_ast: ast.FunctionDef) -> ISignature:
    """
    Parse the signature of the function the generator is defined for into its positional and keyword arguments
    """

    for decorator in getattr(gen_ast, 'decorator_list', []):
        if isinstance(decorator, ast.Call) and decorator.func.id == 'signature':
            if len(decorator.args) == 1 and isinstance(decorator.args[0], ast.Str):
                return ISignature(decorator.args[0].s)
    else:
        raise Exception("Signature missing in generator definition")


def get_target(gen_ast: ast.FunctionDef) -> Callable:
    """
    The target is the function for which the generator is being written.
    The main purpose of the target however is to execute it with a given
    argument combination. Hence it may look quite a bit different from the
    actual function (such as attributes where the target will probably be a lambda
    """

    for decorator in getattr(gen_ast, 'decorator_list', []):
        if isinstance(decorator, ast.Call) and decorator.func.id == 'target':
            if len(decorator.args) == 1 and isinstance(decorator.args[0], (ast.Name, ast.Attribute, ast.Lambda)):
                return decorator.args[0]
    else:
        raise Exception("Target missing in generator definition")


def get_representation(gen_id: str, gen_ast: ast.FunctionDef, sig: ISignature, active_args: List[str]) -> str:
    """
    Representation is the format string that specifies how an assignment to the argument combinations
    will be printed out in a program produced by the search engine.
    For example, "{self}.pivot(columns={columns}, index={index}, values={values})" specifies that in
    a program produced by the search engine, a call to pivot (if any) would look something like
    inps[0].pivot(columns='data', index='item', values='value') where
    {'self': 'inps[0]', 'columns': 'data', 'index': 'item', 'values': 'value'} is the argument combination

    Representations can be specified manually using the @representation decorator, or can be inferred directly
    from the signature. In cases where this inference is not possible, this method will raise an error
    """

    for decorator in getattr(gen_ast, 'decorator_list', []):
        if isinstance(decorator, ast.Call) and decorator.func.id == 'representation':
            if len(decorator.args) == 1 and isinstance(decorator.args[0], ast.Str):
                return decorator.args[0].s

    #  We didn't find any manually-specified representation, so we now try to infer it
    if sig.type == 'method':
        #  We'll treat this as a method
        representation = '{self}.' + gen_id + '('
        representation += ', '.join(itertools.chain(
            map(lambda x: '{' + x + '}',
                (i for i in sig.pos_args if i in active_args and i != 'self')),
            map(lambda x: x + '={' + x + '}',
                (i[0] for i in sig.kw_args if i[0] in active_args and i[0] != 'self'))))

        representation += ')'

        return representation

    elif sig.type == 'attribute':
        representation = '{self}.' + gen_id
        return representation

    else:
        raise Exception("Cannot infer representation from signature")


def get_extension_of(gen_ast: ast.FunctionDef) -> Optional[str]:
    for decorator in getattr(gen_ast, 'decorator_list', []):
        if isinstance(decorator, ast.Call) and decorator.func.id == 'extend':
            if len(decorator.args) == 1 and isinstance(decorator.args[0], ast.Attribute):
                return astutils.to_source(decorator.args[0]).strip()


def get_inheritance(gen_ast: ast.FunctionDef) -> Optional[str]:
    for decorator in getattr(gen_ast, 'decorator_list', []):
        if isinstance(decorator, ast.Call) and decorator.func.id == 'inherit':
            if len(decorator.args) == 1 and isinstance(decorator.args[0], ast.Str):
                return decorator.args[0].s


def get_arity(gen_ast: ast.FunctionDef, arg_irs: Dict[str, IArg]) -> int:
    """
    Get the number of externals (inputs / intermediates) this generator can consume.
    This can be explicitly specified using the @arity decorator. If not specified,
    this is inferred to be the number of arguments where there is a use of the Ext/RExt DSL operator
    """

    for decorator in getattr(gen_ast, 'decorator_list', []):
        if isinstance(decorator, ast.Call) and decorator.func.id == 'arity':
            if len(decorator.args) == 1 and isinstance(decorator.args[0], ast.Num):
                return int(decorator.args[0].n)

    arity = 0

    def contains_ext(d_ast):
        for n in ast.walk(d_ast):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ['Ext', 'RExt']:
                return True

    for name, arg_ir in arg_irs.items():
        arity += any(contains_ext(i) for i in [arg_ir.expr] + arg_ir.defs)

    return arity


def get_tiebreaker(arg_irs: Dict[str, IArg]) -> Dict[str, Tuple[int, int]]:
    """
    This decides enum order in case of no dependencies between arguments
    Arguments with Exts have higher priority
    If more tiebreaking is needed, then the order in which they appear in the generator is used
    """
    result = {}

    def contains_ext(d_ast):
        for n in ast.walk(d_ast):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ['Ext', 'RExt']:
                return True

    for name, arg_ir in arg_irs.items():
        has_ext = any(contains_ext(i) for i in [arg_ir.expr] + arg_ir.defs)
        result[name] = (1 - int(has_ext), arg_ir.expr._parser_idx)

    return result


def get_sym_to_ast_def_map(gen_ast: ast.FunctionDef) -> Dict[str, ast.AST]:
    """
    Establishes a map between symbols and the AST nodes corresponding to their definitions.
    Since generators in our case only contain top-level assignments and function-defs, the map
    is just a map from the symbol on the LHS / function name, to the AST object corresponding
    to the RHS/function-def respectively.

    We do not worry about global imports / functions here. It is guaranteed that they will be available
    in the compiled version of the generator
    """

    result: Dict[str, ast.AST] = {}
    for stmt in gen_ast.body:
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1:
                raise NotImplementedError("Top-Level Assignments should only have a single LHS in generator defs")

            if not isinstance(stmt.targets[0], ast.Name):
                raise NotImplementedError("Only simple names allowed as LHS in top-level assignments in generator defs")

            lhs = stmt.targets[0].id
            result[lhs] = stmt

        elif isinstance(stmt, ast.FunctionDef):
            result[stmt.name] = stmt

        elif isinstance(stmt, ast.Pass):
            pass

        else:
            raise NotImplementedError("Top-Level {} not supported in generator defs".format(type(stmt)))

    return result


def get_arg_irs(gen_ast: ast.FunctionDef, sig: ISignature, sym_to_ast_def_map: Dict[str, ast.AST]) -> Dict[str, IArg]:
    """
    Split up the generator definition into separate parts, each of which contains a single argument's domain
    definition. This is done by slicing the definition based on the symbols used in defining each argument.
    We also compute the dependencies between the arguments. This is also easy to do due to the naming
    convention for argument definitions (a leading underscore).

    This is relatively straightforward due to the restricted generator top-level syntax of simple assgns + functions
    """

    #  First collect the assignments that directly assign to an argument
    #  i.e. where the lhs is of the form _{arg_name}
    arg_defs: Set[ast.Assign] = {stmt for stmt in gen_ast.body if isinstance(stmt, ast.Assign)
                                 and stmt.targets[0].id[len("_"):] in sig.arg_names}

    def get_sym_defs(node: ast.AST) -> Set[ast.AST]:
        return {sym_to_ast_def_map[n.id] for n in ast.walk(node)
                if isinstance(n, ast.Name) and n.id in sym_to_ast_def_map} - arg_defs

    def_map: Dict[str, List[ast.AST]] = {}
    dependency_map: Dict[str, Set[str]] = {}

    #  For each of them do a fixed-point analysis to collect the definitions of all the symbols required
    for arg_def in arg_defs:
        arg_name = arg_def.targets[0].id[len("_"):]
        required_defs = set()
        added_defs = {arg_def}
        while len(added_defs) > 0:
            required_defs |= added_defs
            added_defs.clear()

            for d in required_defs:
                added_defs |= get_sym_defs(d)

            added_defs -= required_defs

        required_defs.remove(arg_def)

        #  Sort it by the order of appearance in the source to automatically get the correct dependence order
        required_defs = sorted(required_defs, key=lambda x: x._parser_idx)
        def_map[arg_name] = required_defs

        #  Get dependencies on other arguments
        #  We simply check for all occurrences of _{arg_name} in the definition of the argument
        dependencies: Set[str] = {n.id[len("_"):] for asgn in required_defs + [arg_def] for n in ast.walk(asgn)
                                  if isinstance(asgn, ast.Assign) and isinstance(n, ast.Name)} & sig.arg_names
        dependencies -= {arg_name}
        dependency_map[arg_name] = dependencies

    #  Do a fixed point on the dependency map to resolve transitive dependencies
    changed = True
    while changed:
        orig_cnt = sum(len(i) for i in dependency_map.values())
        for k, v in dependency_map.items():
            if len(v) > 0:
                #  Doing an extra set inside reduce to create a copy
                dependency_map[k] = functools.reduce(operator.ior, (set(dependency_map[i]) for i in v)) | v

        new_cnt = sum(len(i) for i in dependency_map.values())
        changed = new_cnt > orig_cnt

    #  Now construct the IArg definitions
    result: Dict[str, IArg] = {}
    for arg_def in arg_defs:
        arg_name = arg_def.targets[0].id[len("_"):]
        result[arg_name] = IArg(arg_name=arg_name, expr=astutils.copy_asts(arg_def.value),
                                defs=astutils.copy_asts(def_map[arg_name]), dependencies=dependency_map[arg_name])

    return result
