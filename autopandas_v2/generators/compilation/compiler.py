import ast
import logging
from typing import Tuple, Dict, Optional

from autopandas_v2.generators.compilation.ir.arguments import IArg
from autopandas_v2.generators.compilation.ir.generators import IGenerator
from autopandas_v2.generators.compilation.parser import parse_gen_from_ast
from autopandas_v2.utils import astutils, logger
from autopandas_v2.utils.cli import ArgNamespace


def parse_gens_from_module(spec_ast: ast.Module, cmd_args: ArgNamespace,
                           parse_cache: Dict[str, Optional[IGenerator]]=None) -> Dict[str, Optional[IGenerator]]:
    gen_defs: Dict[Tuple[str, str], ast.FunctionDef] = GenCollector().collect(spec_ast)
    return parse_gens_from_defs(gen_defs, cmd_args, parse_cache=parse_cache)


def parse_gens_from_defs(gen_defs: Dict[Tuple[str, str], ast.FunctionDef], cmd_args: ArgNamespace,
                         parse_cache: Dict[str, Optional[IGenerator]]=None) -> Dict[str, Optional[IGenerator]]:

    parse_results: Dict[str, Optional[IGenerator]] = {}
    if parse_cache is not None:
        parse_results.update(parse_cache)

    for (namespace, gen_id), gen_def in gen_defs.items():
        try:
            logger.info("Parsing {}.{}".format(namespace, gen_id))
            igen: IGenerator = parse_gen_from_ast(gen_def, namespace, gen_id, parse_results, cmd_args)
            parse_results[namespace + '.' + gen_id] = igen
        except Exception as e:
            logger.err("Parsing of {}.{} failed".format(namespace, gen_id))
            logging.exception(e)
            parse_results[namespace + '.' + gen_id] = None

    return parse_results


def compile_gens_from_module(spec_ast: ast.Module, cmd_args: ArgNamespace,
                             parse_cache: Dict[str, Optional[IGenerator]] = None) -> Dict[ast.FunctionDef,
                                                                                          Optional[ast.ClassDef]]:
    #  All the function-defs containing the signature decorator will be treated as generators
    gen_defs: Dict[Tuple[str, str], ast.FunctionDef] = GenCollector().collect(spec_ast)
    compiled_map: Dict[ast.FunctionDef, Optional[ast.ClassDef]] = {}
    if parse_cache is None:
        parse_cache = {}

    parse_cache.update(parse_gens_from_defs(gen_defs, cmd_args, parse_cache=parse_cache))

    for (namespace, gen_id), gen_def in gen_defs.items():
        igen: IGenerator = parse_cache[namespace + '.' + gen_id]
        if igen is None:
            logger.err("Skipping {}.{} because of parse error".format(namespace, gen_id))
            compiled_map[gen_def] = None
            continue

        try:
            logger.info("Compiling {}.{}".format(namespace, gen_id))
            compiled_def: ast.ClassDef = compile_gen(igen)
            compiled_map[gen_def] = compiled_def
        except Exception as e:
            logger.err("Compilation of {}.{} failed".format(namespace, gen_id))
            logging.exception(e)
            compiled_map[gen_def] = None

    return compiled_map


def compile_gens_to_module(target_ast: ast.Module, gens: Dict[ast.FunctionDef, Optional[ast.ClassDef]],
                           cmd_args: ArgNamespace) -> ast.Module:
    #  Replace the function-defs with the new compiled class-defs
    GenSubstituter(gens).visit(target_ast)

    #  Add an import for the base generator definition, search-spec
    target_ast.body.insert(0, astutils.parse("from autopandas_v2.generators.base import BaseGenerator"))
    target_ast.body.insert(0, astutils.parse("from autopandas_v2.iospecs import SearchSpec"))
    target_ast.body.insert(0, astutils.parse("from autopandas_v2.generators.trackers import OpTracker"))
    return target_ast


class GenCollector(ast.NodeVisitor):
    def __init__(self):
        self.namespace = []
        self.collected: Dict[Tuple[str, str], ast.FunctionDef] = {}

    def collect(self, root: ast.Module):
        self.visit(root)
        return self.collected

    def visit_ClassDef(self, node: ast.ClassDef):
        self.namespace.append(node.name)
        self.generic_visit(node)
        self.namespace.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        for decorator in getattr(node, 'decorator_list', []):
            if not (isinstance(decorator, ast.Call) and decorator.func.id in ['signature', 'extend']):
                continue

            if not (len(decorator.args) == 1 and isinstance(decorator.args[0], (ast.Str, ast.Attribute))):
                continue

            namespace = '.'.join(self.namespace)
            self.collected[(namespace, node.name)] = node


class GenSubstituter(ast.NodeTransformer):
    def __init__(self, compiled_map: Dict[ast.FunctionDef, ast.ClassDef]):
        self.compiled_map = compiled_map

    def visit_FunctionDef(self, node: ast.FunctionDef):
        return self.compiled_map.get(node, node)

    def visit_ClassDef(self, node: ast.ClassDef):
        self.generic_visit(node)
        if len(node.body) == 0:
            return None

        return node


def compile_gen(igen: IGenerator) -> ast.ClassDef:
    """
    The compiler converts the internal representation into a class that can be used
    for enumeration, much like the 'Unit' class in autopandas-v1
    """

    #  First instantiate the raw class
    gen_class: ast.ClassDef = astutils.parse("class {}(BaseGenerator): pass".format(igen.name))
    gen_class.body = []

    #  Add the argument generator definitions
    for arg_name in igen.enum_order:
        compile_iarg(arg_name, igen, gen_class)

    #  Add an initialization method for the important generator information
    init_method: ast.FunctionDef = astutils.parse("def init(self): pass")
    init_method.body = []
    init_method.body.append(astutils.parse("self.target = {}".format(astutils.to_source(igen.target).strip())))
    init_method.body.append(astutils.parse("self.enum_order = {}".format(repr(igen.enum_order))))
    init_method.body.append(astutils.parse("self.arg_names = {}".format(repr(igen.sig.arg_names))))
    init_method.body.append(astutils.parse("self.default_vals = {}".format(repr(dict(igen.sig.kw_args)))))

    qual_name = igen.namespace + '.' + igen.name
    init_method.body.append(astutils.parse("self.qual_name = {}".format(repr(qual_name))))
    init_method.body.append(astutils.parse("self.name = {}".format(repr(igen.name))))
    init_method.body.append(astutils.parse("self.arity = {}".format(igen.arity)))

    init_method.body.append(astutils.parse("self.representation = {}".format(repr(igen.representation))))
    gen_class.body.append(init_method)
    return gen_class


def compile_iarg(arg_name: str, igen: IGenerator, gen_class: ast.ClassDef):
    """
    This method creates a method in gen_class that can be invoked to get the generator
    object for the concerned argument. This method will take as arguments, values of the function arguments
    whose values its generator definition depends on
    """

    iarg: IArg = igen.arg_irs[arg_name]

    #  First create the argument string for this method
    #  spec contains the Search I/O spec or the Training I/O spec for the task at hand
    #  mode is either 'training', 'exhaustive' or 'smart'
    #  'training' enables the training data generation for the smart ops (like Select)
    #  'exhaustive' is the original exhaustive generator
    #  'smart' is the neural-guided version

    #  TODO : Add boiler-plate for smart-generators

    arg_str = "self, _spec: SearchSpec, _mode: str, _depth: int"
    if len(iarg.dependencies) > 0:
        arg_str += ", " + ", ".join(map(lambda x: "_" + x, iarg.dependencies))

    #  To take care of don't-care arguments that may be supplied
    arg_str += ", _tracker: OpTracker=None, **kwargs"

    iarg_method = astutils.parse("def _arg_{}({}): pass".format(iarg.arg_name, arg_str))

    #  Now populate the body with the definition
    iarg_method.body = iarg.defs[:]
    iarg_method.body.append(astutils.parse("return placeholder"))
    iarg_method.body[-1].value = iarg.expr

    #  Add it to the class
    gen_class.body.append(iarg_method)

    #  Time for some compiler passes!

    #  -------------------------------
    #  Pass 1 : Add DSL-Ops Boiler-plate
    #  -------------------------------

    #  Pass 1 : Get all calls to the DSL ops, and add the necessary boiler-plate
    #  This boiler-plate includes access to the mode (training, exhaustive, smart),
    #  the I/O spec, a unique identifier corresponding to each distinct location (so
    #  we can train separate models) and the argument_name

    pass_dsl_op(iarg_method, iarg)


def pass_dsl_op(iarg_method: ast.FunctionDef, iarg: IArg):
    #  TODO : Keep adding more
    recognized_ops = ['Ext', 'Select', 'Choice', 'Chain', 'Subsets', 'OrderedSubsets', 'Product', 'RExt']
    identifier = 1
    for call in filter(lambda x: isinstance(x, ast.Call), ast.walk(iarg_method)):
        if not isinstance(call.func, ast.Name):
            continue

        func_name = call.func.id
        if func_name not in recognized_ops:
            continue

        #  Do not modify if already done
        if any(kw.arg == 'identifier' for kw in call.keywords):
            continue

        #  Pass the mode, iospec, arg_name and identifier keyword args
        call.keywords.append(ast.keyword(arg='spec', value=ast.Name(id='_spec')))
        call.keywords.append(ast.keyword(arg='mode', value=ast.Name(id='_mode')))
        call.keywords.append(ast.keyword(arg='depth', value=ast.Name(id='_depth')))
        call.keywords.append(ast.keyword(arg='arg_name', value=ast.Str(iarg.arg_name)))
        call.keywords.append(ast.keyword(arg='identifier', value=ast.Str(str(identifier))))

        #  Pass the tracker
        call.keywords.append(ast.keyword(arg='tracker', value=ast.Name(id='_tracker')))

        #  Pass the extra kwargs
        call.keywords.append(ast.keyword(arg=None, value=ast.Name(id='kwargs')))

        identifier += 1
