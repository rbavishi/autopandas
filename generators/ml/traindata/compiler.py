import ast
from typing import Optional, Dict

from autopandas_v2.generators.compilation.compiler import parse_gens_from_module, compile_gens_to_module, \
    compile_gens_from_module
from autopandas_v2.generators.compilation.ir.generators import IGenerator
from autopandas_v2.utils import logger
from autopandas_v2.utils.cli import ArgNamespace


def compile_randomized_gens(spec_ast: ast.Module, orig_spec_ast: ast.Module,
                            cmd_args: ArgNamespace) -> Dict[ast.FunctionDef, Optional[ast.ClassDef]]:
    """
    spec_ast contains the AST corresponding to the AST containing the randomized specs
    orig_spec_ast contains the AST corresponding to the AST containing the original specs.
    An example of orig_spec_ast may be the one contained in the file autopandas_v2.generators.specs
    """
    orig_parse_results: Dict[str, Optional[IGenerator]] = parse_gens_from_module(orig_spec_ast, cmd_args)
    orig_parse_results = {'s_' + k: v for k, v in orig_parse_results.items()}
    logger.info("---------------------------------")
    logger.info("Parsing of original gen defs done")
    logger.info("---------------------------------")
    randomized_parse_results = parse_gens_from_module(spec_ast, cmd_args, parse_cache=orig_parse_results)
    return compile_gens_from_module(spec_ast, cmd_args, parse_cache=randomized_parse_results)
