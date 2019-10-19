from typing import Dict

from autopandas_v2.generators.base import BaseGenerator
from autopandas_v2.utils.misc import get_all_defined_classes_recursive


def load_generators() -> Dict[str, BaseGenerator]:
    generators: Dict[str, BaseGenerator] = {}
    import autopandas_v2.generators.specs_compiled as gens
    for name, cls in get_all_defined_classes_recursive(gens):
        if issubclass(cls, BaseGenerator):
            generators[name] = cls()

    return generators


def load_randomized_generators() -> Dict[str, BaseGenerator]:
    generators: Dict[str, BaseGenerator] = {}
    import autopandas_v2.generators.ml.traindata.specs_compiled as gens
    for name, cls in get_all_defined_classes_recursive(gens):
        if issubclass(cls, BaseGenerator):
            generators[name] = cls()

    return generators
