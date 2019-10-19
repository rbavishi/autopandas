import glob
import inspect
import os
from typing import Dict, Type

import autopandas_v2.evaluation.benchmarks
from autopandas_v2.evaluation.benchmarks.base import Benchmark
from autopandas_v2.utils.misc import get_all_defined_classes_recursive, import_file


def discover_benchmarks() -> Dict[str, Type[Benchmark]]:
    benchmarks: Dict[str, Type[Benchmark]] = {}
    b_dir = os.path.dirname(inspect.getfile(autopandas_v2.evaluation.benchmarks))
    paths = glob.glob(b_dir + '/**/*.py', recursive=True)
    for path in paths:
        mod = import_file('bench_mod', path)
        for name, cls in get_all_defined_classes_recursive(mod):
            if issubclass(cls, Benchmark) and not issubclass(Benchmark, cls):
                benchmarks[name] = cls

    return benchmarks
