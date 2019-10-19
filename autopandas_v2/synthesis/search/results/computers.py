"""
Computers are functions that given an I/O spec, produce a value for a particular argument of a function.
Although most are going to be simple functions that return concrete values in all cases, there are some that
return a particular index of the input (such as the values returned by the Ext DSL operator. The idea behind
computers is to generalize this concept.
"""
from typing import Any, Dict, Set, Optional

from autopandas_v2.iospecs import SearchSpec


class ArgComputer:
    def __init__(self, name: str, val: Any, annotation: Dict[str, Any]):
        self.name = name
        self.val = val
        self.annotation = annotation

        self.is_default = annotation and annotation.get('is_default', False)
        if annotation is not None and 'fetcher' in annotation:
            self.computer, self.repr = annotation['fetcher']

        else:
            self.computer = None
            self.repr = repr(val)

    def get_used_intermediates(self) -> Set[int]:
        if (not self.annotation) or 'sources' not in self.annotation:
            return set()

        return {idx for src, idx in zip(self.annotation['sources'], self.annotation['indices'])
                if src == 'intermediates'}

    def get_used_inputs(self) -> Set[int]:
        if (not self.annotation) or 'sources' not in self.annotation:
            return set()

        return {idx for src, idx in zip(self.annotation['sources'], self.annotation['indices'])
                if src == 'inps'}

    def __call__(self, iospec: SearchSpec):
        if self.computer is None:
            return self.val
        else:
            return self.computer(iospec)
