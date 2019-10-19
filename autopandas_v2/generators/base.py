import collections
import functools
import glob
import itertools
import logging
import operator
import os
from typing import List, Set, Dict, Any, Generator, Callable, Tuple

from autopandas_v2.generators.dsl.values import Default, Fetcher, AnnotatedVal, Inactive, RandomColumn
from autopandas_v2.generators.ml.traindata.dsl.values import NewInp
from autopandas_v2.generators.trackers import OpTracker
from autopandas_v2.iospecs import SearchSpec
from autopandas_v2.ml.inference.interfaces import RelGraphInterface
from autopandas_v2.utils import logger
from autopandas_v2.utils.exceptions import AutoPandasInversionFailedException


class BaseGenerator:
    def __init__(self):
        self.name: str = None
        self.qual_name: str = None
        self.target: Callable = None
        self.enum_order: List[str] = None
        self.arg_names: Set[str] = None
        self.default_vals: Dict[str, Any] = None
        self.arity: int = None
        self.representation: str = None

        self.init()

    def init(self):
        """
        Will be populated in the sub-classes.
        This method should initialize all the fields specified in __init__
        """
        pass

    def process_val(self, val):
        if isinstance(val, Default):
            return val.val, {'is_default': True}

        if isinstance(val, Inactive):
            #  Returning '' is a very quick way to check if annotations are empty,
            #  it fails the (if <expr>) check and supports the __contains__ method
            return val.val, ''

        if isinstance(val, NewInp):
            return val.val, {'new_inp': True}

        if isinstance(val, AnnotatedVal):
            return val.val, val.annotations

        if isinstance(val, Fetcher):
            return val.val, {'fetcher': (val, val.repr),
                             'sources': [val.source],
                             'indices': [val.idx]}

        if isinstance(val, RandomColumn):
            return val.val, None

        return val, None

    def enumerate_exhaustive(self, spec: SearchSpec, depth: int = None):
        self.init()
        arg_gens = [getattr(self, "_arg_" + aname) for aname in self.enum_order]

        #  Enumeration using cross-product
        #  We proceed in the enumeration order one-by-one

        top: int = 0
        total = len(arg_gens)
        arg_vals: Dict[str, Any] = {}
        arg_annotations: Dict[str, Dict[str, Any]] = {}
        arg_val_params: Dict[str, Any] = {}
        iters: List[Generator] = [None] * total
        if depth is None:
            depth = spec.depth

        while top > -1:
            if top == total:
                yield arg_vals, arg_annotations
                top -= 1

            arg_name = self.enum_order[top]
            try:
                if iters[top] is None:
                    iters[top] = arg_gens[top](_spec=spec, _mode='exhaustive', _depth=depth, **arg_val_params)

                val, annotation = self.process_val(next(iters[top]))
                arg_vals[arg_name] = val
                arg_annotations[arg_name] = annotation
                arg_val_params["_" + arg_name] = val

                top += 1

            except StopIteration:
                iters[top] = None
                top -= 1

    def generate_training_data(self, spec: SearchSpec, depth: int = None):
        self.init()
        arg_gens = [getattr(self, "_arg_" + aname) for aname in self.enum_order]

        #  Enumeration using cross-product
        #  We proceed in the enumeration order one-by-one

        top: int = 0
        total = len(arg_gens)
        arg_vals: Dict[str, Any] = {}
        arg_annotations: Dict[str, Dict[str, Any]] = {}
        arg_val_params: Dict[str, Any] = {}
        iters: List[Generator] = [None] * total
        tracker: OpTracker = OpTracker()
        orig_depth = spec.depth

        if depth is None:
            depth = orig_depth

        #  Hide the depth from the generators
        spec.depth = -1
        while top > -1:
            if top == total:
                yield arg_vals, arg_annotations, tracker.copy()
                top -= 1

            arg_name = self.enum_order[top]
            try:
                if iters[top] is None:
                    iters[top] = arg_gens[top](_spec=spec, _mode='training-data', _depth=depth,
                                               _tracker=tracker, **arg_val_params)

                val, annotation = self.process_val(next(iters[top]))
                arg_vals[arg_name] = val
                arg_annotations[arg_name] = annotation
                arg_val_params["_" + arg_name] = val

                top += 1

            except StopIteration:
                iters[top] = None
                top -= 1

        spec.depth = orig_depth

    def generate_arguments_training_data(self, spec: SearchSpec, depth: int = None, tracker: OpTracker = None):
        self.init()
        arg_gens = [getattr(self, "_arg_" + aname) for aname in self.enum_order]
        training_points: Dict[str, List[Any]] = collections.defaultdict(list)

        #  Enumeration using cross-product
        #  We proceed in the enumeration order one-by-one

        top: int = 0
        total = len(arg_gens)
        iters: List[Generator] = [None] * total
        arg_val_params: Dict[str, Any] = {}
        cur_points: Dict[str, Any] = collections.defaultdict(list)
        mode = 'arguments-training-data' if tracker is not None else 'arguments-training-data-best-effort'
        externals: Dict[str, Any] = {}
        orig_depth = spec.depth

        if depth is None:
            depth = orig_depth

        #  Hide the depth from the generators
        spec.depth = -1
        while top > -1:
            if top == total:
                for k, v in cur_points.items():
                    training_points[k].append(v)

                top -= 1

            try:
                arg_name = self.enum_order[top]
                externals.pop(arg_name, None)
                if iters[top] is None:
                    iters[top] = arg_gens[top](_spec=spec, _mode=mode, _depth=depth,
                                               _tracker=tracker, training_points_collector=cur_points,
                                               externals=externals,
                                               **arg_val_params)

                val, annotation = self.process_val(next(iters[top]))
                if annotation and 'sources' in annotation:
                    externals[arg_name] = val

                arg_val_params["_" + arg_name] = val
                top += 1

            except StopIteration:
                iters[top] = None
                top -= 1

            except AutoPandasInversionFailedException as e:
                iters[top] = None
                top -= 1
                logger.warn("Failed to invert generator")
                logging.exception(e)

        return training_points

    def infer(self, spec: SearchSpec, model_store: Dict[str, RelGraphInterface], depth: int = None, k: int = 5):
        """
        This enumeration does a beam search over the argument combination space, using the probabilities
        returned by DSL operators. The search depth (k) should be chosen wisely
        """

        self.init()
        arg_gens = [getattr(self, "_arg_" + aname) for aname in self.enum_order]

        #  Enumeration using cross-product
        #  We proceed in the enumeration order one-by-one
        #  To do the beam-search, we take the first-k predictions at every point,
        #  and sort at the end. If k is small, this should be finished in reasonable time
        #  TODO : Is it possible to design a hybrid that will work for larger k?

        top: int = 0
        total = len(arg_gens)
        arg_vals: Dict[str, Any] = {}
        arg_annotations: Dict[str, Dict[str, Any]] = {}
        arg_val_params: Dict[str, Any] = {}
        iters: List[Generator] = [None] * total
        prob_store: Dict[str, float] = {}
        externals: Dict[str, Any] = {}

        if depth is None:
            depth = spec.depth

        #  Basically the arg_val and arg_annotation tuple along with the probability
        arg_candidates: List[Tuple[float, Dict[str, Any], Dict[str, Dict[str, Any]]]] = []
        while top > -1:
            if top == total:
                arg_candidates.append((functools.reduce(operator.mul, prob_store.values(), 1.0),
                                       arg_vals.copy(), arg_annotations.copy()))
                top -= 1

            arg_name = self.enum_order[top]
            externals.pop(arg_name, None)
            try:
                if iters[top] is None:
                    iters[top] = itertools.islice(arg_gens[top](_spec=spec, _mode='inference',
                                                                _depth=depth, func=self.qual_name,
                                                                beam_search_k=k,
                                                                model_store=model_store, prob_store=prob_store,
                                                                externals=externals,
                                                                **arg_val_params), k)

                val, annotation = self.process_val(next(iters[top]))
                if annotation and 'sources' in annotation:
                    externals[arg_name] = val

                arg_vals[arg_name] = val
                arg_annotations[arg_name] = annotation
                arg_val_params["_" + arg_name] = val

                top += 1

            except StopIteration:
                iters[top] = None
                top -= 1

        # for model in model_store.values():
        #     model.close()

        arg_candidates = sorted(arg_candidates, key=lambda x: -x[0])
        for prob, arg_vals, arg_annotations in arg_candidates[:k]:
            yield arg_vals, arg_annotations

    def infer_lazy(self, spec: SearchSpec, model_store: Dict[str, RelGraphInterface], depth: int = None, k: int = 5):
        """
        This enumeration uses neural-guided generators but does not do a beam-search.
        This is faster but may not necessarily yield the best-k argument combinations in the correct order
        """

        self.init()
        arg_gens = [getattr(self, "_arg_" + aname) for aname in self.enum_order]

        #  Enumeration using cross-product
        #  We proceed in the enumeration order one-by-one

        top: int = 0
        total = len(arg_gens)
        arg_vals: Dict[str, Any] = {}
        arg_annotations: Dict[str, Dict[str, Any]] = {}
        arg_val_params: Dict[str, Any] = {}
        iters: List[Generator] = [None] * total
        externals: Dict[str, Any] = {}
        prob_store: Dict[str, float] = {}

        if depth is None:
            depth = spec.depth

        while top > -1:
            if top == total:
                yield arg_vals, arg_annotations
                top -= 1

            arg_name = self.enum_order[top]
            externals.pop(arg_name, None)
            try:
                if iters[top] is None:
                    iters[top] = itertools.islice(arg_gens[top](_spec=spec, _mode='inference',
                                                                _depth=depth, func=self.qual_name,
                                                                beam_search_k=k,
                                                                model_store=model_store, prob_store=prob_store,
                                                                externals=externals,
                                                                **arg_val_params), k)

                val, annotation = self.process_val(next(iters[top]))
                if annotation and 'sources' in annotation:
                    externals[arg_name] = val

                arg_vals[arg_name] = val
                arg_annotations[arg_name] = annotation
                arg_val_params["_" + arg_name] = val

                top += 1

            except StopIteration:
                iters[top] = None
                top -= 1

        # for model in model_store.values():
        #     model.close()
