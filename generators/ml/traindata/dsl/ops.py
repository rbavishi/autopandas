import random
from typing import Any, Callable, List, Optional

from autopandas_v2.generators.dsl.values import Fetcher, Value
from autopandas_v2.generators.ml.traindata.dsl.values import NewInp
from autopandas_v2.generators.trackers import OpTracker
from autopandas_v2.iospecs import SearchSpec
from autopandas_v2.utils.exceptions import AutoPandasException
from autopandas_v2.utils.types import DType


def RExt(dtype: DType, rgen=None, spec: SearchSpec = None, depth: int = 1, mode: str = None, tracker: OpTracker = None,
         arg_name: str = None, identifier: str = None, constraint: Callable[[Any], Any] = None, **kwargs):

    if constraint is None:
        def constraint(x):
            return True

    if mode != 'training-data':
        raise AutoPandasException("Unrecognized mode {} in RExt".format(mode))

    pool: List[Optional[Value]] = []
    for idx, val in enumerate(spec.inputs):
        if not (dtype.hasinstance(val) and constraint(val)):
            continue
        pool.append(Fetcher(val=val, source='inps', idx=idx))

    for idx, val in enumerate(spec.intermediates[:depth-1]):
        if not (dtype.hasinstance(val) and constraint(val)):
            continue
        pool.append(Fetcher(val=val, source='intermediates', idx=idx))

    if rgen is not None:
        pool.append(None)

    random.shuffle(pool)
    label = 'ext_' + arg_name + '_' + identifier
    rlabel = 'rext_' + arg_name + '_' + identifier
    for selection in pool:
        tracker.record.pop(label, None)
        tracker.record.pop(rlabel, None)
        if selection is None:
            #  We've decided to create a new input altogether
            val = next(rgen)
            tracker.record[rlabel] = {'val': val, 'arg_name': arg_name}
            yield NewInp(val)

        else:
            selection: Fetcher
            tracker.record[label] = {'source': selection.source, 'idx': selection.idx}
            yield selection
