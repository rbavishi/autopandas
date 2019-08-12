import collections
import itertools
import random
from typing import Collection, Any, Iterable, Callable, Generator, List, Dict, Tuple

from autopandas_v2.generators.dsl.values import Fetcher, Value, RandomColumn
from autopandas_v2.generators.ml.featurization.featurizers import RelationGraphSelect, RelationGraphSubsets, \
    RelationGraphChain, RelationGraphChoice, RelationGraphOrderedSubsets, RelationGraphProduct
from autopandas_v2.generators.trackers import OpTracker
from autopandas_v2.iospecs import SearchSpec, ArgTrainingSpec
from autopandas_v2.ml.inference.interfaces import RelGraphInterface
from autopandas_v2.utils import logger
from autopandas_v2.utils.checker import Checker
from autopandas_v2.utils.exceptions import AutoPandasInversionFailedException
from autopandas_v2.utils.types import DType


def Ext(dtype: DType, spec: SearchSpec = None, depth: int = 1, mode: str = None, tracker: OpTracker = None,
        arg_name: str = None, identifier: str = None, constraint: Callable[[Any], Any] = None, **kwargs):
    if constraint is None:
        def constraint(x):
            return True

    if mode == 'exhaustive' or mode == 'inference':
        for idx, val in enumerate(reversed(spec.intermediates[:depth-1])):
            idx = depth - idx - 2
            if not (dtype.hasinstance(val) and constraint(val)):
                continue
            yield Fetcher(val=val, source='intermediates', idx=idx)

        for idx, val in enumerate(spec.inputs):
            if not (dtype.hasinstance(val) and constraint(val)):
                continue
            yield Fetcher(val=val, source='inps', idx=idx)

    elif mode == 'arguments-training-data':
        label = 'ext_' + arg_name + '_' + identifier
        if label not in tracker.record:
            raise AutoPandasInversionFailedException("Could not find label {} in tracker".format(label))

        record = tracker.record[label]
        idx = record['idx']
        if record['source'] == 'inps':
            yield Fetcher(val=spec.inputs[idx], source='inps', idx=idx)

        elif record['source'] == 'intermediates':
            yield Fetcher(val=spec.intermediates[idx], source='intermediates', idx=idx)

        return

    elif mode == 'arguments-training-data-best-effort':
        training_spec: ArgTrainingSpec = spec
        label = 'ext_' + arg_name + '_' + identifier
        for idx, val in enumerate(spec.inputs):
            if not (dtype.hasinstance(val) and constraint(val)):
                continue

            if Checker.check(val, training_spec.args[arg_name]):
                yield Fetcher(val=val, source='inps', idx=idx)
                return

        for idx, val in enumerate(spec.intermediates[:depth-1]):
            if not (dtype.hasinstance(val) and constraint(val)):
                continue

            if Checker.check(val, training_spec.args[arg_name]):
                yield Fetcher(val=val, source='intermediates', idx=idx)
                return

        raise AutoPandasInversionFailedException("Could not invert generator for {} at {}".format(arg_name, label))


def Select(domain: Collection[Any], spec: SearchSpec = None, depth: int = 1, mode: str = None,
           tracker: OpTracker = None, arg_name: str = None, identifier: str = None, **kwargs):
    label = 'select_' + arg_name + '_' + identifier

    if mode == 'exhaustive' or (mode == 'inference' and label not in kwargs['model_store']):
        if mode == 'inference':
            logger.warn("Did not find model for {}.{}".format(kwargs['func'], label), use_cache=True)

        yield from domain

    elif mode == 'training-data':
        #  The problem with Select is that many generators use the dynamic nature of Select to demonstrate
        #  different runs for the same I/O example in training/enumeration mode. 
        #  Since the output is not available during training-data generation, the value passed to Select in both
        #  modes will be different. Hence we cannot rely on simply storing the idx. So we store the value
        #  explicitly.
        #
        #  Note that this won't be a problem for Chain/Choice as the number of arguments is static
        domain = list(domain)
        random.shuffle(domain)
        for idx, val in enumerate(domain):
            if isinstance(val, Value):
                val = val.val

            tracker.record[label] = {'val': val}
            yield val

        tracker.record.pop(label, None)

    elif mode in ['arguments-training-data', 'arguments-training-data-best-effort']:
        training_collector = kwargs['training_points_collector']
        externals: Dict[str, Any] = kwargs['externals']
        if mode == 'arguments-training-data':
            if label not in tracker.record:
                raise AutoPandasInversionFailedException("Could not find label {} in tracker".format(label))

            target_val = tracker.record[label]['val']

        else:
            training_spec: ArgTrainingSpec = spec
            target_val = training_spec.args[arg_name]

        domain = list(domain)

        #  TODO : Come up with a better more general solution
        randoms = [(idx, val.val) for idx, val in enumerate(domain) if isinstance(val, RandomColumn)]
        domain = [val.val if isinstance(val, RandomColumn) else val for val in domain]

        selected_idx = -1
        selected_val = None

        for idx, val in enumerate(domain):
            if Checker.check(val, target_val):
                selected_idx = idx
                selected_val = val
                break

        else:
            #  So that didn't work out... There was no value in the domain that was equal to the target val.
            #  This can happen when random column names are generated. 

            if isinstance(target_val, str) and target_val.startswith("AUTOPANDAS_"):
                if len(randoms) > 0:
                    #  Great, so we can assume it was one of these randoms and it should be correct in most cases
                    selected_idx = randoms[0][0]
                    domain[selected_idx] = target_val
                    selected_val = target_val

        if selected_idx == -1:
            raise AutoPandasInversionFailedException("Could not invert generator for {} at {}".format(arg_name, label))

        #  Providing (spec.inputs, spec.output) might not be appropriate for higher-depths
        # graph: RelationGraphSelect = RelationGraphSelect.init(spec.inputs, spec.output)
        graph: RelationGraphSelect = RelationGraphSelect.init(list(externals.values()), spec.output)
        graph.add_domain(list(domain), selected_idx)

        encoding = graph.get_encoding()
        encoding['op_label'] = label
        training_collector[label] = encoding
        yield selected_val
        return

    elif mode == 'inference':
        model_store: Dict[str, RelGraphInterface] = kwargs['model_store']
        prob_store: Dict[str, float] = kwargs['prob_store']
        externals: Dict[str, Any] = kwargs['externals']
        domain = list(domain)

        if len(domain) == 0:
            return

        # graph: RelationGraphSelect = RelationGraphSelect.init(spec.inputs, spec.output)
        graph: RelationGraphSelect = RelationGraphSelect.init(list(externals.values()), spec.output)
        graph.add_domain(domain, query=True)

        encoding, reverse_mapping = graph.get_encoding(get_mapping=False, get_reverse_mapping=True)
        encoding['op_label'] = label
        encoding['domain_raw'] = domain
        #  The inference in Select returns a list of tuples (probability, domain_idx)
        inferred: List[Tuple[float, int]] = sorted(model_store[label].predict_graphs([encoding])[0],
                                                   key=lambda x: -x[0])
        for prob, encoding_node_idx in inferred:
            domain_idx = reverse_mapping[encoding_node_idx]
            prob_store[label] = prob
            yield domain[domain_idx]


def Choice(*choices: Any, spec: SearchSpec = None, depth: int = 1, mode: str = None, tracker: OpTracker = None,
           arg_name: str = None, identifier: str = None, **kwargs):
    label = 'choice_' + arg_name + '_' + identifier

    if mode == 'exhaustive' or (mode == 'inference' and label not in kwargs['model_store']):
        if mode == 'inference':
            logger.warn("Did not find model for {}.{}".format(kwargs['func'], label), use_cache=True)

        yield from choices

    elif mode == 'training-data':
        choices_with_idx = list(enumerate(choices))
        random.shuffle(choices_with_idx)
        for idx, val in choices_with_idx:
            tracker.record[label] = {'idx': idx}
            yield val

        tracker.record.pop(label, None)

    elif mode == 'arguments-training-data':
        training_collector = kwargs['training_points_collector']
        externals: Dict[str, Any] = kwargs['externals']
        if label not in tracker.record:
            raise AutoPandasInversionFailedException("Could not find label {} in tracker".format(label))

        idx = tracker.record[label]['idx']
        choices = list(choices)
        # graph: RelationGraphChoice = RelationGraphChoice.init(spec.inputs, spec.output)
        graph: RelationGraphChoice = RelationGraphChoice.init(list(externals.values()), spec.output)
        graph.add_choices(len(choices), chosen=idx)
        encoding = graph.get_encoding()
        encoding['op_label'] = label
        training_collector[label] = encoding

        yield choices[idx]

    elif mode == 'arguments-training-data-best-effort':
        raise NotImplementedError("Best-effort procedure not implemented for Choice")

    elif mode == 'inference':
        model_store: Dict[str, RelGraphInterface] = kwargs['model_store']
        prob_store: Dict[str, float] = kwargs['prob_store']
        externals: Dict[str, Any] = kwargs['externals']
        choices = list(choices)
        # graph: RelationGraphChoice = RelationGraphChoice.init(spec.inputs, spec.output)
        graph: RelationGraphChoice = RelationGraphChoice.init(list(externals.values()), spec.output)
        graph.add_choices(len(choices), query=True)
        encoding = graph.get_encoding(get_mapping=False)
        encoding['op_label'] = label
        encoding['choices_raw'] = choices
        #  The inference in Choice returns a list of tuples (probability, choice_idx)
        inferred: List[Tuple[float, int]] = sorted(model_store[label].predict_graphs([encoding])[0],
                                                   key=lambda x: -x[0])

        for prob, choice_idx in inferred:
            prob_store[label] = prob
            yield choices[choice_idx]


def Chain(*ops: Any, spec: SearchSpec = None, depth: int = 1, mode: str = None, tracker: OpTracker = None,
          arg_name: str = None, identifier: str = None, **kwargs):
    label = 'chain_' + arg_name + '_' + identifier

    if mode == 'exhaustive' or (mode == 'inference' and label not in kwargs['model_store']):
        if mode == 'inference':
            logger.warn("Did not find model for {}.{}".format(kwargs['func'], label), use_cache=True)

        for op in ops:
            if isinstance(op, Generator):
                yield from op
            else:
                yield op

    elif mode == 'training-data':
        ops_with_idx = list(enumerate(ops))
        random.shuffle(ops_with_idx)
        for idx, op in ops_with_idx:
            tracker.record[label] = {'idx': idx}
            if isinstance(op, Generator):
                yield from op
            else:
                yield op

        tracker.record.pop(label, None)

    elif mode == 'arguments-training-data':
        training_collector = kwargs['training_points_collector']
        externals: Dict[str, Any] = kwargs['externals']

        if label not in tracker.record:
            raise AutoPandasInversionFailedException("Could not find label {} in tracker".format(label))

        idx = tracker.record[label]['idx']
        # graph: RelationGraphChain = RelationGraphChain.init(spec.inputs, spec.output)
        graph: RelationGraphChain = RelationGraphChain.init(list(externals.values()), spec.output)
        graph.add_options(len(ops), picked=idx)
        encoding = graph.get_encoding()
        encoding['op_label'] = label
        training_collector[label] = encoding

        op = ops[idx]
        if isinstance(op, Generator):
            yield from op
        else:
            yield op

    elif mode == 'arguments-training-data-best-effort':
        raise NotImplementedError("Best-effort procedure not implemented for Chain")

    elif mode == 'inference':
        model_store: Dict[str, RelGraphInterface] = kwargs['model_store']
        prob_store: Dict[str, float] = kwargs['prob_store']
        externals: Dict[str, Any] = kwargs['externals']

        # graph: RelationGraphChain = RelationGraphChain.init(spec.inputs, spec.output)
        graph: RelationGraphChain = RelationGraphChain.init(list(externals.values()), spec.output)
        graph.add_options(len(ops), query=True)
        encoding = graph.get_encoding()
        encoding['op_label'] = label

        #  The inference in Chain returns a list of tuples (probability, choice_idx)
        inferred: List[Tuple[float, int]] = sorted(model_store[label].predict_graphs([encoding])[0],
                                                   key=lambda x: -x[0])
        for prob, idx in inferred:
            prob_store[label] = prob
            op = ops[idx]
            if isinstance(op, Generator):
                yield from op
            else:
                yield op


def Subsets(vals: Collection[Any], lengths: Iterable[Any] = None, lists: bool = False,
            spec: SearchSpec = None, depth: int = 1, mode: str = None, tracker: OpTracker = None,
            arg_name: str = None, identifier: str = None, **kwargs):
    label = 'subsets_' + arg_name + '_' + identifier

    if mode == 'exhaustive' or (mode == 'inference' and label not in kwargs['model_store']):
        if mode == 'inference':
            logger.warn("Did not find model for {}.{}".format(kwargs['func'], label), use_cache=True)

        if lengths is None:
            lengths = range(1, len(vals) + 1)

        vals = list(vals)
        vals = [val.val if isinstance(val, Value) else val for val in vals]
        for length in lengths:
            if lists:
                yield from map(list, itertools.combinations(vals, length))
            else:
                yield from itertools.combinations(vals, length)

    elif mode == 'training-data':
        #  This faces the same problem as Select
        if lengths is None:
            lengths = range(1, len(vals) + 1)

        lengths = list(lengths)
        if len(lengths) == 0:
            return

        #  We'll go over the lengths in random order, shuffle up the values, and yield systematically
        random.shuffle(lengths)
        vals = list(vals)
        vals = [val.val if isinstance(val, Value) else val for val in vals]
        for length in lengths:
            random.shuffle(vals)
            for subset in itertools.combinations(vals, length):
                if lists:
                    subset = list(subset)

                raw_subset = [i.val if isinstance(i, Value) else i for i in subset]
                tracker.record[label] = {'subset': raw_subset, 'length': len(subset)}
                yield subset

        tracker.record.pop(label, None)

    elif mode in ['arguments-training-data', 'arguments-training-data-best-effort']:
        training_collector = kwargs['training_points_collector']
        externals: Dict[str, Any] = kwargs['externals']
        vals = list(vals)

        #  TODO : Come up with a better more general solution
        randoms = [(idx, val.val) for idx, val in enumerate(vals) if isinstance(val, RandomColumn)]
        vals = [val.val if isinstance(val, Value) else val for val in vals]

        def raise_inversion_error():
            raise AutoPandasInversionFailedException("Could not invert generator for {} at {}".format(arg_name,
                                                                                                      label))

        if mode == 'arguments-training-data':
            if label not in tracker.record:
                raise AutoPandasInversionFailedException("Could not find label {} in tracker".format(label))

            target_length = tracker.record[label]['length']
            target_subset = tracker.record[label]['subset']

        else:
            training_spec: ArgTrainingSpec = spec
            target_subset = training_spec.args[arg_name]
            target_length = len(target_subset)

        if target_length > len(vals):
            raise_inversion_error()

        selected_indices: List[int] = []
        subset = []
        for target_val in target_subset:
            for idx, val in enumerate(vals):
                if Checker.check(val, target_val):
                    selected_indices.append(idx)
                    subset.append(val)
                    break
            else:
                # So that didn't work out... There was no value in the domain that was equal to the target val.
                # This can happen when random column names are generated. 

                if isinstance(target_val, str) and target_val.startswith("AUTOPANDAS_"):
                    if len(randoms) > 0:
                        #  Great, so we can assume it was one of these randoms and it should be correct in most cases
                        picked_idx = randoms[0][0]
                        selected_indices.append(picked_idx)
                        vals[picked_idx] = target_val
                        subset.append(target_val)
                        randoms = randoms[1:]

                    else:
                        raise_inversion_error()
                else:
                    raise_inversion_error()

        #  Providing (spec.inputs, spec.output) might not be appropriate for higher-depths
        # graph: RelationGraphSubsets = RelationGraphSubsets.init(spec.inputs, spec.output)
        graph: RelationGraphSubsets = RelationGraphSubsets.init(list(externals.values()), spec.output)
        graph.add_set(vals, selected_indices)

        encoding = graph.get_encoding()
        encoding['op_label'] = label
        training_collector[label] = encoding

        if lists:
            yield subset
        else:
            yield tuple(subset)

        return

    elif mode == 'inference':
        model_store: Dict[str, RelGraphInterface] = kwargs['model_store']
        prob_store: Dict[str, float] = kwargs['prob_store']
        externals: Dict[str, Any] = kwargs['externals']
        beam_search_k = kwargs['beam_search_k']
        vals = list(vals)
        vals = [val.val if isinstance(val, Value) else val for val in vals]
        if lengths is None:
            lengths = range(1, len(vals) + 1)

        lengths = set(lengths)

        if len(vals) == 0 or len(lengths) == 0:
            return

        # graph: RelationGraphSubsets = RelationGraphSubsets.init(spec.inputs, spec.output)
        graph: RelationGraphSubsets = RelationGraphSubsets.init(list(externals.values()), spec.output)
        graph.add_set(vals, query=True)

        encoding, reverse_mapping = graph.get_encoding(get_reverse_mapping=True)
        encoding['op_label'] = label
        encoding['raw_vals'] = vals

        #  The inference in Subset returns a list of tuples (discard prob, keep prob, idx)
        #  We now need to iterate over the subsets in the decreasing order of their joint probability
        inferred: List[Tuple[float, float, int]] = model_store[label].predict_graphs([encoding])[0]
        inferred = [(i[0], i[1], reverse_mapping[i[2]]) for i in inferred]

        def beam_search(items: List[Tuple[float, float, int]], width: int):
            beam: List[Tuple[float, List[int]]] = [(1.0, [])]
            for d_prob, k_prob, val_idx in items:
                new_beam = []
                for cum_prob, elems in beam:
                    new_beam.append((cum_prob * d_prob, elems[:]))
                    new_beam.append((cum_prob * k_prob, elems + [val_idx]))

                beam = list(reversed(sorted(new_beam)))[:width]

            yield from beam

        for prob, subset_indices in beam_search(inferred, width=beam_search_k):
            prob_store[label] = prob
            subset = tuple(vals[idx] for idx in sorted(subset_indices))
            if lists:
                subset = list(subset)

            yield subset


def OrderedSubsets(vals: Collection[Any], lengths: Iterable[Any] = None, lists: bool = False,
                   spec: SearchSpec = None, depth: int = 1, mode: str = None, tracker: OpTracker = None,
                   arg_name: str = None, identifier: str = None, **kwargs):
    label = 'orderedsubsets_' + arg_name + '_' + identifier

    if mode == 'exhaustive' or (mode == 'inference' and label not in kwargs['model_store']):
        if mode == 'inference':
            logger.warn("Did not find model for {}.{}".format(kwargs['func'], label), use_cache=True)

        if lengths is None:
            lengths = range(1, len(vals) + 1)

        vals = list(vals)
        vals = [val.val if isinstance(val, Value) else val for val in vals]
        for length in lengths:
            if lists:
                yield from map(list, itertools.permutations(vals, length))
            else:
                yield from itertools.permutations(vals, length)

    elif mode == 'training-data':
        #  This faces the same problem as Select
        if lengths is None:
            lengths = range(1, len(vals) + 1)

        lengths = list(lengths)
        if len(lengths) == 0:
            return

        #  We'll go over the lengths in random order, shuffle up the values, and yield systematically
        random.shuffle(lengths)
        vals = list(vals)
        vals = [val.val if isinstance(val, Value) else val for val in vals]
        for length in lengths:
            random.shuffle(vals)
            for subset in itertools.permutations(vals, length):
                if lists:
                    subset = list(subset)

                raw_subset = [i.val if isinstance(i, Value) else i for i in subset]
                tracker.record[label] = {'subset': raw_subset, 'length': len(subset)}
                yield subset

        tracker.record.pop(label, None)

    elif mode in ['arguments-training-data', 'arguments-training-data-best-effort']:
        training_collector = kwargs['training_points_collector']
        externals: Dict[str, Any] = kwargs['externals']
        vals = list(vals)

        #  TODO : Come up with a better more general solution
        randoms = [(idx, val.val) for idx, val in enumerate(vals) if isinstance(val, RandomColumn)]
        vals = [val.val if isinstance(val, Value) else val for val in vals]

        def raise_inversion_error():
            raise AutoPandasInversionFailedException("Could not invert generator for {} at {}".format(arg_name,
                                                                                                      label))

        if mode == 'arguments-training-data':
            if label not in tracker.record:
                raise AutoPandasInversionFailedException("Could not find label {} in tracker".format(label))

            target_length = tracker.record[label]['length']
            target_subset = tracker.record[label]['subset']

        else:
            training_spec: ArgTrainingSpec = spec
            target_subset = training_spec.args[arg_name]
            target_length = len(target_subset)

        if target_length > len(vals):
            raise_inversion_error()

        selected_indices: List[int] = []
        subset = []
        for target_val in target_subset:
            for idx, val in enumerate(vals):
                if Checker.check(val, target_val):
                    selected_indices.append(idx)
                    subset.append(val)
                    break
            else:
                # So that didn't work out... There was no value in the domain that was equal to the target val.
                # This can happen when random column names are generated. 

                if isinstance(target_val, str) and target_val.startswith("AUTOPANDAS_"):
                    if len(randoms) > 0:
                        #  Great, so we can assume it was one of these randoms and it should be correct in most cases
                        picked_idx = randoms[0][0]
                        selected_indices.append(picked_idx)
                        vals[picked_idx] = target_val
                        subset.append(target_val)
                        randoms = randoms[1:]

                    else:
                        raise_inversion_error()
                else:
                    raise_inversion_error()

        #  Providing (spec.inputs, spec.output) might not be appropriate for higher-depths
        # graph: RelationGraphSubsets = RelationGraphOrderedSubsets.init(spec.inputs, spec.output)
        graph: RelationGraphSubsets = RelationGraphOrderedSubsets.init(list(externals.values()), spec.output)
        graph.add_set(vals, selected_indices)

        encoding = graph.get_encoding()
        encoding['op_label'] = label
        training_collector[label] = encoding

        if lists:
            yield subset
        else:
            yield tuple(subset)

        return

    elif mode == 'inference':
        model_store: Dict[str, RelGraphInterface] = kwargs['model_store']
        prob_store: Dict[str, float] = kwargs['prob_store']
        externals: Dict[str, Any] = kwargs['externals']
        beam_search_k = kwargs['beam_search_k']
        vals = list(vals)
        vals = [val.val if isinstance(val, Value) else val for val in vals]
        if lengths is None:
            lengths = range(1, len(vals) + 1)

        lengths = set(lengths)

        if len(vals) == 0 or len(lengths) == 0:
            return

        # graph: RelationGraphSubsets = RelationGraphOrderedSubsets.init(spec.inputs, spec.output)
        graph: RelationGraphSubsets = RelationGraphOrderedSubsets.init(list(externals.values()), spec.output)
        graph.add_set(vals, query=True)

        encoding, reverse_mapping = graph.get_encoding(get_reverse_mapping=True)
        encoding['op_label'] = label
        encoding['raw_vals'] = vals

        inferred: List[List[Tuple[float, int]]] = model_store[label].predict_graphs([encoding])[0]
        for preds in inferred:
            for i in range(len(preds)):
                preds[i] = (preds[i][0], reverse_mapping[preds[i][1]])

        inferred = inferred[:len(vals) + 1]

        def beam_search(items: List[List[Tuple[float, int]]], width: int, num_elems: int):
            results: List[Tuple[float, List[int]]] = []
            beam: List[Tuple[float, List[int]]] = [(1.0, [])]
            for depth, preds in enumerate(items):
                new_beam: List[Tuple[float, List[int]]] = []
                for prob, val_idx in preds:
                    if val_idx == num_elems:
                        results.extend([(cum_prob * prob, elems[:]) for cum_prob, elems in beam
                                        if len(elems) in lengths])
                    else:
                        new_beam.extend([(cum_prob * prob, elems + [val_idx]) for cum_prob, elems in beam
                                         if val_idx not in elems])

                beam = list(reversed(sorted(new_beam)))[:width]

            yield from reversed(sorted(results))

        for prob, subset_indices in beam_search(inferred, width=beam_search_k, num_elems=len(vals)):
            prob_store[label] = prob
            subset = tuple(vals[idx] for idx in subset_indices)
            if lists:
                subset = list(subset)

            yield subset


def Product(*domains: Any, spec: SearchSpec = None, depth: int = 1, mode: str = None, tracker: OpTracker = None,
            arg_name: str = None, identifier: str = None, **kwargs):

    label = 'product_' + arg_name + '_' + identifier

    if mode == 'exhaustive' or (mode == 'inference' and label not in kwargs['model_store']):
        if mode == 'inference':
            logger.warn("Did not find model for {}.{}".format(kwargs['func'], label), use_cache=True)

        yield from itertools.product(*domains)

    elif mode == 'training-data':
        domains_with_idx = [list(enumerate(domain)) for domain in domains]
        for domain in domains_with_idx:
            random.shuffle(domain)

        for product_with_idx in itertools.product(*domains_with_idx):
            # https://stackoverflow.com/questions/12974474/how-to-unzip-a-list-of-tuples-into-individual-lists
            indices, product = list(zip(*product_with_idx))
            tracker.record[label] = {'indices': indices}
            yield product

    elif mode == 'arguments-training-data' or mode == 'arguments-training-data-best-effort':
        training_collector = kwargs['training_points_collector']
        externals: Dict[str, Any] = kwargs['externals']

        if label not in tracker.record:
            raise AutoPandasInversionFailedException("Could not find label {} in tracker".format(label))

        indices = tracker.record[label]['indices']
        domains = [list(domain) for domain in domains]
        selected = [domain[idx] for domain, idx in zip(domains, indices)]

        graph: RelationGraphProduct = RelationGraphProduct.init(list(externals.values()), spec.output)
        graph.add_iterables(domains, selected_indices=indices)
        encoding = graph.get_encoding()
        encoding['op_label'] = label
        training_collector[label] = encoding

        yield tuple(selected)
