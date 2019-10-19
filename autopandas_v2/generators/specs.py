import itertools
import collections
import random
import re

import dateutil
import pandas as pd
import numpy as np
from autopandas_v2.utils.datastructures import oset
from numpy import nan
from typing import Callable, List, Sequence, Union
from autopandas_v2.generators.compilation.stubs import _spec, signature, target, representation, inp_types, \
    out_types, inherit, arity
from autopandas_v2.generators.dsl.ops import Select, Ext, Choice, Chain, Subsets, OrderedSubsets, Product
from autopandas_v2.generators.dsl.values import Default, AnnotatedVal, Inactive, RandomColumn
from autopandas_v2.utils.types import DType, FType

pd_dfgroupby = pd.core.groupby.DataFrameGroupBy


# noinspection PyMethodParameters
class df:
    # ----------------------------------------------------------------------- #
    #  Attributes
    # ----------------------------------------------------------------------- #
    #  These are not callable, but its values are sometimes useful, and hence
    #  they are included in synthesis
    # ----------------------------------------------------------------------- #
    @signature('DataFrame.index')
    @target(lambda self: self.index)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.Index), DType(list), DType(tuple)])
    def index():
        _self = Ext(DType(pd.DataFrame))

    @signature('DataFrame.columns')
    @target(lambda self: self.columns)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.Index), DType(list), DType(tuple)])
    def columns():
        _self = Ext(DType(pd.DataFrame))

    @signature('DataFrame.dtypes')
    @target(lambda self: self.dtypes)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.Series)])
    def dtypes():
        _self = Ext(DType(pd.DataFrame))

    @signature('DataFrame.ftypes')
    @target(lambda self: self.ftypes)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.Series)])
    def ftypes():
        _self = Ext(DType(pd.DataFrame))

    @signature('DataFrame.values')
    @target(lambda self: self.values)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(np.ndarray)])
    def values():
        _self = Ext(DType(pd.DataFrame))

    @signature('DataFrame.axes')
    @target(lambda self: self.axes)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(list), DType(tuple)])
    def axes():
        _self = Ext(DType(pd.DataFrame))

    @signature('DataFrame.ndim')
    @target(lambda self: self.ndim)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(int), DType(np.integer)])
    def ndim():
        _self = Ext(DType(pd.DataFrame))

    @signature('DataFrame.size')
    @target(lambda self: self.size)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(int), DType(np.integer)])
    def size():
        _self = Ext(DType(pd.DataFrame))

    @signature('DataFrame.shape')
    @target(lambda self: self.shape)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(list), DType(tuple)])
    def shape():
        _self = Ext(DType(pd.DataFrame))

    @signature('DataFrame.T')
    @target(lambda self: self.T)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def T():
        _self = Ext(DType(pd.DataFrame))

    # ----------------------------------------------------------------------- #
    #  Methods
    # ----------------------------------------------------------------------- #

    # ------------------------------------------------------------------- #
    #  Attributes & Underlying Data
    # ------------------------------------------------------------------- #

    @signature('DataFrame.as_matrix(self, columns=None)')
    @target(pd.DataFrame.as_matrix)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(np.ndarray)])
    def as_matrix():
        _self = Ext(DType(pd.DataFrame))
        _columns = Chain(Default(None),
                         OrderedSubsets(_self.columns, lists=True))

    @signature('DataFrame.get_dtype_counts(self)')
    @target(pd.DataFrame.get_dtype_counts)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.Series)])
    def get_dtype_counts():
        _self = Ext(DType(pd.DataFrame))

    @inherit('df.get_dtype_counts')
    @signature('DataFrame.get_ftype_counts(self)')
    @target(pd.DataFrame.get_ftype_counts)
    def get_ftype_counts():
        pass

    @signature('DataFrame.select_dtypes(self, include=None, exclude=None)')
    @target(pd.DataFrame.select_dtypes)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def select_dtypes():
        def arg_include(v_self: pd.DataFrame):
            dtypes = set(map(str, v_self.dtypes))
            lengths = None
            if _spec.depth == _spec.max_depth:
                output: pd.DataFrame = _spec.output
                dtypes = set(map(str, output.dtypes))
                lengths = [len(dtypes)]

            for val in Subsets(vals=dtypes, lengths=lengths, lists=True):
                yield AnnotatedVal(val, cost=0)

        def arg_exclude_none(v_include):
            if v_include is not None:
                yield Default(None)

        def arg_exclude(v_self: pd.DataFrame, v_include):
            dtypes = (set(map(str, v_self.dtypes)) - set((v_include or [])))
            lengths = None
            if _spec.depth == _spec.max_depth:
                output: pd.DataFrame = _spec.output
                dtypes = (dtypes - set(map(str, output.dtypes)))
                lengths = [len(dtypes)]

            for val in Subsets(vals=dtypes, lengths=lengths, lists=True):
                yield AnnotatedVal(val, cost=0)

        _self = Ext(DType(pd.DataFrame))
        _include = Chain(Default(None), arg_include(_self))
        _exclude = Chain(arg_exclude_none(_include), arg_exclude(_self, _include))

        # ------------------------------------------------------------------- #
        #  Conversion
        # ------------------------------------------------------------------- #

    @signature("DataFrame.astype(self, dtype, copy=True, errors='raise')")
    @target(pd.DataFrame.astype)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def astype():

        def arg_astype_partial(v_self):
            if _spec.depth == _spec.max_depth:
                v_self: pd.DataFrame = v_self
                output: pd.DataFrame = _spec.output
                try:
                    if set(output.columns).issubset(set(v_self.columns)):
                        yield dict(output.dtypes)
                except:
                    pass

        _self = Ext(DType(pd.DataFrame))
        _dtype = Chain(Ext(DType([str, dict])), arg_astype_partial(_self))
        _errors = Choice(Default('raise'), 'ignore')

    @signature('DataFrame.isna(self)')
    @target(pd.DataFrame.isna)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def isna():
        _self = Ext(DType(pd.DataFrame))

    @signature('DataFrame.notna(self)')
    @target(pd.DataFrame.notna)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def notna():
        _self = Ext(DType(pd.DataFrame))

        # ------------------------------------------------------------------- #
        #  Indexing & Iteration
        # ------------------------------------------------------------------- #

    @signature('DataFrame.head(self, n=5)')
    @target(pd.DataFrame.head)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def head():
        def arg_head_partial(v_self: pd.DataFrame):
            if _spec.depth == _spec.max_depth:
                output: pd.DataFrame = _spec.output
                yield AnnotatedVal(output.shape[0], cost=0)

            yield from Select(list(range(1, v_self.shape[0]+1)))

        _self = Ext(DType(pd.DataFrame))
        _n = Chain(Default(5), Ext(DType(int)), arg_head_partial(_self))

    @signature('DataFrame.at.__getitem__(self, key)')
    @target(lambda self, key: self.at[key])
    @inp_types([DType(pd.DataFrame)])
    def at_getitem():
        def arg_key(v_self: pd.DataFrame):
            indices = v_self.index
            columns = v_self.columns
            for key in Product(indices, columns):
                yield AnnotatedVal(key, cost=5)

        _self = Ext(DType(pd.DataFrame))
        _key = Chain(Ext(DType([list, tuple])), arg_key(_self))

    @signature('DataFrame.iat.__getitem__(self, key)')
    @target(lambda self, key: self.iat[key])
    @inp_types([DType(pd.DataFrame)])
    def iat_getitem():
        def arg_key(v_self: pd.DataFrame):
            indices = range(len(v_self.index))
            columns = range(len(v_self.columns))
            for key in Product(indices, columns):
                yield AnnotatedVal(key, cost=5)

        _self = Ext(DType(pd.DataFrame))
        _key = Chain(Ext(DType([list, tuple])), arg_key(_self))

    @signature('DataFrame.loc.__getitem__(self, key)')
    @target(lambda self, key: self.loc[key])
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame), DType(pd.Series)])
    def loc_getitem():

        def get_slice_or_list(o_val: List, i_val: List):
            i_s_idx = i_val.index(o_val[0])
            i_e_idx = i_val.index(o_val[(- 1)])
            if (abs((i_e_idx - i_s_idx)) + 1) == len(o_val):
                return slice(o_val[0], o_val[(- 1)], (1 if (i_s_idx <= i_e_idx) else (- 1)))
            else:
                return o_val

        def arg_key(v_self):
            self_indices = list(v_self.index)
            self_cols = list(v_self.columns)
            if _spec.depth == _spec.max_depth:
                output = _spec.output
                if isinstance(output, pd.DataFrame):
                    o_indices = list(output.index)
                    o_cols = list(output.columns)
                    if (not o_indices) or (not set(o_indices).issubset(set(self_indices))):
                        return
                    if (not o_cols) or (not set(o_cols).issubset(set(self_cols))):
                        return
                    axis0 = get_slice_or_list(o_indices, self_indices)
                    axis1 = get_slice_or_list(o_cols, self_cols)
                    yield AnnotatedVal((axis0, axis1), cost=2)
                else:
                    for (i, idx) in enumerate(v_self.index):
                        yield (idx, slice(None))
                    for (i, idx) in enumerate(v_self.columns):
                        yield (idx, slice(None))
                return

            for dir1 in Choice(1, -1):
                for dir2 in Choice(1, -1):
                    for i0, i1 in Select(list(itertools.combinations(self_indices, 2))):
                        for j0, j1 in Select(list(itertools.combinations(self_cols, 2))):
                            s1 = slice(i0, i1, 1) if dir1 == 1 else slice(i1, i0, -1)
                            s2 = slice(j0, j1, 1) if dir2 == 1 else slice(j1, j0, -1)
                            yield AnnotatedVal((s1, s2), cost=5)

        _self = Ext(DType(pd.DataFrame))
        _key = Chain(Ext(DType([Callable, list, tuple, pd.Series])), arg_key(_self))

    @signature('DataFrame.iloc.__getitem__(self, key)')
    @target(lambda self, key: self.iloc[key])
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame), DType(pd.Series)])
    def iloc_getitem():

        def get_slice_or_list(o_val: List, i_val: List):
            i_s_idx = i_val.index(o_val[0])
            i_e_idx = i_val.index(o_val[(- 1)])
            if (abs((i_e_idx - i_s_idx)) + 1) == len(o_val):
                sgn = (1 if (i_s_idx <= i_e_idx) else (- 1))
                return slice(i_s_idx, (i_e_idx + sgn), sgn)
            else:
                return [i_val.index(k) for k in o_val]

        def arg_key(v_self):
            self_indices = list(v_self.index)
            self_cols = list(v_self.columns)
            if _spec.depth == _spec.max_depth:
                output = _spec.output
                if isinstance(output, pd.DataFrame):
                    o_indices = list(output.index)
                    o_cols = list(output.columns)
                    if (not o_indices) or (not set(o_indices).issubset(set(self_indices))):
                        return
                    if (not o_cols) or (not set(o_cols).issubset(set(self_cols))):
                        return
                    axis0 = get_slice_or_list(o_indices, self_indices)
                    axis1 = get_slice_or_list(o_cols, self_cols)
                    yield AnnotatedVal((axis0, axis1), cost=2)
                else:
                    for (i, idx) in enumerate(v_self.index):
                        yield (idx, slice(None))
                    for (i, idx) in enumerate(v_self.columns):
                        yield (idx, slice(None))
                return

            ri_indices = list(range(len(self_indices)))
            ri_cols = list(range(len(self_cols)))

            for dir1 in Choice(1, -1):
                for dir2 in Choice(1, -1):
                    for i0, i1 in Select(list(itertools.combinations(ri_indices, 2))):
                        for j0, j1 in Select(list(itertools.combinations(ri_cols, 2))):
                            s1 = slice(i0, i1, 1) if dir1 == 1 else slice(i1, i0, -1)
                            s2 = slice(j0, j1, 1) if dir2 == 1 else slice(j1, j0, -1)
                            yield AnnotatedVal((s1, s2), cost=5)

        _self = Ext(DType(pd.DataFrame))
        _key = Chain(Ext(DType([Callable, list, tuple])), arg_key(_self))

    @signature('DataFrame.lookup(self, row_labels, col_labels)')
    @target(pd.DataFrame.lookup)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(np.ndarray)])
    def lookup():
        _self = Ext(DType(pd.DataFrame))
        _row_labels = OrderedSubsets(vals=_self.index,
                                     lengths=range(1, min(len(_self.columns), len(_self.index)) + 1), lists=True)
        _col_labels = OrderedSubsets(vals=_self.columns,
                                     lengths=[len(_row_labels)], lists=True)

    @inherit('df.head')
    @signature('DataFrame.tail(self, n=5)')
    @target(pd.DataFrame.tail)
    def tail():
        pass

    @signature('DataFrame.xs(self, key, axis=0, level=None, drop_level=True)')
    @target(pd.DataFrame.xs)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame), DType(pd.Series)])
    def xs():
        def arg_level(v_self: pd.DataFrame, v_axis: int):
            if _spec.depth == _spec.max_depth:
                output = _spec.output
                try:
                    if isinstance(output, pd.Series):
                        if (all(output.index == v_self.index) and (v_axis == 0)) or (
                                all(output.index == v_self.columns) and (v_axis == 1)):
                            return
                    elif isinstance(output, pd.DataFrame):
                        if (all((output.index == v_self.index)) and (v_axis == 0)) or (
                                all((output.columns == v_self.columns)) and (v_axis == 1)):
                            return
                except StopIteration:
                    return
                except:
                    pass

            src = (v_self.index if (v_axis == 0) else v_self.columns)
            if src.nlevels == 1:
                yield Default(None)
            else:
                yield from Subsets(vals=range(src.nlevels),
                                   lengths=range(1, (src.nlevels + 1)))

        def arg_key(v_self: pd.DataFrame, v_axis: int, v_level):
            src = (v_self.index if (v_axis == 0) else v_self.columns)
            if src.nlevels == 1:
                yield from Select(list(src))
            else:
                level_prods = [src.levels[i] for i in v_level]
                yield from map(list, Product(*level_prods))

        _self = Ext(DType(pd.DataFrame))
        _drop_level = Choice(Default(True), False)
        _axis = Choice(Default(0), 1)
        _level = arg_level(_self, _axis)
        _key = arg_key(_self, _axis, _level)

    @signature('DataFrame.isin(self, values)')
    @target(pd.DataFrame.isin)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def isin():
        _self = Ext(DType(pd.DataFrame))
        _values = Ext(DType([list, tuple, pd.Series, dict, pd.DataFrame]))

    @signature("DataFrame.where(self, cond, other=nan, inplace=False, axis=None, level=None, errors='raise', "
               "try_cast=False, raise_on_error=None)")
    @target(pd.DataFrame.where)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def where():
        _self = Ext(DType(pd.DataFrame))
        _cond = Ext(DType([Sequence, pd.DataFrame, Callable]))
        _other = Ext(DType([Sequence, pd.DataFrame, Callable]))
        _errors = Choice(Default('raise'), 'ignore')

    @inherit('df.where')
    @signature("DataFrame.mask(self, cond, other=nan, inplace=False, axis=None, level=None, errors='raise', "
               "try_cast=False, raise_on_error=None)")
    @target(pd.DataFrame.mask)
    def mask():
        pass

    @signature('DataFrame.query(self, expr, inplace=False)')
    @target(pd.DataFrame.query)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def query():
        _self = Ext(DType(pd.DataFrame))
        _expr = Ext(DType(str))

    @signature('DataFrame.__getitem__(self, key)')
    @target(pd.DataFrame.__getitem__)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame), DType(pd.Series)])
    def __getitem__():
        def arg_key(v_self: pd.DataFrame):
            lengths = None
            if _spec.depth == _spec.max_depth:
                if isinstance(_spec.output, pd.DataFrame):
                    yield AnnotatedVal(list(_spec.output.columns), cost=1)
                else:
                    for col in Select(v_self.columns):
                        yield AnnotatedVal(col, cost=1)

                return

            yield from map(lambda x: AnnotatedVal(x, cost=1),
                           Chain(Select(v_self.columns),
                                 OrderedSubsets(vals=v_self.columns, lengths=lengths, lists=True)))

        _self = Ext(DType(pd.DataFrame))
        _key = Chain(arg_key(_self), Ext(DType([pd.Series, list, tuple])))

        # ------------------------------------------------------------------- #
        #  Binary Operator Functions
        # ------------------------------------------------------------------- #

    @signature("DataFrame.add(self, other, axis='columns', level=None, fill_value=None)")
    @target(pd.DataFrame.add)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def add():
        def arg_axis(v_other):
            if isinstance(v_other, pd.Series):
                yield 'index'

        def arg_level(v_self, v_axis):
            index: Union[(pd.Index, pd.MultiIndex)] = (v_self.index if (v_axis == 0) else v_self.columns)
            levels = [(index.names[i] or i) for i in range(index.nlevels)]
            yield from Select(levels)

        _self = Ext(DType(pd.DataFrame))
        _fill_value = Chain(Default(None), Ext(DType(float)))
        _other = Ext(DType(object))
        _axis = Chain(Default('columns'), arg_axis(_other))
        _level = Chain(Default(None), arg_level(_self, _axis))

    @inherit('df.add')
    @signature("DataFrame.sub(self, other, axis='columns', level=None, fill_value=None)")
    @target(pd.DataFrame.sub)
    def sub():
        pass

    @inherit('df.add')
    @signature("DataFrame.mul(self, other, axis='columns', level=None, fill_value=None)")
    @target(pd.DataFrame.mul)
    def mul():
        def arg_axis(v_self, v_other):
            #  Only return something if all the columns
            #  are of integer types. Otherwise things can
            #  get nasty and cause memory issues
            #  For example 1000 * 1000 * 1000 * "abcd"
            #  would wreak havoc on the system
            if len(v_self.select_dtypes(include=np.number).columns) != len(v_self.columns):
                return

            if isinstance(v_other, pd.DataFrame):
                if len(v_other.select_dtypes(include=np.number).columns) != len(v_other.columns):
                    return

            elif isinstance(v_other, pd.Series):
                if not issubclass(v_other.dtype.type, np.number):
                    return

            elif isinstance(v_other, str):
                return

            if isinstance(v_other, pd.Series):
                yield from Choice(Default('columns'), 'index')
            else:
                yield Default('columns')

        _self = Ext(DType(pd.DataFrame))
        _other = Ext(DType(object))
        _axis = arg_axis(_self, _other)

    @inherit('df.add')
    @signature("DataFrame.div(self, other, axis='columns', level=None, fill_value=None)")
    @target(pd.DataFrame.div)
    def div():
        pass

    @inherit('df.add')
    @signature("DataFrame.truediv(self, other, axis='columns', level=None, fill_value=None)")
    @target(pd.DataFrame.truediv)
    def truediv():
        pass

    @inherit('df.add')
    @signature("DataFrame.floordiv(self, other, axis='columns', level=None, fill_value=None)")
    @target(pd.DataFrame.floordiv)
    def floordiv():
        pass

    @inherit('df.add')
    @signature("DataFrame.mod(self, other, axis='columns', level=None, fill_value=None)")
    @target(pd.DataFrame.mod)
    def mod():
        pass

    @inherit('df.add')
    @signature("DataFrame.pow(self, other, axis='columns', level=None, fill_value=None)")
    @target(pd.DataFrame.pow)
    def pow():
        pass

    @inherit('df.add')
    @signature("DataFrame.radd(self, other, axis='columns', level=None, fill_value=None)")
    @target(pd.DataFrame.radd)
    def radd():
        pass

    @inherit('df.add')
    @signature("DataFrame.rsub(self, other, axis='columns', level=None, fill_value=None)")
    @target(pd.DataFrame.rsub)
    def rsub():
        pass

    @inherit('df.add')
    @signature("DataFrame.rmul(self, other, axis='columns', level=None, fill_value=None)")
    @target(pd.DataFrame.rmul)
    def rmul():
        pass

    @inherit('df.add')
    @signature("DataFrame.rdiv(self, other, axis='columns', level=None, fill_value=None)")
    @target(pd.DataFrame.rdiv)
    def rdiv():
        pass

    @inherit('df.add')
    @signature("DataFrame.rtruediv(self, other, axis='columns', level=None, fill_value=None)")
    @target(pd.DataFrame.rtruediv)
    def rtruediv():
        pass

    @inherit('df.add')
    @signature("DataFrame.rfloordiv(self, other, axis='columns', level=None, fill_value=None)")
    @target(pd.DataFrame.rfloordiv)
    def rfloordiv():
        pass

    @inherit('df.add')
    @signature("DataFrame.rmod(self, other, axis='columns', level=None, fill_value=None)")
    @target(pd.DataFrame.rmod)
    def rmod():
        pass

    @inherit('df.add')
    @signature("DataFrame.rpow(self, other, axis='columns', level=None, fill_value=None)")
    @target(pd.DataFrame.rpow)
    def rpow():
        pass

    @inherit('df.add')
    @signature("DataFrame.lt(self, other, axis='columns', level=None)")
    @target(pd.DataFrame.lt)
    def lt():
        pass

    @inherit('df.add')
    @signature("DataFrame.gt(self, other, axis='columns', level=None)")
    @target(pd.DataFrame.gt)
    def gt():
        pass

    @inherit('df.add')
    @signature("DataFrame.le(self, other, axis='columns', level=None)")
    @target(pd.DataFrame.le)
    def le():
        pass

    @inherit('df.add')
    @signature("DataFrame.ge(self, other, axis='columns', level=None)")
    @target(pd.DataFrame.ge)
    def ge():
        pass

    @inherit('df.add')
    @signature("DataFrame.ne(self, other, axis='columns', level=None)")
    @target(pd.DataFrame.ne)
    def ne():
        pass

    @inherit('df.add')
    @signature("DataFrame.eq(self, other, axis='columns', level=None)")
    @target(pd.DataFrame.eq)
    def eq():
        pass

    @signature('DataFrame.combine(self, other, func, fill_value=None, overwrite=True)')
    @target(pd.DataFrame.combine)
    @inp_types([DType(pd.DataFrame), DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def combine():
        _self = Ext(DType(pd.DataFrame))
        _other = Ext(DType(pd.DataFrame))
        _func = Ext(DType(Callable))
        _fill_value = Chain(Default(None), Ext(FType(np.isscalar)))
        _overwrite = Choice(Default(True), False)

    @signature('DataFrame.combine_first(self, other)')
    @target(pd.DataFrame.combine_first)
    @inp_types([DType(pd.DataFrame), DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def combine_first():
        _self = Ext(DType(pd.DataFrame))
        _other = Ext(DType(pd.DataFrame))

        # ------------------------------------------------------------------- #
        #  Function application, GroupBy & Window
        # ------------------------------------------------------------------- #

    @signature('DataFrame.apply(self, func, axis=0, broadcast=False, raw=False, reduce=None)')
    @target(pd.DataFrame.apply)
    @inp_types([DType(pd.DataFrame), DType(Callable)])
    @out_types([DType(pd.DataFrame), DType(pd.Series)])
    def apply():
        _self = Ext(DType(pd.DataFrame))
        _func = Ext(DType(Callable))
        _axis = Choice(Default(0), 1)
        _broadcast = Choice(Default(False), True)
        _raw = Choice(Default(False), True)

    @signature('DataFrame.applymap(self, func)')
    @target(pd.DataFrame.applymap)
    @inp_types([DType(pd.DataFrame), DType(Callable)])
    @out_types([DType(pd.DataFrame), DType(pd.Series)])
    def applymap():
        _self = Ext(DType(pd.DataFrame))
        _func = Ext(DType(Callable))

    @signature('DataFrame.agg(self, func, axis=0)')
    @target(pd.DataFrame.agg)
    @inp_types([DType(pd.DataFrame), DType([Callable, str, dict, list, tuple])])
    @out_types([DType(pd.DataFrame)])
    def agg():
        _self = Ext(DType(pd.DataFrame))
        _func = Ext(DType([Callable, str, dict, list, tuple]))
        _axis = Choice(Default(0), 1)

    @signature('DataFrame.transform(self, func)')
    @target(pd.DataFrame.transform)
    @inp_types([DType(pd.DataFrame), DType([Callable, str, dict, list, tuple])])
    @out_types([DType(pd.DataFrame)])
    def transform():
        _self = Ext(DType(pd.DataFrame))
        _func = Ext(DType([Callable, str, dict, list, tuple]))

    @signature(
        'DataFrame.groupby(self, by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=False)')
    @target(pd.DataFrame.groupby)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd_dfgroupby)])
    def groupby():
        def arg_level(v_self: pd.DataFrame, v_axis):
            index = (v_self.index if (v_axis == 0) else v_self.columns)
            yield from Chain(Select(range(0, index.nlevels - 1)),
                             OrderedSubsets(vals=range(index.nlevels),
                                            lengths=range(2, (index.nlevels + 1)), lists=True))

        def arg_by(v_level, v_self, v_axis):
            if v_level is not None:
                yield Default(None)
                return

            yield from Chain(arg_by_index(v_self, v_axis),
                             arg_by_ext(v_self, v_axis))

        def arg_by_index(v_self, v_axis):
            if v_axis != 0:
                return

            cols = list(v_self.columns)
            index = v_self.index
            index_cols = [index.names[i] for i in range(index.nlevels) if index.names[i] is not None]
            yield from Subsets(vals=cols + list(index_cols), lists=True)

        def arg_by_ext(v_self, v_axis):
            idx_len = v_self.shape[0] if v_axis == 0 else v_self.shape[1]
            yield from Ext(DType([pd.Series, list, tuple, dict, np.ndarray]),
                           constraint=lambda x: len(x) == idx_len)

        _self = Ext(DType(pd.DataFrame))
        _axis = Choice(Default(0), 1)
        _sort = Choice(Default(True), False)
        _as_index = Choice(Default(True), False)
        _level = Chain(Default(None), arg_level(_self, _axis))
        _by = arg_by(_level, _self, _axis)

        # ------------------------------------------------------------------- #
        #  Computations/Descriptive Stats
        # ------------------------------------------------------------------- #

    @signature('DataFrame.abs(self)')
    @target(pd.DataFrame.abs)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def abs():
        _self = Ext(DType(pd.DataFrame))

    @signature('DataFrame.all(self, axis=None, bool_only=None, skipna=None, level=None)')
    @target(pd.DataFrame.all)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame), DType(pd.Series)])
    def all():

        def arg_level(v_self: pd.DataFrame, v_axis: int):
            index = (v_self.index if (v_axis == 0) else v_self.columns)
            levels = [(index.names[i] or i) for i in range(index.nlevels)]
            yield from OrderedSubsets(vals=levels, lists=True)

        _self = Ext(DType(pd.DataFrame))
        _axis = Choice(Default(0), 1)
        _bool_only = Choice(Default(None), True, False)
        _skipna = Choice(Default(True), False)
        _level = Chain(Default(None), arg_level(_self, _axis))

    @inherit('df.all')
    @signature('DataFrame.any(self, axis=None, bool_only=None, skipna=None, level=None)')
    @target(pd.DataFrame.any)
    def any():
        pass

    @signature('DataFrame.clip(self, lower=None, upper=None, axis=None, inplace=False)')
    @target(pd.DataFrame.clip)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def clip():
        _axis = Choice(Default(0), 1)
        _lower = Chain(Default(None), Ext(DType(
            [float, np.dtype('float').type, np.dtype('float32').type, np.dtype('float64').type, Sequence, pd.Series])))
        _upper = Chain(Default(None), Ext(DType(
            [float, np.dtype('float').type, np.dtype('float32').type, np.dtype('float64').type, Sequence, pd.Series])))
        _self = Ext(DType(pd.DataFrame))

    @signature('DataFrame.clip_lower(self, threshold, axis=None, inplace=False)')
    @target(pd.DataFrame.clip_lower)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def clip_lower():
        _axis = Choice(Default(0), 1)
        _threshold = Ext(DType(
            [float, np.dtype('float').type, np.dtype('float32').type, np.dtype('float64').type, Sequence, pd.Series]))
        _self = Ext(DType(pd.DataFrame))

    @inherit('df.clip_lower')
    @signature('DataFrame.clip_upper(self, threshold, axis=None, inplace=False)')
    @target(pd.DataFrame.clip_upper)
    def clip_upper():
        pass

    @signature("DataFrame.corr(self, method='pearson', min_periods=1)")
    @target(pd.DataFrame.corr)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def corr():
        _min_periods = Chain(Default(1), Ext(DType(int)))
        _self = Ext(DType(pd.DataFrame))
        _method = Choice(Default('pearson'), 'kendall', 'spearman')

    @signature('DataFrame.corrwith(self, other, axis=0, drop=False)')
    @target(pd.DataFrame.corrwith)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.Series)])
    def corrwith():
        _self = Ext(DType(pd.DataFrame))
        _other = Ext(DType(pd.DataFrame))
        _drop = Choice(Default(False), True)
        _axis = Choice(Default(0), 1)

    @signature('DataFrame.count(self, axis=0, level=None, numeric_only=False)')
    @target(pd.DataFrame.count)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.Series), DType(pd.DataFrame)])
    def count():

        def arg_level(v_self: pd.DataFrame, v_axis: int):
            index = (v_self.index if (v_axis == 0) else v_self.columns)
            levels = [(index.names[i] or i) for i in range(index.nlevels)]
            for perm in OrderedSubsets(vals=levels, lists=True):
                yield perm

        _self = Ext(DType(pd.DataFrame))
        _axis = Choice(Default(0), 1)
        _numeric_only = Choice(Default(False), True)
        _level = Chain(Default(None), arg_level(_self, _axis))

    @signature('DataFrame.cov(self, min_periods=None)')
    @target(pd.DataFrame.cov)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def cov():
        _min_periods = Chain(Default(None), Ext(DType(int)))
        _self = Ext(DType(pd.DataFrame))

    @signature('DataFrame.cummax(self, axis=None, skipna=True)')
    @target(pd.DataFrame.cummax)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.Series), DType(pd.DataFrame)])
    def cummax():
        _self = Ext(DType(pd.DataFrame))
        _axis = Choice(Default(0), 1)
        _skipna = Choice(Default(True), False)

    @inherit('df.cummax')
    @signature('DataFrame.cummin(self, axis=None, skipna=True)')
    @target(pd.DataFrame.cummin)
    def cummin():
        pass

    @inherit('df.cummax')
    @signature('DataFrame.cumprod(self, axis=None, skipna=True)')
    @target(pd.DataFrame.cumprod)
    def cumprod():

        def arg_axis(v_self):
            #  Only return something if all the columns
            #  are of integer types. Otherwise things can
            #  get nasty and cause memory issues
            #  For example 1000 * 1000 * 1000 * "abcd"
            #  would wreak havoc on the system
            if len(v_self.select_dtypes(include=np.number).columns) == len(v_self.columns):
                yield from Choice(Default(0), 1)

        _self = Ext(DType(pd.DataFrame))
        _axis = arg_axis(_self)

    @inherit('df.cummax')
    @signature('DataFrame.cumsum(self, axis=None, skipna=True)')
    @target(pd.DataFrame.cumsum)
    def cumsum():
        pass

    @signature('DataFrame.diff(self, periods=1, axis=0)')
    @target(pd.DataFrame.diff)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def diff():
        _periods = Chain(Default(1), Ext(DType(int)))
        _self = Ext(DType(pd.DataFrame))
        _axis = Choice(Default(0), 1)

    @signature('DataFrame.eval(self, expr, inplace=False)')
    @target(pd.DataFrame.eval)
    @inp_types([DType(pd.DataFrame)])
    def eval():
        _self = Ext(DType(pd.DataFrame))
        _expr = Ext(DType(str))

    @signature('DataFrame.kurt(self, axis=None, skipna=None, level=None, numeric_only=None)')
    @target(pd.DataFrame.kurt)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.Series), DType(pd.DataFrame)])
    def kurt():
        def arg_level(v_self: pd.DataFrame, v_axis: int):
            index = (v_self.index if (v_axis == 0) else v_self.columns)
            levels = [(index.names[i] or i) for i in range(index.nlevels)]
            yield from OrderedSubsets(vals=levels, lists=True)

        _self = Ext(DType(pd.DataFrame))
        _axis = Choice(Default(0), 1)
        _level = Chain(Default(None), arg_level(_self, _axis))
        _skipna = Choice(Default(True), False)
        _numeric_only = Choice(Default(None), True, False)

    @inherit('df.kurt')
    @signature('DataFrame.mad(self, axis=None, skipna=None, level=None)')
    @target(pd.DataFrame.mad)
    def mad():
        pass

    @inherit('df.kurt')
    @signature('DataFrame.max(self, axis=None, skipna=None, level=None, numeric_only=None)')
    @target(pd.DataFrame.max)
    def max():
        pass

    @inherit('df.kurt')
    @signature('DataFrame.mean(self, axis=None, skipna=None, level=None, numeric_only=None)')
    @target(pd.DataFrame.mean)
    def mean():
        pass

    @inherit('df.kurt')
    @signature('DataFrame.median(self, axis=None, skipna=None, level=None, numeric_only=None)')
    @target(pd.DataFrame.median)
    def median():
        pass

    @inherit('df.kurt')
    @signature('DataFrame.min(self, axis=None, skipna=None, level=None, numeric_only=None)')
    @target(pd.DataFrame.min)
    def min():
        pass

    @signature('DataFrame.mode(self, axis=0, numeric_only=False)')
    @target(pd.DataFrame.mode)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def mode():
        _self = Ext(DType(pd.DataFrame))
        _axis = Choice(Default(0), 1)
        _numeric_only = Choice(Default(False), True)

    @signature("DataFrame.pct_change(self, periods=1, fill_method='pad', limit=None, freq=None)")
    @target(pd.DataFrame.pct_change)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def pct_change():
        _periods = Chain(Default(1), Ext(DType(int)))
        _limit = Chain(Default(None), Ext(DType(int)))
        _self = Ext(DType(pd.DataFrame))

    @signature('DataFrame.prod(self, axis=None, skipna=None, level=None, numeric_only=None, min_count=0)')
    @target(pd.DataFrame.prod)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.Series), DType(pd.DataFrame)])
    def prod():

        def arg_level(v_self: pd.DataFrame, v_axis: int):
            index = (v_self.index if (v_axis == 0) else v_self.columns)
            levels = [(index.names[i] or i) for i in range(index.nlevels)]
            yield from OrderedSubsets(vals=levels, lists=True)

        _self = Ext(DType(pd.DataFrame))
        _axis = Choice(Default(0), 1)
        _skipna = Choice(Default(True), False)
        #  Only return something if all the columns
        #  are of integer types. Otherwise things can
        #  get nasty and cause memory issues
        #  For example 1000 * 1000 * 1000 * "abcd"
        #  would wreak havoc on the system
        _numeric_only = Choice(Default(True))
        _level = Chain(Default(None), arg_level(_self, _axis))
        _min_count = Chain(Ext(DType(int)), Choice(Default(0), 1))

    @signature("DataFrame.quantile(self, q=0.5, axis=0, numeric_only=True, interpolation='linear')")
    @target(pd.DataFrame.quantile)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.Series), DType(pd.DataFrame)])
    def quantile():
        _q = Chain(Default(0.5), Ext(DType([float, Sequence])))
        _self = Ext(DType(pd.DataFrame))
        _axis = Choice(Default(0), 1)
        _numeric_only = Choice(Default(True), False)
        _interpolation = Choice(Default('linear'), 'lower', 'higher', 'midpoint', 'nearest')

    @signature(
        "DataFrame.rank(self, axis=0, method='average', numeric_only=None, na_option='keep', ascending=True, pct=False)")
    @target(pd.DataFrame.rank)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def rank():
        _self = Ext(DType(pd.DataFrame))
        _axis = Choice(Default(0), 1)
        _method = Choice(Default('average'), 'min', 'max', 'first', 'dense')
        _na_option = Choice(Default('keep'), 'top', 'bottom')
        _numeric_only = Choice(Default(None), True, False)
        _ascending = Choice(Default(True), False)
        _pct = Choice(Default(False), True)

    @signature('DataFrame.round(self, decimals=0)')
    @target(pd.DataFrame.round)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def round():
        _decimals = Chain(Default(0), Ext(DType([int, dict, pd.Series])))
        _self = Ext(DType(pd.DataFrame))

    @signature('DataFrame.sem(self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None)')
    @target(pd.DataFrame.sem)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.Series), DType(pd.DataFrame)])
    def sem():

        def arg_level(v_self: pd.DataFrame, v_axis: int):
            index = (v_self.index if (v_axis == 0) else v_self.columns)
            levels = [(index.names[i] or i) for i in range(index.nlevels)]
            yield from OrderedSubsets(vals=levels, lists=True)

        def arg_ddof(v_self: pd.DataFrame, v_axis: int):
            if v_axis == 0:
                yield from Select(range(0, v_self.shape[0]))
            else:
                yield from Select(range(0, v_self.shape[1]))

        _self = Ext(DType(pd.DataFrame))
        _axis = Choice(Default(0), 1)
        _numeric_only = Choice(Default(None), True, False)
        _level = Chain(Default(None), arg_level(_self, _axis))
        _skipna = Choice(Default(None), True, False)
        _ddof = Chain(Default(1), Ext(DType(int)), arg_ddof(_self, _axis))

    @inherit('df.kurt')
    @signature('DataFrame.skew(self, axis=None, skipna=None, level=None, numeric_only=None)')
    @target(pd.DataFrame.skew)
    def skew():
        pass

    @inherit('df.prod')
    @signature('DataFrame.sum(self, axis=None, skipna=None, level=None, numeric_only=None, min_count=0)')
    @target(pd.DataFrame.sum)
    def sum():
        pass

    @inherit('df.sem')
    @signature('DataFrame.std(self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None)')
    @target(pd.DataFrame.std)
    def std():
        pass

    @inherit('df.sem')
    @signature('DataFrame.var(self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None)')
    @target(pd.DataFrame.var)
    def var():
        pass

        # ------------------------------------------------------------------- #
        #  Reindexing/Selection/Label Manipulations
        # ------------------------------------------------------------------- #

    @signature('DataFrame.add_prefix(self, prefix)')
    @target(pd.DataFrame.add_prefix)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def add_prefix():
        _self = Ext(DType(pd.DataFrame))
        _prefix = Ext(DType(str))

    @signature('DataFrame.add_suffix(self, suffix)')
    @target(pd.DataFrame.add_suffix)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def add_suffix():
        _self = Ext(DType(pd.DataFrame))
        _suffix = Ext(DType(str))

    @signature(
        "DataFrame.align(self, other, join='outer', axis=None, level=None, copy=True, fill_value=None, method=None, "
        "limit=None, fill_axis=0, broadcast_axis=None)")
    @target(pd.DataFrame.align)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(tuple)])
    def align():

        def arg_level(v_self: pd.DataFrame, v_axis: int):
            index = (v_self.index if (v_axis == 0) else v_self.columns)
            levels = [(index.names[i] or i) for i in range(index.nlevels)]
            yield from OrderedSubsets(vals=levels, lists=True)

        _self = Ext(DType(pd.DataFrame))
        _other = Ext(DType([pd.DataFrame, pd.Series]))
        _join = Choice(Default('outer'), 'inner', 'left', 'right')
        _axis = Choice(Default(None), 0, 1)
        _broadcast_axis = Choice(Default(None), 0, 1)
        _level = Chain(Default(None), arg_level(_self, _axis))

    @signature(
        "DataFrame.drop(self, labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')")
    @target(pd.DataFrame.drop)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def drop():

        def arg_level(v_self: pd.DataFrame, v_axis: int):
            index = (v_self.index if (v_axis == 0) else v_self.columns)
            if index.nlevels == 1:
                yield Default(None)
            else:
                yield from Select(list(index.names[i] or i for i in range(0, index.nlevels)))

        def arg_labels(v_self: pd.DataFrame, v_axis: int, v_level):
            index = (v_self.index if (v_axis == 0) else v_self.columns)
            src = (set(index.get_level_values(v_level)) if (v_level is not None) else set(index))
            yield from Subsets(vals=src, lengths=range(1, len(src)), lists=True)

        _self = Ext(DType(pd.DataFrame))
        _axis = Choice(Default(0), 1)
        _errors = Choice(Default('raise'), 'ignore')
        _level = arg_level(_self, _axis)
        _labels = arg_labels(_self, _axis, _level)

    @signature("DataFrame.drop_duplicates(self, subset=None, keep='first', inplace=False)")
    @target(pd.DataFrame.drop_duplicates)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def drop_duplicates():
        _self = Ext(DType(pd.DataFrame))
        _subset = Subsets(_self.columns, lists=True)
        _keep = Choice(Default('first'), 'last', False)

    @signature("DataFrame.duplicated(self, subset=None, keep='first')")
    @target(pd.DataFrame.duplicated)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.Series)])
    def duplicated():
        _self = Ext(DType(pd.DataFrame))
        _subset = Subsets(_self.columns, lists=True)
        _keep = Choice(Default('first'), 'last', False)

    @signature('DataFrame.equals(self, other)')
    @target(pd.DataFrame.equals)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(bool)])
    def equals():
        _self = Ext(DType(pd.DataFrame))
        _other = Ext(DType(pd.DataFrame))

    @signature('DataFrame.filter(self, items=None, like=None, regex=None, axis=None)')
    @target(pd.DataFrame.filter)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def filter():
        def arg_axis(v_items):
            if v_items is not None:
                yield Inactive(None)
                return

            yield from Choice(0, 1)

        def arg_like(v_items):
            if v_items is not None:
                yield Inactive(None)
                return

            yield from Chain(Inactive(None), Ext(DType(str)))

        def arg_regex(v_items, v_like):
            if (v_items is not None) or (v_like is not None):
                yield Inactive(None)
                return

            yield from Ext(DType(str))

        _self = Ext(DType(pd.DataFrame))
        _items = Chain(Inactive(None), Subsets(vals=_self.columns, lists=True))
        _axis = arg_axis(_items)
        _like = arg_like(_items)
        _regex = arg_regex(_items, _like)

    @signature('DataFrame.first(self, offset)')
    @target(pd.DataFrame.first)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def first():
        _self = Ext(DType(pd.DataFrame))
        _offset = Ext(DType([str, pd.DateOffset, dateutil.relativedelta.relativedelta]))

    @signature('DataFrame.idxmax(self, axis=0, skipna=True)')
    @target(pd.DataFrame.idxmax)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.Series)])
    def idxmax():
        _self = Ext(DType(pd.DataFrame))
        _axis = Choice(Default(0), 1)
        _skipna = Choice(Default(True), False)

    @inherit('df.idxmax')
    @signature('DataFrame.idxmin(self, axis=0, skipna=True)')
    @target(pd.DataFrame.idxmin)
    def idxmin():
        pass

    @inherit('df.first')
    @signature('DataFrame.last(self, offset)')
    @target(pd.DataFrame.last)
    def last():
        pass

    @signature(
        'DataFrame.reindex(self, labels=None, index=None, columns=None, axis=None, method=None, copy=True, level=None, '
        'fill_value=nan, limit=None, tolerance=None)')
    @target(pd.DataFrame.reindex)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def reindex():
        def arg_level(v_self: pd.DataFrame, v_axis: int):
            if v_axis is None:
                yield Default(None)
                return

            index = (v_self.index if (v_axis == 0) else v_self.columns)
            if index.nlevels == 1:
                yield Default(None)
            else:
                yield from Select(list(index.names[i] or i for i in range(0, index.nlevels)))

        def arg_index(v_axis):
            if v_axis is not None:
                yield Inactive(None)
                return

            if _spec.depth == _spec.max_depth:
                yield list(_spec.output.index)

        def arg_columns(v_axis):
            if v_axis is not None:
                yield Inactive(None)
                return

            if _spec.depth == _spec.max_depth:
                yield list(_spec.output.columns)

        def arg_labels(v_axis, v_level):
            if v_axis is None:
                yield Default(None)
                return

            if _spec.depth == _spec.max_depth:
                output: pd.DataFrame = _spec.output
                v_axis = v_axis
                v_level = v_level
                index = (output.index if (v_axis == 0) else output.columns)
                if v_level is not None:
                    try:
                        yield list(collections.OrderedDict.fromkeys(index.get_level_values(v_level)).keys())
                    except:
                        pass
                else:
                    yield list(index)

        _fill_value = Chain(Default(np.NaN), Ext(FType(np.isscalar)))
        _limit = Chain(Default(None), Ext(DType(int)))
        _self = Ext(DType(pd.DataFrame))
        _axis = Chain(Inactive(None), Choice(0, 1))
        _level = arg_level(_self, _axis)
        _labels = arg_labels(_axis, _level)
        _index = arg_index(_axis)
        _columns = arg_columns(_axis)
        _method = Choice(Default(None), 'bfill', 'pad', 'nearest')

    @signature('DataFrame.reindex_like(self, other, method=None, copy=True, limit=None, tolerance=None)')
    @target(pd.DataFrame.reindex_like)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def reindex_like():
        _self = Ext(DType(pd.DataFrame))
        _limit = Chain(Default(None))
        _other = Ext(DType(pd.DataFrame))
        _method = Choice(Default(None), 'bfill', 'pad', 'nearest')

    @signature(
        'DataFrame.rename(self, mapper=None, index=None, columns=None, axis=None, copy=True, inplace=False, level=None)')
    @target(pd.DataFrame.rename)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def rename():
        def arg_mapper(v_axis):
            if v_axis is None:
                yield Inactive(None)
                return
            yield from Ext(DType([dict, Callable]))

        def arg_level(v_self: pd.DataFrame, v_axis: int):
            if v_axis is None:
                yield Inactive(None)
                return

            index = (v_self.index if (v_axis == 0) else v_self.columns)
            if index.nlevels == 1:
                yield Default(None)
            else:
                yield from Select(list(index.names[i] or i for i in range(0, index.nlevels)))

        def arg_index(v_axis):
            if v_axis is not None:
                yield Inactive(None)
                return
            yield from Ext(DType([dict, Callable]))

        def arg_columns(v_axis):
            if v_axis is not None:
                yield Inactive(None)
                return
            yield from Ext(DType([dict, Callable]))

        _self = Ext(DType(pd.DataFrame))
        _axis = Chain(Inactive(None), Choice(0, 1))
        _mapper = arg_mapper(_axis)
        _level = arg_level(_self, _axis)
        _index = arg_index(_axis)
        _columns = arg_columns(_axis)

    @signature("DataFrame.reset_index(self, level=None, drop=False, inplace=False, col_level=0, col_fill='')")
    @target(pd.DataFrame.reset_index)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def reset_index():
        def arg_level(v_self):
            index = v_self.index
            if index.nlevels > 1:
                levels = [(index.names[i] or i) for i in range(index.nlevels)]
                yield from Subsets(vals=levels, lengths=range(1, index.nlevels), lists=True)

        def arg_col_level(v_self: pd.DataFrame):
            colindex = v_self.columns
            yield from Select(list(colindex.names[i] or i for i in range(1, colindex.nlevels)))

        _col_fill = Chain(Default(None), Ext(DType(str)))
        _self = Ext(DType(pd.DataFrame))
        _drop = Choice(Default(True), False)
        _level = Chain(Default(None), arg_level(_self))
        _col_level = Chain(Default(0), arg_col_level(_self))

    @signature('DataFrame.set_index(self, keys, drop=True, append=False, inplace=False, verify_integrity=False)')
    @target(pd.DataFrame.set_index)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def set_index():
        def arg_keys(v_self: pd.DataFrame):
            colnames = v_self.columns
            if _spec.depth == _spec.max_depth:
                output = _spec.output
                out_index = output.index.names
                vals = (set(colnames) & set(out_index))
                yield [i for i in colnames if (i in vals)]
                return

            yield from OrderedSubsets(vals=colnames, lengths=range(1, len(colnames)), lists=True)

        _self = Ext(DType(pd.DataFrame))
        _drop = Choice(Default(True), False)
        _append = Choice(Default(False), True)
        _keys = arg_keys(_self)

    @signature('DataFrame.take(self, indices, axis=0, convert=None, is_copy=True)')
    @target(pd.DataFrame.take)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def take():
        _self = Ext(DType(pd.DataFrame))
        _indices = Ext(DType(Sequence))
        _axis = Choice(Default(0), 1)

        # ------------------------------------------------------------------- #
        #  Missing Data Handling
        # ------------------------------------------------------------------- #

    @signature("DataFrame.dropna(self, axis=0, how='any', thresh=None, subset=None, inplace=False)")
    @target(pd.DataFrame.dropna)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def dropna():
        def arg_subset(v_self: pd.DataFrame, v_axis):
            index = (v_self.columns if (v_axis == 0) else v_self.index)
            yield from Subsets(vals=index, lengths=range(1, len(index)), lists=True)

        _thresh = Chain(Default(None), Ext(DType(int)))
        _self = Ext(DType(pd.DataFrame))
        _axis = Choice(Default(0), 1)
        _how = Choice(Default('any'), 'all')
        _subset = Chain(Default(None), arg_subset(_self, _axis))

    @signature('DataFrame.fillna(self, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None)')
    @target(pd.DataFrame.fillna)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def fillna():
        def arg_value():
            if _spec.depth == _spec.max_depth:
                output: pd.DataFrame = _spec.output
                all_values = set([i for col in output for i in output[col] if (not pd.isnull(i))])
                yield from map(lambda x: AnnotatedVal(x, cost=2),
                               Select(all_values))

        def arg_limit(v_self: pd.DataFrame):
            yield from map(lambda x: AnnotatedVal(x, cost=5),
                           Select(range(1, (max(v_self.shape) + 1))))

        _self = Ext(DType(pd.DataFrame))
        _axis = Choice(Default(None), 0, 1)
        _method = Choice(Default(None), 'backfill', 'bfill', 'pad', 'ffill')
        _limit = Chain(Default(None), Ext(DType(int)), arg_limit(_self))
        _value = Chain(Default(None), Ext(FType(np.isscalar)), Ext(DType([dict, pd.Series, pd.DataFrame])), arg_value())

        # ------------------------------------------------------------------- #
        #  Reshaping, Sorting, Transposing
        # ------------------------------------------------------------------- #

    @signature("DataFrame.pivot_table(self, values=None, index=None, columns=None, aggfunc='mean', fill_value=None, "
               "margins=False, dropna=True, margins_name='All')")
    @target(pd.DataFrame.pivot_table)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def pivot_table():

        def arg_columns(v_self: pd.DataFrame):
            columns = v_self.columns
            if _spec.depth == _spec.max_depth:
                output: pd.DataFrame = _spec.output
                columns = (set(columns) & (set(output.index.names) | set(output.columns.names)))
                lengths = None
                yield from OrderedSubsets(vals=columns, lengths=lengths, lists=True)

            else:
                yield from Chain(arg_columns_1(v_self),
                                 arg_columns_2(v_self))

        def arg_columns_1(v_self: pd.DataFrame):
            columns = v_self.columns
            numeric_cols = set(v_self.select_dtypes(include=np.number).columns)
            columns = (set(columns) - numeric_cols)
            if len(numeric_cols) == 0:
                return

            for perm in OrderedSubsets(vals=columns, lists=True):
                for perm_num in OrderedSubsets(vals=numeric_cols, lengths=range(0, len(numeric_cols)),
                                               lists=True):
                    yield (perm + perm_num)

        def arg_columns_2(v_self: pd.DataFrame):
            columns = v_self.columns
            numeric_cols = set(v_self.select_dtypes(include=np.number).columns)
            columns = (set(columns) - numeric_cols)
            if len(numeric_cols) == 0:
                return

            for perm_num in OrderedSubsets(vals=numeric_cols, lengths=range(1, len(numeric_cols)),
                                           lists=True):
                yield perm_num

        def arg_index(v_self: pd.DataFrame, v_columns):
            columns = (set(v_self.columns) - set(v_columns))
            if _spec.depth == _spec.max_depth:
                output: pd.DataFrame = _spec.output
                columns = (set(columns) & (set(output.index.names) | set(output.columns.names)))
                lengths = None
                if v_columns:
                    yield Default([])
                for perm in OrderedSubsets(vals=columns, lengths=lengths, lists=True):
                    yield perm
            else:
                if v_columns:
                    yield from Chain(Default([]), arg_index_1(v_self, v_columns),
                                     arg_index_2(v_self, v_columns))
                else:
                    yield from Chain(arg_index_1(v_self, v_columns),
                                     arg_index_2(v_self, v_columns))

        def arg_index_1(v_self: pd.DataFrame, v_columns):
            columns = (set(v_self.columns) - set(v_columns))
            numeric_cols = (set(v_self.select_dtypes(include=np.number).columns) - set(v_columns))
            if len(numeric_cols) == 0:
                return
            columns = (columns - numeric_cols)
            for perm in OrderedSubsets(vals=columns, lists=True):
                for perm_num in OrderedSubsets(vals=numeric_cols, lengths=range(0, len(numeric_cols)),
                                               lists=True):
                    yield (perm + perm_num)

        def arg_index_2(v_self: pd.DataFrame, v_columns):
            columns = (set(v_self.columns) - set(v_columns))
            numeric_cols = (set(v_self.select_dtypes(include=np.number).columns) - set(v_columns))
            if len(numeric_cols) == 0:
                return
            columns = (columns - numeric_cols)
            for perm_num in OrderedSubsets(vals=numeric_cols, lengths=range(1, len(numeric_cols)),
                                           lists=True):
                yield perm_num

        def arg_values(v_self: pd.DataFrame, v_columns, v_index, v_aggfunc):
            try:
                _ = v_aggfunc(pd.Series(['a', 'b']))
                use_all = True
            except:
                use_all = False
            if not use_all:
                columns = ((set(v_self.select_dtypes(include=np.number).columns) - set(v_columns)) - (
                    set(v_index) if ((v_index is not None) and (v_index != [])) else set()))
            else:
                columns = ((set(v_self.columns) - set(v_columns)) - (
                    set(v_index) if ((v_index is not None) and (v_index != [])) else set()))

            col_domain = [col for col in columns if not isinstance(col, (list, tuple))]
            yield from Chain(Select(col_domain),
                             OrderedSubsets(vals=columns, lists=True))

        def arg_margins(v_self: pd.DataFrame):
            yield Default(False)
            if (v_self.index.nlevels == 1) and (v_self.columns.nlevels == 1):
                yield True

        _self = Ext(DType(pd.DataFrame))
        _columns = Chain(Default([]), arg_columns(_self))
        _index = arg_index(_self, _columns)
        _aggfunc = Chain(Default('mean'), Ext(DType([Callable, list, tuple, str])),
                         Choice(np.sum, np.max, np.min, np.median))

        _values = arg_values(_self, _columns, _index, _aggfunc)
        _fill_value = Chain(Default(None), Ext(FType(np.isscalar)))
        _margins = arg_margins(_self)
        _dropna = Choice(Default(True), False)
        _margins_name = Chain(Default('All'), Ext(DType(str)))

    @signature('DataFrame.pivot(self, index=None, columns=None, values=None)')
    @target(pd.DataFrame.pivot)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def pivot():
        def arg_columns(v_self: pd.DataFrame):
            if v_self.columns.nlevels > 1:
                return

            yield from Select(v_self.columns)

        def arg_index(v_self: pd.DataFrame, v_columns: str):
            candidates = set(v_self.columns) - {v_columns}

            def dup_filter(cand):
                try:
                    return not any(v_self[[cand, v_columns]].duplicated())
                except:
                    return True

            candidates = list(filter(dup_filter, candidates))
            candidates.append(Default(None))

            yield from Select(candidates)

        def arg_values(v_self: pd.DataFrame, v_index: str):
            if v_self.index.nlevels > 1 and v_index is None:
                yield Default(None)
                return

            candidates = set(v_self.columns) | {Default(None)}
            yield from Select(candidates)

        _self = Ext(DType(pd.DataFrame))
        _columns = arg_columns(_self)
        _index = arg_index(_self, _columns)
        _values = arg_values(_self, _index)

    @signature('DataFrame.reorder_levels(self, order, axis=0)')
    @target(pd.DataFrame.reorder_levels)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def reorder_levels():

        def arg_order(v_self: pd.DataFrame, v_axis):
            index = (v_self.index if (v_axis == 0) else v_self.columns)
            if index.nlevels > 1:
                levels = [(index.names[i] or i) for i in range(index.nlevels)]
                yield from OrderedSubsets(vals=levels, lists=True)

        _self = Ext(DType(pd.DataFrame))
        _axis = Choice(Default(0), 1)
        _order = arg_order(_self, _axis)

    @signature(
        "DataFrame.sort_values(self, by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')")
    @target(pd.DataFrame.sort_values)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def sort_values():
        def arg_by(v_self: pd.DataFrame, v_axis: int):
            if v_axis == 0:
                src = v_self.columns
                duplicated = [col for col in src if any(v_self.duplicated(col))]
                duplicated += [i for i in v_self.index.names if (i is not None)]
                nondups = (set(src) - set(duplicated))
            else:
                dft = v_self.T
                src = v_self.index
                duplicated = [idx for idx in src if any(dft.duplicated(idx))]
                nondups = (set(src) - set(duplicated))

            yield from Chain(arg_by1(duplicated, nondups),
                             arg_by2(duplicated, nondups))

        def arg_by1(duplicated, nondups):
            for i in Select(nondups):
                yield [i]

        def arg_by2(duplicated, nondups):
            for perm in OrderedSubsets(vals=duplicated, lists=True):
                yield from Chain(perm,
                                 arg_by3(perm, nondups))

        def arg_by3(perm, nondups):
            for i in Select(nondups):
                yield perm + [i]

        def arg_ascending(v_by):
            if _spec.depth == _spec.max_depth:
                length_by = len(v_by)
                if length_by > 1:
                    for i in range((1 << length_by)):
                        yield list(map((lambda x: (x == '1')), (('{:0' + str(length_by)) + 'b}').format(i)))

        _self = Ext(DType(pd.DataFrame))
        _axis = Choice(Default(0), 1)
        _na_position = Choice(Default('last'), 'first')
        _by = arg_by(_self, _axis)
        _ascending = Chain(Choice(Default(True), False), arg_ascending(_by))

    @signature('DataFrame.stack(self, level=-1, dropna=True)')
    @target(pd.DataFrame.stack)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame), DType(pd.Series)])
    def stack():
        def arg_level(v_self: pd.DataFrame):
            columns = v_self.columns
            if columns.nlevels > 1:
                levels = [(columns.names[i] or i) for i in range(columns.nlevels)]
                yield from OrderedSubsets(vals=levels, lists=True)

        _self = Ext(DType(pd.DataFrame))
        _dropna = Choice(Default(True), False)
        _level = Chain(Default((- 1)), arg_level(_self))

    @signature('DataFrame.unstack(self, level=-1, fill_value=None)')
    @target(pd.DataFrame.unstack)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame), DType(pd.Series)])
    def unstack():
        def arg_level(v_self):
            index = v_self.index
            if index.nlevels > 1:
                levels = [(index.names[i] or i) for i in range(index.nlevels)]
                yield from OrderedSubsets(vals=levels, lists=True)

        def arg_fill_value():
            if _spec.depth == _spec.max_depth:
                output: Union[(pd.DataFrame, pd.Series)] = _spec.output
                if isinstance(output, pd.Series):
                    all_values = set([i for i in output if (not pd.isnull(i))])
                else:
                    all_values = set([i for col in output for i in output[col] if (not pd.isnull(i))])

                yield from map(lambda x: AnnotatedVal(x, cost=2),
                               Select(all_values))

        _self = Ext(DType(pd.DataFrame))
        _level = Chain(Default((- 1)), arg_level(_self))
        _fill_value = Chain(Default(None), Ext(FType(np.isscalar)), arg_fill_value())

    @signature("DataFrame.melt(self, id_vars=None, value_vars=None, var_name=None, value_name='value', col_level=None)")
    @target(pd.DataFrame.melt)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def melt():
        def arg_value_vars(v_self, v_id_vars):
            columns = set(v_self.columns)
            v_id_vars = (v_id_vars or [])
            yield from OrderedSubsets(vals=list((columns - set(v_id_vars))), lists=True)

        def arg_col_level(v_self):
            columns = v_self.columns
            if columns.nlevels > 1:
                yield from Select(range(0, columns.nlevels))

        _var_name = Chain(Default(None), Ext(DType(str)))
        _value_name = Chain(Default('value'), Ext(DType(str)))
        _self = Ext(DType(pd.DataFrame))
        _id_vars = Chain(Default(None), OrderedSubsets(_self.columns, lists=True))
        _value_vars = Chain(Default(None), arg_value_vars(_self, _id_vars))
        _col_level = Chain(Default(None), arg_col_level(_self))

        # ------------------------------------------------------------------- #
        #  Combining/Joining/Merging
        # ------------------------------------------------------------------- #

    @signature("DataFrame.merge(self, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, "
               "right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)")
    @target(pd.DataFrame.merge)
    @inp_types([DType(pd.DataFrame)])
    @out_types([DType(pd.DataFrame)])
    def merge():
        def arg_on(v_self: pd.DataFrame, v_right: pd.DataFrame):
            columns = (set(v_self.columns) & set(v_right.columns))
            lengths = None
            if (_spec.depth == _spec.max_depth) and isinstance(_spec.output, pd.DataFrame):
                columns = (columns & set(_spec.output.columns))
                lengths = range(len(columns), 0, (- 1))

            yield from Subsets(vals=columns, lengths=lengths, lists=True)

        def arg_left_index(v_on):
            if v_on is not None:
                yield Inactive(False)
            else:
                yield from Choice(Default(False), True)

        def arg_right_index(v_on):
            if v_on is not None:
                yield Inactive(False)
            else:
                yield from Choice(Default(False), True)

        def arg_left_on(v_self: pd.DataFrame, v_right, v_on, v_left_index, v_right_index):
            if (v_on is not None) or v_left_index is True:
                yield Inactive(None)
                return

            columns = set(v_self.columns)
            lengths = None
            if (_spec.depth == _spec.max_depth) and isinstance(_spec.output, pd.DataFrame):
                columns = (columns & set(_spec.output.columns))
                lengths = range(len(columns), 0, (- 1))

            if v_right_index:
                v_right: pd.DataFrame = v_right
                nlevels = v_right.index.nlevels
                if (nlevels <= 0) or (nlevels > len(columns)):
                    return
                lengths = [nlevels]

            yield from Subsets(vals=columns, lengths=lengths, lists=True)

        def arg_right_on(v_self: pd.DataFrame, v_right: pd.DataFrame, v_on, v_left_on, v_left_index, v_right_index):
            if (v_on is not None) or v_right_index is True:
                yield Inactive(None)
                return

            columns = set(v_right.columns)
            lengths = None
            if (_spec.depth == _spec.max_depth) and isinstance(_spec.output, pd.DataFrame):
                columns = (columns & set(_spec.output.columns))
                lengths = range(len(columns), 0, (- 1))

            if v_left_on is not None:
                lengths = [len(v_left_on)]

            elif v_left_index is True:
                idxlen = len(v_self.index)
                if (idxlen <= 0) or (idxlen > len(columns)):
                    return

                lengths = [idxlen]

            yield from OrderedSubsets(vals=columns, lengths=lengths, lists=True)

        _self = Ext(DType(pd.DataFrame))
        _right = Ext(DType(pd.DataFrame))
        _how = Choice(Default('inner'), 'outer', 'left', 'right')
        _on = Chain(Inactive(None), arg_on(_self, _right))
        _left_index = arg_left_index(_on)
        _right_index = arg_right_index(_on)
        _left_on = arg_left_on(_self, _right, _on, _left_index, _right_index)
        _right_on = arg_right_on(_self, _right, _on, _left_on, _left_index, _right_index)
        _sort = Choice(Default(False), True)
        # mutex = MutEx(['on', Codep([MutEx(['left_on', 'left_index']), MutEx(['right_on', 'right_index'])])])


# noinspection PyMethodParameters
class dfgroupby:

    @signature('DataFrameGroupBy.count(self)')
    @target(pd_dfgroupby.count)
    @inp_types([DType(pd_dfgroupby)])
    @out_types([DType(pd.DataFrame)])
    def count():
        _self = Ext(DType(pd_dfgroupby))

    @inherit('dfgroupby.count')
    @signature('DataFrameGroupBy.first(self)')
    @target(pd_dfgroupby.first)
    def first():
        pass

    @inherit('dfgroupby.count')
    @signature('DataFrameGroupBy.last(self)')
    @target(pd_dfgroupby.last)
    def last():
        pass

    @inherit('dfgroupby.count')
    @signature('DataFrameGroupBy.max(self)')
    @target(pd_dfgroupby.max)
    def max():
        pass

    @inherit('dfgroupby.count')
    @signature('DataFrameGroupBy.mean(self)')
    @target(pd_dfgroupby.mean)
    def mean():
        pass

    @inherit('dfgroupby.count')
    @signature('DataFrameGroupBy.median(self)')
    @target(pd_dfgroupby.median)
    def median():
        pass

    @inherit('dfgroupby.count')
    @signature('DataFrameGroupBy.min(self)')
    @target(pd_dfgroupby.min)
    def min():
        pass

    @inherit('dfgroupby.count')
    @signature('DataFrameGroupBy.idxmin(self)')
    @target(pd_dfgroupby.idxmin)
    def idxmin():
        pass

    @inherit('dfgroupby.count')
    @signature('DataFrameGroupBy.idxmax(self)')
    @target(pd_dfgroupby.idxmax)
    def idxmax():
        pass

    @inherit('dfgroupby.count')
    @signature('DataFrameGroupBy.prod(self)')
    @target(pd_dfgroupby.prod)
    def prod():
        pass

    @inherit('dfgroupby.count')
    @signature('DataFrameGroupBy.size(self)')
    @target(pd_dfgroupby.size)
    def size():
        pass

    @inherit('dfgroupby.count')
    @signature('DataFrameGroupBy.sum(self)')
    @target(pd_dfgroupby.sum)
    def sum():
        pass

    @signature('DataFrameGroupBy.transform(self, func)')
    @target(pd_dfgroupby.transform)
    @inp_types([DType(pd_dfgroupby)])
    @out_types([DType(pd.DataFrame), DType(pd.Series)])
    def transform():
        _self = Ext(DType(pd_dfgroupby))
        _func = Ext(DType([Callable, str, dict, list, tuple]))


# noinspection PyMethodParameters
class pandas:

    @signature("pd.concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False, keys=None, levels=None, "
               "names=None, verify_integrity=False, sort=None, copy=True) ")
    @target(pd.pandas.concat)
    def concat():

        def arg_objs():
            all_objs = []
            for (_, val, val_repr, _, _) in _Ext(_spec, DType([pd.Series, pd.DataFrame])):
                all_objs.append((val, val_repr))
            for num_concatted in range(2, (len(all_objs) + 1)):
                for permutation in itertools.permutations(all_objs, num_concatted):
                    (value_list, repr_list) = [list(x) for x in zip(*permutation)]
                    repr_string = repr_list.__str__()
                    yield AnnotatedVal(value_list, repr=repr_string)

        def arg_keys(v_objs):
            const_vals = _spec.spec.constants
            num_concatted = len(v_objs)
            for constants_permutation in itertools.permutations(const_vals, num_concatted):
                yield list(constants_permutation)
            output = _spec.spec.output
            if isinstance(output, pd.DataFrame):
                for col_level in range(output.columns.nlevels):
                    for i in range(0, col_level):
                        level_values = output.columns.get_level_values(i)
                        unique_level_values = set(level_values)
                        for columns_permutation in itertools.permutations(unique_level_values, num_concatted):
                            yield list(columns_permutation)
            if isinstance(output, pd.Series) or isinstance(output, pd.DataFrame):
                for index_level in range(output.index.nlevels):
                    for i in range(0, index_level):
                        level_values = output.index.get_level_values(i)
                        unique_level_values = set(level_values)
                        for indices_permutation in itertools.product(unique_level_values, repeat=num_concatted):
                            yield list(indices_permutation)

        _objs = arg_objs()
        _axis = Choice(Default(0), 1)
        _join = Choice(Default('outer'), 'inner')
        _join_axes = Default(None)
        _ignore_index = Choice(Default(False), True)
        _keys = Chain(Default(None), arg_keys(_objs))
        _levels = Default(None)
        _names = Default(None)
        _verify_integrity = Choice(Default(False))
        _sort = Default(None)
        _copy = Choice(Default(True))
        mutex = MutEx(['join', 'join_axes'])
