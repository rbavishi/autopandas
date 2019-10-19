import collections
import random
from typing import Sequence, Callable, List, Union

import numpy as np
import pandas as pd
import itertools
import dateutil
from numpy import nan
import re
from autopandas_v2.generators.specs import df as s_df, dfgroupby as s_dfgroupby
from autopandas_v2.generators.ml.traindata.randutils import RandDf, StrColGen, IntColGen, FloatColGen, BoolColGen, \
    RandStr, NaturalRandDf, bool_bags, ints_bags, floats_bags, string_bags, moar_nans_floats_bag

from autopandas_v2.utils.datastructures import oset
from autopandas_v2.generators.compilation.stubs import extend, _spec, signature, target, inherit
from autopandas_v2.utils.types import DType, FType, Lambda, is_float, is_int

from autopandas_v2.generators.ml.traindata.dsl.ops import RExt
from autopandas_v2.generators.dsl.ops import Select, Ext, Choice, Chain, Subsets, OrderedSubsets, Product
from autopandas_v2.generators.dsl.values import Default, AnnotatedVal, Inactive, RandomColumn

pd_dfgroupby = pd.core.groupby.DataFrameGroupBy


def defaultRandDf(min_width=1, min_height=1, max_width=7, max_height=7, **kwargs):
    # return iter(RandDf(min_width, min_height, max_width, max_height, **kwargs))
    return iter(NaturalRandDf(min_width, min_height, max_width, max_height, **kwargs))


def coin_flip():
    return random.randint(0, 1)


# noinspection PyMethodParameters
class df:

    @extend(s_df.index)
    def index():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())

    @extend(s_df.columns)
    def columns():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())

    @extend(s_df.dtypes)
    def dtypes():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())

    @extend(s_df.ftypes)
    def ftypes():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())

    @extend(s_df.values)
    def values():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())

    @extend(s_df.axes)
    def axes():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())

    @extend(s_df.ndim)
    def ndim():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())

    @extend(s_df.size)
    def size():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())

    @extend(s_df.shape)
    def shape():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())

    @extend(s_df.T)
    def T():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())

    @extend(s_df.as_matrix)
    def as_matrix():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())

    @extend(s_df.get_dtype_counts)
    def get_dtype_counts():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())

    @extend(s_df.get_ftype_counts)
    def get_ftype_counts():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())

    @extend(s_df.select_dtypes)
    def select_dtypes():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())

    @extend(s_df.astype)
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

        def arg_dtype(v_self: pd.DataFrame):
            pool = ['int32', 'uint32', 'float64', 'float32', 'int64', 'uint64']
            mapping = {pool[i]: (([None] + pool[:i]) + pool[(i + 1):]) for i in range(len(pool))}
            mapping['object'] = [None]
            res = {

            }
            for col in v_self.columns:
                chosen = random.choice(mapping[str(v_self.dtypes[col])])
                if chosen is not None:
                    res[col] = chosen
            yield res

        _self = RExt(DType(pd.DataFrame), defaultRandDf())
        _dtype = Chain(RExt(DType(dict), arg_dtype(_self)), arg_astype_partial(_self))

    @extend(s_df.isna)
    def isna():
        # _self = RExt(DType(pd.DataFrame), defaultRandDf(
        #     column_gens=([StrColGen(), IntColGen()] + [FloatColGen(nan_prob=0.5) for i in range(3)])))
        _self = RExt(DType(pd.DataFrame), defaultRandDf(nan_prob=0.5))

    @extend(s_df.notna)
    def notna():
        _self = RExt(DType(pd.DataFrame), defaultRandDf(nan_prob=0.5))
        # _self = RExt(DType(pd.DataFrame), defaultRandDf(
        #     column_gens=([StrColGen(), IntColGen()] + [FloatColGen(nan_prob=0.5) for i in range(3)])))

    @extend(s_df.head)
    def head():
        def arg_head_partial(v_self: pd.DataFrame):
            if _spec.depth == _spec.max_depth:
                output: pd.DataFrame = _spec.output
                yield AnnotatedVal(output.shape[0], cost=0)

            yield from Select(list(range(1, v_self.shape[0] + 1)))

        def arg_n(v_self: pd.DataFrame):
            pool = list(set(([5] + list(range(1, len(v_self))))))
            yield random.choice(pool)

        _self = RExt(DType(pd.DataFrame), defaultRandDf())
        _n = Chain(Default(5), RExt(DType(int), arg_n(_self)), arg_head_partial(_self))

    @extend(s_df.at_getitem)
    def at_getitem():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())

    @extend(s_df.iat_getitem)
    def iat_getitem():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())

    @extend(s_df.loc_getitem)
    def loc_getitem():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())

    @extend(s_df.iloc_getitem)
    def iloc_getitem():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())

    @extend(s_df.lookup)
    def lookup():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())

    @inherit("df.head")
    @signature('DataFrame.tail(self, n=5)')
    @target(pd.DataFrame.tail)
    def tail():
        pass

    @extend(s_df.xs)
    def xs():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())

    @extend(s_df.isin)
    def isin():

        def arg_values(v_self: pd.DataFrame):
            vals = list(v_self.values.flatten())
            sample_size = random.randint(1, max((len(vals) - 1), 1))
            yield list(random.sample(vals, sample_size))

        _self = RExt(DType(pd.DataFrame), defaultRandDf())
        _values = RExt(DType(dict), arg_values(_self))

    @extend(s_df.where)
    def where():

        def arg_cond(v_self: pd.DataFrame):
            (nr, nc) = v_self.shape
            val: pd.DataFrame = next(defaultRandDf(num_rows=nr, num_columns=nc, value_bags=bool_bags))
            val.columns = v_self.columns
            val.index = v_self.index
            yield val

        def arg_other(v_self: pd.DataFrame):
            (nr, nc) = v_self.shape
            val: pd.DataFrame = next(defaultRandDf(num_rows=nr, num_columns=nc))
            val.columns = v_self.columns
            val.index = v_self.index
            yield val

        _self = RExt(DType(pd.DataFrame), defaultRandDf())
        _cond = RExt(DType([Sequence, pd.DataFrame, Callable]), arg_cond(_self))
        _other = RExt(DType([Sequence, pd.DataFrame, Callable]), arg_other(_self))

    @extend(s_df.mask)
    def mask():

        def arg_cond(v_self: pd.DataFrame):
            (nr, nc) = v_self.shape
            val: pd.DataFrame = next(defaultRandDf(num_rows=nr, num_columns=nc, value_bags=bool_bags))
            val.columns = v_self.columns
            val.index = v_self.index
            yield val

        def arg_other(v_self: pd.DataFrame):
            (nr, nc) = v_self.shape
            val: pd.DataFrame = next(defaultRandDf(num_rows=nr, num_columns=nc))
            val.columns = v_self.columns
            val.index = v_self.index
            yield val

        _self = RExt(DType(pd.DataFrame), defaultRandDf())
        _cond = RExt(DType([Sequence, pd.DataFrame, Callable]), arg_cond(_self))
        _other = RExt(DType([Sequence, pd.DataFrame, Callable]), arg_other(_self))

    @extend(s_df.query)
    def query():

        def arg_expr(v_self: pd.DataFrame):
            pool = []
            dtypes = v_self.dtypes
            for col in v_self:
                dtype = dtypes[col]
                vals = list(v_self[col])
                if ('int' in str(dtype)) or ('float' in str(dtype)):
                    pool.append('{} > {}'.format(col, random.choice(vals)))
                    pool.append('{} < {}'.format(col, random.choice(vals)))
                    pool.append('{} == {}'.format(col, random.choice(vals)))
                    pool.append('{} != {}'.format(col, random.choice(vals)))
                elif 'object' in str(dtype):
                    pool.append('{} == {}'.format(col, random.choice(vals)))
                    pool.append('{} != {}'.format(col, random.choice(vals)))
            sample_size = random.randint(1, min(5, len(pool)))
            sample = random.sample(pool, sample_size)
            expr = sample[0]
            for i in range(1, len(sample)):
                expr += ' {} '.format(random.choice(['and', 'or']))
                expr += sample[i]
            yield expr

        _self = RExt(DType(pd.DataFrame), defaultRandDf())
        _expr = RExt(DType(str), arg_expr(_self))

    @extend(s_df.__getitem__)
    def __getitem__():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())

    @extend(s_df.add)
    def add():

        def arg_other(v_self: pd.DataFrame):
            (nr, nc) = v_self.shape
            v_nc = random.choice([nc, nc - 1, nc + 1])
            val = next(defaultRandDf(num_rows=nr, num_columns=v_nc,
                                     column_levels=v_self.columns.nlevels, col_prefix='i1_',
                                     value_bags=[*ints_bags, *floats_bags]))
            val.index = v_self.index
            if (coin_flip() == 0) and (len(val.columns) == nc):
                val.columns = v_self.columns
            elif v_self.columns.nlevels == 1:
                val.columns = pd.Index(random.sample(set((list(v_self.columns) + list(val.columns))), len(val.columns)))
            else:
                val.columns = pd.MultiIndex.from_tuples(
                    random.sample(set((list(v_self.columns) + list(val.columns))), len(val.columns)))
            yield val

        def arg_fill_value():
            yield random.uniform((- 100), 100)

        _self = RExt(DType(pd.DataFrame), defaultRandDf(value_bags=[*ints_bags, *floats_bags]))
        _other = RExt(DType(pd.DataFrame), arg_other(_self))
        _fill_value = Chain(Default(None), RExt(DType(float), arg_fill_value()))

    @inherit('df.add')
    @signature("DataFrame.sub(self, other, axis='columns', level=None, fill_value=None)")
    @target(pd.DataFrame.sub)
    def sub():
        pass

    @inherit('df.add')
    @signature("DataFrame.mul(self, other, axis='columns', level=None, fill_value=None)")
    @target(pd.DataFrame.mul)
    def mul():
        pass

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

    @extend(s_df.lt)
    def lt():

        def arg_other(v_self: pd.DataFrame):
            (nr, nc) = v_self.shape
            val = next(defaultRandDf(num_rows=nr, column_levels=v_self.columns.nlevels, col_prefix='i1_',
                                     value_bags=[*ints_bags, *floats_bags]))
            val.index = v_self.index
            if (coin_flip() == 0) and (len(val.columns) == nc):
                val.columns = v_self.columns
            elif v_self.columns.nlevels == 1:
                val.columns = pd.Index(random.sample(set((list(v_self.columns) + list(val.columns))), len(val.columns)))
            else:
                val.columns = pd.MultiIndex.from_tuples(
                    random.sample(set((list(v_self.columns) + list(val.columns))), len(val.columns)))
            yield val

        _self = RExt(DType(pd.DataFrame), defaultRandDf(value_bags=[*ints_bags, *floats_bags]))
        _other = RExt(DType(pd.DataFrame), arg_other(_self))

    @inherit('df.lt')
    @signature("DataFrame.gt(self, other, axis='columns', level=None)")
    @target(pd.DataFrame.gt)
    def gt():
        pass

    @inherit('df.lt')
    @signature("DataFrame.le(self, other, axis='columns', level=None)")
    @target(pd.DataFrame.le)
    def le():
        pass

    @inherit('df.lt')
    @signature("DataFrame.ge(self, other, axis='columns', level=None)")
    @target(pd.DataFrame.ge)
    def ge():
        pass

    @inherit('df.lt')
    @signature("DataFrame.ne(self, other, axis='columns', level=None)")
    @target(pd.DataFrame.ne)
    def ne():

        def arg_other(v_self: pd.DataFrame):
            (nr, nc) = v_self.shape
            cond: pd.DataFrame = next(defaultRandDf(num_rows=nr, num_columns=nc, value_bags=bool_bags))
            cond.columns = v_self.columns
            cond.index = v_self.index
            val: pd.DataFrame = next(defaultRandDf(num_rows=nr, num_columns=nc))
            val.columns = v_self.columns
            val.index = v_self.index
            yield v_self.where(cond, val)

        _self = RExt(DType(pd.DataFrame), defaultRandDf())
        _other = RExt(DType(pd.DataFrame), arg_other(_self))

    @inherit('df.ne')
    @signature("DataFrame.eq(self, other, axis='columns', level=None)")
    @target(pd.DataFrame.eq)
    def eq():
        pass

    @extend(s_df.combine)
    def combine():

        def arg_func():
            pool = [Lambda('lambda s1, s2: s1.mask(s1 < s2, s2)'), Lambda('lambda s1, s2: s1.mask(s1 > s2, s2)')]
            yield random.choice(pool)

        def arg_other(v_self: pd.DataFrame):
            (nr, nc) = v_self.shape
            val: pd.DataFrame = next(
                defaultRandDf(num_rows=nr, num_columns=nc, value_bags=[*ints_bags, *floats_bags]))
            val.columns = v_self.columns
            val.index = v_self.index
            yield val

        _self = RExt(DType(pd.DataFrame), defaultRandDf(value_bags=[*ints_bags, *floats_bags]))
        _func = RExt(DType(Callable), arg_func())
        _other = RExt(DType(pd.DataFrame), arg_other(_self))

    @extend(s_df.combine_first)
    def combine_first():

        def arg_other(v_self: pd.DataFrame):
            (nr, nc) = v_self.shape
            if coin_flip() == 0:
                val = next(defaultRandDf(num_columns=nc, num_rows=nr, value_bags=(
                        [*string_bags, *ints_bags] + [*floats_bags, moar_nans_floats_bag])))
                val.columns = v_self.columns
                val.index = v_self.index
            else:
                val = next(defaultRandDf(index_levels=v_self.index.nlevels, column_levels=v_self.columns.nlevels,
                                         col_prefix='i1_', value_bags=(
                            [*string_bags, *ints_bags] + [*floats_bags, moar_nans_floats_bag])))
                if v_self.index.nlevels == 1:
                    val.index = pd.Index(random.sample(set((list(v_self.index) + list(val.index))), len(val.index)))
                else:
                    val.index = pd.MultiIndex.from_tuples(
                        random.sample(set((list(v_self.index) + list(val.index))), len(val.index)))
                if v_self.columns.nlevels == 1:
                    val.columns = pd.Index(
                        random.sample(set((list(v_self.columns) + list(val.columns))), len(val.columns)))
                else:
                    val.columns = pd.MultiIndex.from_tuples(
                        random.sample(set((list(v_self.columns) + list(val.columns))), len(val.columns)))
            yield val

        _self = RExt(DType(pd.DataFrame), defaultRandDf(value_bags=[*string_bags, *ints_bags] + [*floats_bags, moar_nans_floats_bag]))
        _other = RExt(DType(pd.DataFrame), arg_other(_self))

    @extend(s_df.apply)
    def apply():

        def arg_func(v_self: pd.DataFrame):
            numeric_cols = v_self.select_dtypes(include=np.number).columns
            if len(numeric_cols) == 0:
                return
            choice = random.choice(list(numeric_cols))
            yield Lambda('lambda x: x["{}"] > 1'.format(choice))
            yield Lambda('lambda x: x["{}"] + 1'.format(choice))

        _self = RExt(DType(pd.DataFrame), defaultRandDf())
        _func = RExt(DType(Callable), arg_func(_self))

    @extend(s_df.groupby)
    def groupby():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())

    @extend(s_df.abs)
    def abs():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())

    @extend(s_df.all)
    def all():
        _self = RExt(DType(pd.DataFrame), defaultRandDf(
            value_bags=[*ints_bags, *string_bags, *bool_bags, *bool_bags]))

    @inherit('df.all')
    @signature('DataFrame.any(self, axis=None, bool_only=None, skipna=None, level=None)')
    @target(pd.DataFrame.any)
    def any():
        pass

    @extend(s_df.clip)
    def clip():

        def arg_lower(v_self: pd.DataFrame):
            vals = list(filter((lambda x: (is_int(x) or is_float(x))), list(v_self.values.flatten())))
            if len(vals) == 0:
                return
            yield random.uniform(min(vals), max(vals))

        def arg_upper(v_self: pd.DataFrame, v_lower):
            vals = list(filter((lambda x: (is_int(x) or is_float(x))), list(v_self.values.flatten())))
            if len(vals) == 0:
                return
            if v_lower is None:
                v_lower = min(vals)
            yield random.uniform(v_lower, max(vals))

        _self = RExt(DType(pd.DataFrame), defaultRandDf(value_bags=[*ints_bags, *floats_bags]))
        _lower = Chain(Default(None), RExt(DType(float), arg_lower(_self)))
        _upper = Chain(Default(None), RExt(DType(float), arg_upper(_self, _lower)))

    @extend(s_df.clip_lower)
    def clip_lower():

        def arg_threshold(v_self: pd.DataFrame):
            vals = list(filter((lambda x: (not isinstance(x, str))), list(v_self.values.flatten())))
            yield random.uniform(min(vals), max(vals))

        _self = RExt(DType(pd.DataFrame), defaultRandDf(value_bags=[*ints_bags, *floats_bags]))
        _threshold = RExt(DType(float), arg_threshold(_self))

    @inherit('df.clip_lower')
    @signature('DataFrame.clip_upper(self, threshold, axis=None, inplace=False)')
    @target(pd.DataFrame.clip_upper)
    def clip_upper():
        pass

    @extend(s_df.corr)
    def corr():
        _self = RExt(DType(pd.DataFrame), defaultRandDf(value_bags=[*ints_bags, *floats_bags]))

    @extend(s_df.corrwith)
    def corrwith():

        def arg_other(v_self: pd.DataFrame):
            (nr, nc) = v_self.shape
            val = next(defaultRandDf(num_rows=nr, column_levels=v_self.columns.nlevels, col_prefix='i1_',
                                     value_bags=[*ints_bags, *floats_bags]))
            val.index = v_self.index
            if (coin_flip() == 0) and (len(val.columns) == nc):
                val.columns = v_self.columns
            elif v_self.columns.nlevels == 1:
                val.columns = pd.Index(random.sample(set((list(v_self.columns) + list(val.columns))), len(val.columns)))
            else:
                val.columns = pd.MultiIndex.from_tuples(
                    random.sample(set((list(v_self.columns) + list(val.columns))), len(val.columns)))
            yield val

        _self = RExt(DType(pd.DataFrame), defaultRandDf(value_bags=[*ints_bags, *floats_bags]))
        _other = RExt(DType(pd.DataFrame), arg_other(_self))

    @extend(s_df.count)
    def count():
        _self = RExt(DType(pd.DataFrame), defaultRandDf(nan_prob=0.5))

    @extend(s_df.cov)
    def cov():
        _self = RExt(DType(pd.DataFrame), defaultRandDf(nan_prob=0.1))

    @extend(s_df.cummax)
    def cummax():
        _self = RExt(DType(pd.DataFrame), defaultRandDf(nan_prob=0.1))

    @inherit('df.cummax')
    @signature('DataFrame.cummin(self, axis=None, skipna=True)')
    @target(pd.DataFrame.cummin)
    def cummin():
        pass

    @extend(s_df.cumprod)
    def cumprod():
        _self = RExt(DType(pd.DataFrame), defaultRandDf(nan_prob=0.1, value_bags=[*ints_bags, *floats_bags]))

    @inherit('df.cummax')
    @signature('DataFrame.cumsum(self, axis=None, skipna=True)')
    @target(pd.DataFrame.cumsum)
    def cumsum():
        pass

    @extend(s_df.diff)
    def diff():

        def arg_periods(v_self: pd.DataFrame):
            (nr, _) = v_self.shape
            yield random.choice(range((- (nr - 1)), nr))

        _self = RExt(DType(pd.DataFrame), defaultRandDf(nan_prob=0.1))
        _periods = Chain(Default(1), RExt(DType(int), arg_periods(_self)))

    @extend(s_df.kurt)
    def kurt():
        _self = RExt(DType(pd.DataFrame), defaultRandDf(nan_prob=0.1))

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

    @extend(s_df.mode)
    def mode():
        _self = RExt(DType(pd.DataFrame), defaultRandDf(nan_prob=0.1))

    @extend(s_df.pct_change)
    def pct_change():
        _self = RExt(DType(pd.DataFrame), defaultRandDf(nan_prob=0.1))

    @extend(s_df.prod)
    def prod():
        _self = RExt(DType(pd.DataFrame), defaultRandDf(nan_prob=0.1))

    @extend(s_df.quantile)
    def quantile():
        _self = RExt(DType(pd.DataFrame), defaultRandDf(nan_prob=0.1))

    @extend(s_df.rank)
    def rank():
        _self = RExt(DType(pd.DataFrame), defaultRandDf(nan_prob=0.1))

    @extend(s_df.round)
    def round():

        def arg_decimals():
            yield random.choice([1, 2, 3, 4, 5])

        _self = RExt(DType(pd.DataFrame), defaultRandDf(nan_prob=0.1))
        _decimals = Chain(Default(0), RExt(DType(int), arg_decimals()))

    @extend(s_df.sem)
    def sem():
        _self = RExt(DType(pd.DataFrame), defaultRandDf(nan_prob=0.1))

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

    @extend(s_df.add_prefix)
    def add_prefix():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())
        _prefix = RExt(DType(str), RandStr())

    @extend(s_df.add_suffix)
    def add_suffix():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())
        _suffix = RExt(DType(str), RandStr())

    @extend(s_df.align)
    def align():

        def arg_other(v_self: pd.DataFrame):
            (nr, nc) = v_self.shape
            val = next(defaultRandDf(num_rows=random.choice([max((nr - 1), 1), nr, (nr + 1)]),
                                     num_columns=random.choice([max((nc - 1), 1), nc, (nc + 1)]), col_prefix='i1_',
                                     index_levels=v_self.index.nlevels, column_levels=v_self.columns.nlevels,
                                     value_bags=[*ints_bags, *floats_bags]))
            if (coin_flip() == 0) and (len(val.index) == nr):
                val.index = v_self.index
            elif v_self.index.nlevels == 1:
                val.index = pd.Index(random.sample(set((list(v_self.index) + list(val.index))), len(val.index)))
            else:
                val.index = pd.MultiIndex.from_tuples(
                    random.sample(set((list(v_self.index) + list(val.index))), len(val.index)))
            if (coin_flip() == 0) and (len(val.columns) == nc):
                val.columns = v_self.columns
            elif v_self.columns.nlevels == 1:
                val.columns = pd.Index(random.sample(set((list(v_self.columns) + list(val.columns))), len(val.columns)))
            else:
                val.columns = pd.MultiIndex.from_tuples(
                    random.sample(set((list(v_self.columns) + list(val.columns))), len(val.columns)))
            yield val

        _self = RExt(DType(pd.DataFrame), defaultRandDf())
        _other = RExt(DType([pd.DataFrame, pd.Series]), arg_other(_self))

    @extend(s_df.drop)
    def drop():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())

    @extend(s_df.drop_duplicates)
    def drop_duplicates():
        _self = RExt(DType(pd.DataFrame), defaultRandDf(min_height=3))

    @extend(s_df.duplicated)
    def duplicated():
        _self = RExt(DType(pd.DataFrame), defaultRandDf(min_height=3))

    @extend(s_df.equals)
    def equals():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())
        _other = RExt(DType(pd.DataFrame), defaultRandDf(col_prefix=random.choice(['', 'i1_'])))

    @extend(s_df.filter)
    def filter():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())

    @extend(s_df.idxmax)
    def idxmax():
        _self = RExt(DType(pd.DataFrame), defaultRandDf(nan_prob=0.2))

    @inherit('df.idxmax')
    @signature('DataFrame.idxmin(self, axis=0, skipna=True)')
    @target(pd.DataFrame.idxmin)
    def idxmin():
        pass

    @extend(s_df.reindex)
    def reindex():

        def arg_labels(v_self: pd.DataFrame):
            (nr, nc) = v_self.shape
            if coin_flip() == 0:
                vals = list(v_self.index)
                new_vals = list(StrColGen(all_distinct=True).generate((nr // 2))[1].values())
                yield list(random.sample((vals + new_vals), nr))
            else:
                vals = list(v_self.columns)
                new_vals = list(StrColGen(all_distinct=True).generate((nc // 2))[1].values())
                yield list(random.sample((vals + new_vals), nc))

        def arg_fill_value():
            yield random.uniform((- 100), 100)

        _self = RExt(DType(pd.DataFrame), defaultRandDf())
        _labels = RExt(DType([list, dict]), arg_labels(_self))
        _fill_value = Chain(Default(np.NaN), RExt(DType(float), arg_fill_value()))

    @extend(s_df.reindex_like)
    def reindex_like():

        def arg_other(v_self: pd.DataFrame):
            val = next(defaultRandDf(index_levels=v_self.index.nlevels, col_prefix=random.choice(['', 'i1_']),
                                     value_bags=[*ints_bags, *floats_bags]))
            if v_self.index.nlevels == 1:
                val.index = pd.Index(random.sample(set((list(v_self.index) + list(val.index))), len(val.index)))
            else:
                val.index = pd.MultiIndex.from_tuples(
                    random.sample(set((list(v_self.index) + list(val.index))), len(val.index)))
            yield val

        _self = RExt(DType(pd.DataFrame), defaultRandDf())
        _other = RExt(DType(pd.DataFrame), arg_other(_self))

    @extend(s_df.reset_index)
    def reset_index():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())

    @extend(s_df.set_index)
    def set_index():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())

    @extend(s_df.take)
    def take():

        def arg_indices(v_self: pd.DataFrame):
            (nr, nc) = v_self.shape
            if coin_flip() == 0:
                val = random.sample(range(nr), random.choice(range(1, (nr + 1))))
                random.shuffle(val)
                yield val
            else:
                val = random.sample(range(nc), random.choice(range(1, (nc + 1))))
                random.shuffle(val)
                yield val

        _self = RExt(DType(pd.DataFrame), defaultRandDf())
        _indices = RExt(DType(Sequence), arg_indices(_self))

    @extend(s_df.dropna)
    def dropna():
        _self = RExt(DType(pd.DataFrame), defaultRandDf(nan_prob=0.5))

    @extend(s_df.fillna)
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

        def rarg_limit(v_self: pd.DataFrame):
            (nr, nc) = v_self.shape
            if coin_flip() == 0:
                yield random.choice(range(1, max(nr, 2)))
            else:
                yield random.choice(range(1, max(nc, 2)))

        def rarg_value():
            yield random.uniform((- 1000), 1000)

        _self = RExt(DType(pd.DataFrame), defaultRandDf(nan_prob=0.5))
        _limit = Chain(Default(None), RExt(DType(int), rarg_limit(_self)), arg_limit(_self))
        _value = Chain(Default(None), RExt(FType(np.isscalar), rarg_value()),
                       Ext(DType([dict, pd.Series, pd.DataFrame])), arg_value())

    @extend(s_df.pivot_table)
    def pivot_table():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())

    @extend(s_df.pivot)
    def pivot():
        _self = RExt(DType(pd.DataFrame), defaultRandDf(column_levels=1))

    @extend(s_df.reorder_levels)
    def reorder_levels():
        _self = RExt(DType(pd.DataFrame), defaultRandDf(multi_index_prob=0.6))

    @extend(s_df.sort_values)
    def sort_values():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())

    @extend(s_df.stack)
    def stack():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())

    @extend(s_df.unstack)
    def unstack():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())

    @extend(s_df.melt)
    def melt():
        _self = RExt(DType(pd.DataFrame), defaultRandDf())

    @extend(s_df.merge)
    def merge():

        def arg_right(v_self: pd.DataFrame):
            new_df: pd.DataFrame = next(defaultRandDf(col_prefix='i1_'))
            dg1 = collections.defaultdict(list)
            dg2 = collections.defaultdict(list)
            for (k, v) in dict(v_self.dtypes).items():
                dg1[v].append(k)
            for (k, v) in dict(new_df.dtypes).items():
                dg2[v].append(k)
            c = (set(dg1.keys()) & set(dg2.keys()))
            for dt in c:
                cols1 = list(dg1[dt])
                cols2 = list(dg2[dt])
                random.shuffle(cols1)
                random.shuffle(cols2)
                pairs = list(zip(cols1, cols2))
                for pair in pairs:
                    if coin_flip() == 0:
                        new_df[pair[1]] = random.sample((list(new_df[pair[1]]) + list(v_self[pair[0]])),
                                                        new_df.shape[0])
                        if (coin_flip() == 0) and (pair[0] not in new_df.columns):
                            new_df = new_df.rename({
                                pair[1]: pair[0],
                            }, axis=1)
            yield new_df

        _self = RExt(DType(pd.DataFrame), defaultRandDf())
        _right = RExt(DType(pd.DataFrame), arg_right(_self))


# noinspection PyMethodParameters
class dfgroupby:

    @extend(s_dfgroupby.count)
    def count():
        _self = RExt(DType(pd_dfgroupby))

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
