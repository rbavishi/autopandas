import logging
import unittest
from typing import Any, List

import pandas as pd
import numpy as np

from autopandas_v2.iospecs import IOSpec
from autopandas_v2.synthesis.search.engines.functions import BFSEngine, BaseEngine
from autopandas_v2.utils import logger


class TestGenerators(unittest.TestCase):
    def check(self, inputs: List[Any], output: Any, funcs: List[str], seqs: List[List[int]],
              constants: List[Any] = None):
        if constants is not None:
            inputs += constants
        iospec: IOSpec = IOSpec(inputs, output)
        iospec.funcs = funcs
        iospec.seqs = seqs

        engine: BFSEngine = BFSEngine(iospec)
        engine.max_depth = max(len(i) for i in seqs)
        engine.silent = True
        engine.stop_first_solution = True
        engine.use_spec_funcs = True
        engine.use_spec_seqs = True

        # try:
        #     engine.search()
        # except Exception as e:
        #     logging.exception(e)

        self.check_engine(engine)

    def check_engine(self, engine: BaseEngine):
        self.assertTrue(engine.search(), msg='Did not find a solution')

    def test_df_index(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].index
        funcs = ['df.index']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_index_2(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = [0, 1, 2, 3]
        funcs = ['df.index']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_columns(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].columns
        funcs = ['df.columns']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_columns_2(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = ['A', 'B', 'C']
        funcs = ['df.columns']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_dtypes(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].dtypes
        funcs = ['df.dtypes']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_ftypes(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].ftypes
        funcs = ['df.ftypes']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_values(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].values
        funcs = ['df.values']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_axes(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].axes
        funcs = ['df.axes']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_ndim(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].ndim
        funcs = ['df.ndim']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_size(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].size
        funcs = ['df.size']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_shape(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].shape
        funcs = ['df.shape']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_T(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].T
        funcs = ['df.T']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_as_matrix(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].as_matrix(['A', 'C', 'B'])
        funcs = ['df.as_matrix']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_as_matrix_2(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].as_matrix()
        funcs = ['df.as_matrix']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_get_dtype_counts(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].get_dtype_counts()
        funcs = ['df.get_dtype_counts']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_get_ftype_counts(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].get_ftype_counts()
        funcs = ['df.get_ftype_counts']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_select_dtypes(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0][['A', 'C']]
        funcs = ['df.select_dtypes']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_astype(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0][['A', 'C']].astype('int32')
        funcs = ['df.astype', 'df.__getitem__']
        seqs = [[1, 0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_isna(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].isna()
        funcs = ['df.isna']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_notna(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].notna()
        funcs = ['df.notna']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_head(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].head(2)
        funcs = ['df.head']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_at_getitem(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].at[(1, 'B')]
        funcs = ['df.at_getitem']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_at_getitem_2(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].at[(2, 'A')]
        funcs = ['df.at_getitem']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_iat_getitem(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].iat[(1, 1)]
        funcs = ['df.iat_getitem']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_iat_getitem_2(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].iat[(2, 2)]
        funcs = ['df.iat_getitem']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_loc_getitem(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].loc[1:3, 'B':'A':(- 1)]
        funcs = ['df.loc_getitem']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_loc_getitem_2(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].loc[[1, 3], 'C':'A':(- 1)].head(1)
        funcs = ['df.loc_getitem', 'df.head']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_iloc_getitem(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].iloc[1:3, 1:0:(- 1)]
        funcs = ['df.iloc_getitem']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_iloc_getitem_2(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].iloc[[1, 3], 2:0:(- 1)].head(1)
        funcs = ['df.iloc_getitem', 'df.head']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_lookup(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].lookup([0, 2], ['A', 'C'])
        funcs = ['df.lookup']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_lookup_2(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].lookup([0, 1], ['A', 'B'])
        funcs = ['df.lookup']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_xs(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].xs(0, axis=0)
        funcs = ['df.xs']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_xs_2(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].xs('C', axis=1)
        funcs = ['df.xs']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_xs_3(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                                             ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].xs(['one', 'bar'], level=[1, 0])
        funcs = ['df.xs']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_xs_4(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                                             ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        inputs = [inputs[0].T]
        output = inputs[0].xs(['one', 'bar'], level=[1, 0], axis=1)
        funcs = ['df.xs']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_tail(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0].tail(2)
        funcs = ['df.tail']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_isin(self):
        constants = [[1, 3, 12, 'a']]
        inputs = [pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'f'],
        })]
        output = inputs[0].isin([1, 3, 12, 'a'])
        funcs = ['df.isin']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_where(self):
        constants = [(lambda _df: ((_df % 3) == 0))]
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B']),
                  (- pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B']))]
        output = inputs[0].where(constants[0], (- inputs[1]))
        funcs = ['df.where']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_mask(self):
        constants = [(lambda _df: ((_df % 3) == 0))]
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B']),
                  (- pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B']))]
        output = inputs[0].mask(constants[0], (- inputs[1]))
        funcs = ['df.mask']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_query(self):
        constants = ['a > b']
        inputs = [pd.DataFrame(np.random.randn(10, 2), columns=list('ab'))]
        output = inputs[0].query('a > b')
        funcs = ['df.query']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_getitem(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0]['A']
        funcs = ['df.__getitem__']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_getitem_2(self):
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': pd.Timestamp('20130102'),
            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
        })]
        output = inputs[0][['B', 'C', 'A']]
        funcs = ['df.__getitem__']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_add(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].add(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.add']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_sub(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].sub(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.sub']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_mul(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].mul(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.mul']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_div(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].div(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.div']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_truediv(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].truediv(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.truediv']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_floordiv(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].floordiv(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.floordiv']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_mod(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].mod(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.mod']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_pow(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].pow(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.pow']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_radd(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].radd(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.radd']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_rsub(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].rsub(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.rsub']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_rmul(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].rmul(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.rmul']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_rdiv(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].rdiv(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.rdiv']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_rtruediv(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].rtruediv(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.rtruediv']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_rfloordiv(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].rfloordiv(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.rfloordiv']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_rmod(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].rmod(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.rmod']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_rpow(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].rpow(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.rpow']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_lt(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].lt(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.lt']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_gt(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].gt(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.gt']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_le(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].le(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.le']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_ge(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].ge(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.ge']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_ne(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].ne(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.ne']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_eq(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].eq(inputs[0]['A'], axis=0)
        funcs = ['df.__getitem__', 'df.eq']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_combine(self):
        constants = [(lambda s1, s2: (s1 if (s1.sum() < s2.sum()) else s2))]
        inputs = [pd.DataFrame({
            'A': [0, 0],
            'B': [4, 4],
        }), pd.DataFrame({
            'A': [1, 1],
            'B': [3, 3],
        })]
        output = inputs[0].combine(inputs[1], constants[0])
        funcs = ['df.combine']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_combine_first(self):
        inputs = [pd.DataFrame([[1, np.nan]]), pd.DataFrame([[3, 4]])]
        output = inputs[0].combine_first(inputs[1])
        funcs = ['df.combine_first']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_apply(self):
        constants = [np.sum]
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].apply(np.sum, axis=0)
        funcs = ['df.apply']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_apply_2(self):
        constants = [np.sum]
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])]
        output = inputs[0].apply(np.sum, axis=1)
        funcs = ['df.apply']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_applymap(self):
        constants = [(lambda x: ('%.2f' % x))]
        inputs = [pd.DataFrame(np.random.randn(3, 3))]
        output = inputs[0].applymap((lambda x: ('%.2f' % x)))
        funcs = ['df.applymap']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_agg(self):
        constants = [['sum', 'min']]
        inputs = [
            pd.DataFrame(np.random.randn(10, 3), columns=['A', 'B', 'C'], index=pd.date_range('1/1/2000', periods=10))]
        output = inputs[0].agg(['sum', 'min'])
        funcs = ['df.agg']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_agg_2(self):
        constants = [{
            'A': ['sum', 'min'],
            'B': ['min', 'max'],
        }]
        inputs = [
            pd.DataFrame(np.random.randn(10, 3), columns=['A', 'B', 'C'], index=pd.date_range('1/1/2000', periods=10))]
        output = inputs[0].agg({
            'A': ['sum', 'min'],
            'B': ['min', 'max'],
        })
        funcs = ['df.agg']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_transform(self):
        constants = [(lambda x: ((x - x.mean()) / x.std()))]
        inputs = [
            pd.DataFrame(np.random.randn(10, 3), columns=['A', 'B', 'C'], index=pd.date_range('1/1/2000', periods=10))]
        output = inputs[0].transform((lambda x: ((x - x.mean()) / x.std())))
        funcs = ['df.transform']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_groupby(self):
        inputs = [pd.DataFrame({
            'A': [1, 1, 1, 1, 2, 2, 3, 3],
            'B': np.arange(8),
        }, index=pd.MultiIndex.from_arrays([['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                            ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']],
                                           names=['first', 'second']))]
        output = inputs[0].groupby(by=['A', 'second'], axis=0).sum()
        funcs = ['df.groupby', 'dfgroupby.sum']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_df_groupby_2(self):
        inputs = [pd.DataFrame({
            'a': [1, 0, 0],
            'b': [0, 1, 0],
            'c': [1, 0, 0],
            'd': [2, 3, 4],
        })]
        output = inputs[0].groupby(inputs[0].sum(), axis=1).sum()
        funcs = ['df.sum', 'df.groupby', 'dfgroupby.sum']
        seqs = [[0, 1, 2]]
        self.check(inputs, output, funcs, seqs)

    def test_df_abs(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                                             ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].abs()
        funcs = ['df.abs']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_all(self):
        inputs = [pd.DataFrame({
            'A': [1, 1, 1, 1, 2, 2, 3, 3],
            'B': np.arange(8),
        }, index=pd.MultiIndex.from_arrays([['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                            ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']],
                                           names=['first', 'second']))]
        output = inputs[0].all(axis=0, level=1)
        funcs = ['df.all']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_any(self):
        inputs = [pd.DataFrame({
            'A': [1, 1, 1, 1, 2, 2, 3, 3],
            'B': np.arange(8),
        }, index=pd.MultiIndex.from_arrays([['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                            ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']],
                                           names=['first', 'second']))]
        output = inputs[0].any(axis=0, level=1)
        funcs = ['df.any']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_clip(self):
        constants = [(- 0.3), 0.5]
        inputs = [pd.DataFrame(np.random.randn(8, 4), index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                                             ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].clip(lower=(- 0.3), upper=0.5)
        funcs = ['df.clip']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_clip_lower(self):
        constants = [(- 0.3), 0.5]
        inputs = [pd.DataFrame(np.random.randn(8, 4), index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                                             ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].clip_lower((- 0.3), axis=0)
        funcs = ['df.clip_lower']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_clip_lower_2(self):
        constants = [(- 0.3), 0.5]
        inputs = [pd.DataFrame(np.random.randn(8, 4), index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                                             ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].clip_lower((- 0.3), axis=1)
        funcs = ['df.clip_lower']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_clip_upper(self):
        constants = [(- 0.3), 0.5]
        inputs = [pd.DataFrame(np.random.randn(8, 4), index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                                             ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].clip_upper((- 0.3), axis=0)
        funcs = ['df.clip_upper']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_clip_upper_2(self):
        constants = [(- 0.3), 0.5]
        inputs = [pd.DataFrame(np.random.randn(8, 4), index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                                             ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].clip_upper((- 0.3), axis=1)
        funcs = ['df.clip_upper']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_corr(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                                             ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].corr(method='kendall')
        funcs = ['df.corr']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_corr_2(self):
        constants = [2]
        inputs = [pd.DataFrame(np.random.randn(8, 4), index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                                             ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].corr(method='spearman', min_periods=2)
        funcs = ['df.corr']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_corrwith(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                                             ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]),
                  pd.DataFrame(np.random.randn(8, 4), index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                                             ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].corrwith(inputs[1], axis=0)
        funcs = ['df.corrwith']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_corrwith_2(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                                             ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]),
                  pd.DataFrame(np.random.randn(8, 4), index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                                             ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].corrwith(inputs[1], axis=1)
        funcs = ['df.corrwith']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_count(self):
        inputs = [pd.DataFrame({
            'lkey': ['foo', 'foo', 'bar', 'bar', 'baz', np.NaN],
            'value_x': [1, 4, 2, 2, 3, np.NaN],
            'rkey': ['foo', 'foo', 'bar', 'bar', np.NaN, 'qux'],
            'value_y': [5, 5, 6, 8, np.NaN, 7],
        })[['lkey', 'value_x', 'rkey', 'value_y']]]
        output = inputs[0].count(axis=0)
        funcs = ['df.count']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_count_2(self):
        inputs = [pd.DataFrame({
            'lkey': ['foo', 'foo', 'bar', 'bar', 'baz', np.NaN],
            'value_x': [1, 4, 2, 2, 3, np.NaN],
            'rkey': ['foo', 'foo', 'bar', 'bar', np.NaN, 'qux'],
            'value_y': [5, 5, 6, 8, np.NaN, 7],
        })[['lkey', 'value_x', 'rkey', 'value_y']]]
        output = inputs[0].count(axis=1)
        funcs = ['df.count']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_cov(self):
        constants = [2]
        inputs = [pd.DataFrame(np.random.randn(8, 4), index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                                             ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].cov(min_periods=2)
        funcs = ['df.cov']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_cummax(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                                             ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].cummax(axis=0)
        funcs = ['df.cummax']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_cummax_2(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                                             ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].cummax(axis=1)
        funcs = ['df.cummax']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_cummin(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                                             ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].cummin(axis=0)
        funcs = ['df.cummin']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_cummin_2(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                                             ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].cummin(axis=1)
        funcs = ['df.cummin']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_cumprod(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                                             ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].cumprod(axis=0)
        funcs = ['df.cumprod']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_cumprod_2(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                                             ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].cumprod(axis=1)
        funcs = ['df.cumprod']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_cumsum(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                                             ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].cumsum(axis=0)
        funcs = ['df.cumsum']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_cumsum_2(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                                             ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].cumsum(axis=1)
        funcs = ['df.cumsum']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_diff(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                                             ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].diff(axis=0)
        funcs = ['df.diff']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_diff_2(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                                             ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].diff(axis=1)
        funcs = ['df.diff']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_diff_3(self):
        constants = [3]
        inputs = [pd.DataFrame(np.random.randn(8, 4), index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                                             ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].diff(periods=3, axis=0)
        funcs = ['df.diff']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_diff_4(self):
        constants = [3]
        inputs = [pd.DataFrame(np.random.randn(8, 4), index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                                             ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].diff(periods=3, axis=1)
        funcs = ['df.diff']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_eval(self):
        constants = ['a + b']
        inputs = [pd.DataFrame(np.random.randn(8, 4), columns=['a', 'b', 'c', 'd'],
                               index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                      ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].eval('a + b')
        funcs = ['df.eval']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_kurt(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), columns=['a', 'b', 'c', 'd'],
                               index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                      ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].kurt(axis=0, level=1)
        funcs = ['df.kurt']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_mad(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), columns=['a', 'b', 'c', 'd'],
                               index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                      ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].mad(axis=0, level=1)
        funcs = ['df.mad']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_max(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), columns=['a', 'b', 'c', 'd'],
                               index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                      ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].max(axis=0, level=1)
        funcs = ['df.max']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_mean(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), columns=['a', 'b', 'c', 'd'],
                               index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                      ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].mean(axis=0, level=1)
        funcs = ['df.mean']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_median(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), columns=['a', 'b', 'c', 'd'],
                               index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                      ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].median(axis=0, level=1)
        funcs = ['df.median']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_min(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), columns=['a', 'b', 'c', 'd'],
                               index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                      ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].min(axis=0, level=1)
        funcs = ['df.min']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_mode(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), columns=['a', 'b', 'c', 'd'],
                               index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                      ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].mode(axis=0)
        funcs = ['df.mode']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_pct_change(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), columns=['a', 'b', 'c', 'd'],
                               index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                      ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].pct_change()
        funcs = ['df.pct_change']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_pct_change_2(self):
        constants = [3]
        inputs = [pd.DataFrame(np.random.randn(8, 4), columns=['a', 'b', 'c', 'd'],
                               index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                      ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].pct_change(periods=3)
        funcs = ['df.pct_change']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_prod(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), columns=['a', 'b', 'c', 'd'],
                               index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                      ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].prod(axis=0, level=1)
        funcs = ['df.prod']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_quantile(self):
        constants = [0.1, 0.5]
        inputs = [pd.DataFrame(np.array([[1, 1], [2, 10], [3, 100], [4, 100]]), columns=['a', 'b'])]
        output = inputs[0].quantile(0.1)
        funcs = ['df.quantile']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_quantile_2(self):
        constants = [[0.1, 0.5]]
        inputs = [pd.DataFrame(np.array([[1, 1], [2, 10], [3, 100], [4, 100]]), columns=['a', 'b'])]
        output = inputs[0].quantile([0.1, 0.5])
        funcs = ['df.quantile']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_rank(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), columns=['a', 'b', 'c', 'd'],
                               index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                      ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].rank(axis=1, method='dense')
        funcs = ['df.rank']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_round(self):
        constants = [1]
        inputs = [pd.DataFrame(np.random.randn(8, 4), columns=['a', 'b', 'c', 'd'],
                               index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                      ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].round(decimals=1)
        funcs = ['df.round']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_sem(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), columns=['a', 'b', 'c', 'd'],
                               index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                      ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].sem(axis=0, ddof=2)
        funcs = ['df.sem']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_skew(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), columns=['a', 'b', 'c', 'd'],
                               index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                      ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].skew(axis=0, level=1)
        funcs = ['df.skew']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_sum(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), columns=['a', 'b', 'c', 'd'],
                               index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                      ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].sum(axis=0, level=1)
        funcs = ['df.sum']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_std(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), columns=['a', 'b', 'c', 'd'],
                               index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                      ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].std(axis=0, ddof=2)
        funcs = ['df.std']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_var(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), columns=['a', 'b', 'c', 'd'],
                               index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                      ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].var(axis=0, ddof=2)
        funcs = ['df.var']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_add_prefix(self):
        constants = ['#']
        inputs = [pd.DataFrame(np.random.randn(8, 4), columns=['a', 'b', 'c', 'd'],
                               index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                      ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].add_prefix('#')
        funcs = ['df.add_prefix']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_add_suffix(self):
        constants = ['#']
        inputs = [pd.DataFrame(np.random.randn(8, 4), columns=['a', 'b', 'c', 'd'],
                               index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                      ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].add_suffix('#')
        funcs = ['df.add_suffix']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_align(self):
        inputs = [pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B']),
                  pd.DataFrame(np.arange(10).reshape((- 1), 2), columns=['A', 'B'])[['A']].tail(2)]
        output = inputs[0].align(inputs[1], axis=0, join='left')
        funcs = ['df.align']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_drop(self):
        constants = ['#']
        inputs = [pd.DataFrame(np.random.randn(8, 4), columns=['a', 'b', 'c', 'd'],
                               index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                      ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].drop(labels=['one'], axis=0, level=1)
        funcs = ['df.drop']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_drop_2(self):
        constants = ['#']
        inputs = [pd.DataFrame(np.random.randn(8, 4), columns=['a', 'b', 'c', 'd'],
                               index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                      ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].drop(index=['bar', 'baz'], level=0)
        funcs = ['df.drop']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_drop_duplicates(self):
        inputs = [pd.DataFrame({
            'A': [1, 1, 1, 1, 2, 2, 3, 3],
            'B': np.arange(8),
        }, index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                  ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].drop_duplicates(subset=['A'], keep='last')
        funcs = ['df.drop_duplicates']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_duplicated(self):
        inputs = [pd.DataFrame({
            'A': [1, 1, 1, 1, 2, 2, 3, 3],
            'B': np.arange(8),
        }, index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                  ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].duplicated(subset=['A'], keep='last')
        funcs = ['df.duplicated']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_equals(self):
        inputs = [pd.DataFrame({
            'A': [1, 1, 1, 1, 2, 2, 3, 3],
            'B': np.arange(8),
        }, index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                  ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].equals(inputs[0])
        funcs = ['df.equals']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_filter(self):
        inputs = [pd.DataFrame({
            'one': [1, 4],
            'two': [2, 5],
            'three': [3, 6],
        }, index=['mouse', 'rabbit'])]
        output = inputs[0].filter(items=['one', 'three'])
        funcs = ['df.filter']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_filter_2(self):
        constants = ['e$']
        inputs = [pd.DataFrame({
            'one': [1, 4],
            'two': [2, 5],
            'three': [3, 6],
        }, index=['mouse', 'rabbit'])]
        output = inputs[0].filter(regex='e$', axis=1)
        funcs = ['df.filter']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_filter_3(self):
        constants = ['bbi']
        inputs = [pd.DataFrame({
            'one': [1, 4],
            'two': [2, 5],
            'three': [3, 6],
        }, index=['mouse', 'rabbit'])]
        output = inputs[0].filter(like='bbi', axis=0)
        funcs = ['df.filter']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_first(self):
        constants = ['1D']
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': [pd.Timestamp('20130102'), pd.Timestamp('20130104')],
            'C': pd.Series(1, index=list(range(2)), dtype='float64'),
        }).set_index('B')]
        output = inputs[0].first('1D')
        funcs = ['df.first']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_idxmax(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), columns=['a', 'b', 'c', 'd'],
                               index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                      ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].idxmax(axis=0)
        funcs = ['df.idxmax']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_idxmin(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), columns=['a', 'b', 'c', 'd'],
                               index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                      ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].idxmin(axis=0)
        funcs = ['df.idxmin']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_last(self):
        constants = ['1D']
        inputs = [pd.DataFrame({
            'A': 1.0,
            'B': [pd.Timestamp('20130102'), pd.Timestamp('20130104')],
            'C': pd.Series(1, index=list(range(2)), dtype='float64'),
        }).set_index('B')]
        output = inputs[0].last('1D')
        funcs = ['df.last']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_reindex(self):
        constants = ['missing']
        inputs = [pd.DataFrame({
            'http_status': [200, 200, 404, 404, 301],
            'response_time': [0.04, 0.02, 0.07, 0.08, 1.0],
        }, index=['Firefox', 'Chrome', 'Safari', 'IE10', 'Konqueror'])]
        output = inputs[0].reindex(['Safari', 'Iceweasel', 'Comodo Dragon', 'IE10', 'Chrome'], fill_value='missing')
        funcs = ['df.reindex']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_reindex_2(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), columns=['a', 'b', 'c', 'd'],
                               index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                      ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].reindex(labels=['two', 'one'], axis=0, level=1)
        funcs = ['df.reindex']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_reindex_like(self):
        inputs = [pd.DataFrame(np.random.randn(8, 4), columns=['a', 'b', 'c', 'd'],
                               index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                      ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]),
                  pd.DataFrame(np.random.randn(8, 4), columns=['a', 'b', 'c', 'd'],
                               index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                      ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]).reindex(
                      labels=['two', 'one'], axis=0, level=1)]
        output = inputs[0].reindex(labels=['two', 'one'], axis=0, level=1)
        funcs = ['df.reindex_like']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_rename(self):
        constants = [str.lower]
        inputs = [pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
        })]
        output = inputs[0].rename(columns=str.lower)
        funcs = ['df.rename']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_reset_index(self):
        constants = ['genus']
        inputs = [pd.DataFrame([(389.0, 'fly'), (24.0, 'fly'), (80.5, 'run'), (np.nan, 'jump')],
                               index=pd.MultiIndex.from_tuples(
                                   [('bird', 'falcon'), ('bird', 'parrot'), ('mammal', 'lion'), ('mammal', 'monkey')],
                                   names=['class', 'name']),
                               columns=pd.MultiIndex.from_tuples([('speed', 'max'), ('species', 'type')]))]
        output = inputs[0].reset_index(level='class', col_level=1, col_fill='genus')
        funcs = ['df.reset_index']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_set_index(self):
        inputs = [pd.DataFrame([(389.0, 'fly'), (24.0, 'fly'), (80.5, 'run'), (np.nan, 'jump')],
                               index=pd.MultiIndex.from_tuples(
                                   [('bird', 'falcon'), ('bird', 'parrot'), ('mammal', 'lion'), ('mammal', 'monkey')],
                                   names=['class', 'name']),
                               columns=pd.MultiIndex.from_tuples([('speed', 'max'), ('species', 'type')]))]
        output = inputs[0].set_index([('speed', 'max')], append=True)
        funcs = ['df.set_index']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_take(self):
        constants = [[1, 2]]
        inputs = [pd.DataFrame([('falcon', 'bird', 389.0), ('parrot', 'bird', 24.0), ('lion', 'mammal', 80.5),
                                ('monkey', 'mammal', np.nan)], columns=('name', 'class', 'max_speed'),
                               index=[0, 2, 3, 1])]
        output = inputs[0].take([1, 2], axis=1)
        funcs = ['df.take']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_dropna(self):
        inputs = [pd.DataFrame([[np.nan, 2, np.nan, 0], [3, 4, np.nan, 1], [np.nan, np.nan, np.nan, 5]],
                               columns=list('ABCD'))]
        output = inputs[0].dropna(axis=1, how='all')
        funcs = ['df.dropna']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_dropna_2(self):
        inputs = [pd.DataFrame([[np.nan, 2, np.nan, 0], [3, 4, np.nan, 1], [np.nan, np.nan, np.nan, 5]],
                               columns=list('ABCD'))]
        output = inputs[0].dropna(axis=0, how='any')
        funcs = ['df.dropna']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_dropna_3(self):
        constants = [2]
        inputs = [pd.DataFrame([[np.nan, 2, np.nan, 0], [3, 4, np.nan, 1], [np.nan, np.nan, np.nan, 5]],
                               columns=list('ABCD'))]
        output = inputs[0].dropna(thresh=2)
        funcs = ['df.dropna']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_fillna(self):
        inputs = [pd.DataFrame(
            [[np.nan, 2, np.nan, 0], [3, 4, np.nan, 1], [np.nan, np.nan, np.nan, 5], [np.nan, 3, np.nan, 4]],
            columns=list('ABCD'))]
        output = inputs[0].fillna(0)
        funcs = ['df.fillna']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_fillna_2(self):
        constants = [{
            'A': 0,
            'B': 1,
            'C': 2,
            'D': 3,
        }]
        inputs = [pd.DataFrame(
            [[np.nan, 2, np.nan, 0], [3, 4, np.nan, 1], [np.nan, np.nan, np.nan, 5], [np.nan, 3, np.nan, 4]],
            columns=list('ABCD'))]
        output = inputs[0].fillna({
            'A': 0,
            'B': 1,
            'C': 2,
            'D': 3,
        })
        funcs = ['df.fillna']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_fillna_3(self):
        constants = [{
            'A': 0,
            'B': 1,
            'C': 2,
            'D': 3,
        }]
        inputs = [pd.DataFrame(
            [[np.nan, 2, np.nan, 0], [3, 4, np.nan, 1], [np.nan, np.nan, np.nan, 5], [np.nan, 3, np.nan, 4]],
            columns=list('ABCD'))]
        output = inputs[0].fillna({
            'A': 0,
            'B': 1,
            'C': 2,
            'D': 3,
        }, limit=1)
        funcs = ['df.fillna']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_pivot_table(self):
        inputs = [pd.DataFrame([(389.0, 'fly', 'a'), (24.0, 'fly', 'b'), (80.5, 'run', 'c'), (np.nan, 'jump', 'd')],
                               index=pd.MultiIndex.from_tuples(
                                   [('bird', 'falcon'), ('bird', 'parrot'), ('mammal', 'lion'), ('mammal', 'monkey')],
                                   names=['class', 'name']), columns=pd.MultiIndex.from_tuples(
                [('speed', 'max'), ('species', 'type'), ('species', 'hello')]))]
        output = inputs[0].pivot_table(values=[('speed', 'max')], index=[('species', 'hello')],
                                       columns=[('species', 'type')])
        funcs = ['df.pivot_table']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_pivot_table_2(self):
        constants = [0]
        inputs = [pd.DataFrame([(389.0, 'fly', 'a'), (24.0, 'fly', 'b'), (80.5, 'run', 'c'), (np.nan, 'jump', 'd')],
                               index=pd.MultiIndex.from_tuples(
                                   [('bird', 'falcon'), ('bird', 'parrot'), ('mammal', 'lion'), ('mammal', 'monkey')],
                                   names=['class', 'name']), columns=pd.MultiIndex.from_tuples(
                [('speed', 'max'), ('species', 'type'), ('species', 'hello')]))]
        output = inputs[0].pivot_table(values=[('speed', 'max')], index=[('species', 'hello')],
                                       columns=[('species', 'type')], fill_value=0.0)
        funcs = ['df.pivot_table']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs, constants=constants)

    def test_df_pivot(self):
        inputs = [pd.DataFrame({
            'foo': ['one', 'one', 'one', 'two', 'two', 'two'],
            'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
            'baz': [1, 2, 3, 4, 5, 6],
        })]
        output = inputs[0].pivot(index='foo', columns='bar', values='baz')
        funcs = ['df.pivot']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_pivot_2(self):
        inputs = [pd.DataFrame({
            'foo': ['one', 'one', 'one', 'two', 'two', 'two'],
            'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
            'baz': [1, 2, 3, 4, 5, 6],
        })]
        output = inputs[0].pivot(columns='bar', values='baz')
        funcs = ['df.pivot']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_pivot_3(self):
        inputs = [pd.DataFrame({
            'foo': ['one', 'one', 'one', 'two', 'two', 'two'],
            'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
            'baz': [1, 2, 3, 4, 5, 6],
        })]
        output = inputs[0].pivot(columns='bar')
        funcs = ['df.pivot']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_reorder_levels(self):
        inputs = [pd.DataFrame([(389.0, 'fly', 'a'), (24.0, 'fly', 'b'), (80.5, 'run', 'c'), (np.nan, 'jump', 'd')],
                               index=pd.MultiIndex.from_tuples(
                                   [('bird', 'falcon'), ('bird', 'parrot'), ('mammal', 'lion'), ('mammal', 'monkey')],
                                   names=['class', 'name']), columns=pd.MultiIndex.from_tuples(
                [('speed', 'max'), ('species', 'type'), ('species', 'hello')]))]
        output = inputs[0].reorder_levels([1, 0], axis=0)
        funcs = ['df.reorder_levels']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_sort_values(self):
        inputs = [pd.DataFrame(np.random.randint(1, 33, (8, 4)),
                               index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                      ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].sort_values([0, 2], axis=0)
        funcs = ['df.sort_values']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_sort_values_2(self):
        inputs = [pd.DataFrame(np.random.randint(1, 33, (8, 4)),
                               index=[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                      ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']])]
        output = inputs[0].sort_values([('bar', 'two'), ('baz', 'one')], axis=1)
        funcs = ['df.sort_values']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_stack(self):
        inputs = [pd.DataFrame([(389.0, 'fly', 'a'), (24.0, 'fly', 'b'), (80.5, 'run', 'c'), (np.nan, 'jump', 'd')],
                               index=pd.MultiIndex.from_tuples(
                                   [('bird', 'falcon'), ('bird', 'parrot'), ('mammal', 'lion'), ('mammal', 'monkey')],
                                   names=['class', 'name']), columns=pd.MultiIndex.from_tuples(
                [('speed', 'max'), ('species', 'type'), ('species', 'hello')]))]
        output = inputs[0].stack(level=[1, 0])
        funcs = ['df.stack']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_unstack(self):
        inputs = [pd.DataFrame([(389.0, 'fly', 'a'), (24.0, 'fly', 'b'), (80.5, 'run', 'c'), (np.nan, 'jump', 'd')],
                               index=pd.MultiIndex.from_tuples(
                                   [('bird', 'falcon'), ('bird', 'parrot'), ('mammal', 'lion'), ('mammal', 'monkey')],
                                   names=['class', 'name']), columns=pd.MultiIndex.from_tuples(
                [('speed', 'max'), ('species', 'type'), ('species', 'hello')]))]
        output = inputs[0].unstack(level=[1, 0])
        funcs = ['df.unstack']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_melt(self):
        inputs = [pd.DataFrame([(389.0, 'fly'), (24.0, 'fly'), (80.5, 'run'), (np.nan, 'jump')],
                               index=pd.MultiIndex.from_tuples(
                                   [('bird', 'falcon'), ('bird', 'parrot'), ('mammal', 'lion'), ('mammal', 'monkey')],
                                   names=['class', 'name']),
                               columns=pd.MultiIndex.from_tuples([('speed', 'max'), ('species', 'type')]))]
        output = inputs[0].melt(id_vars=[('species', 'type')], value_vars=[('speed', 'max')])
        funcs = ['df.melt']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_melt_2(self):
        inputs = [pd.DataFrame({
            'A': {
                0: 'a',
                1: 'b',
                2: 'c',
            },
            'B': {
                0: 1,
                1: 3,
                2: 5,
            },
            'C': {
                0: 2,
                1: 4,
                2: 6,
            },
        })]
        output = inputs[0].melt(id_vars=['A'], value_vars=['C', 'B'])
        funcs = ['df.melt']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_merge(self):
        inputs = [pd.DataFrame({
            'lkey': ['foo', 'bar', 'baz', 'foo'],
            'value': [1, 2, 3, 4],
        }), pd.DataFrame({
            'rkey': ['foo', 'bar', 'qux', 'bar'],
            'value': [5, 6, 7, 8],
        })]
        output = pd.DataFrame({
            'lkey': ['foo', 'foo', 'bar', 'bar', 'baz', np.NaN],
            'value_x': [1, 4, 2, 2, 3, np.NaN],
            'rkey': ['foo', 'foo', 'bar', 'bar', np.NaN, 'qux'],
            'value_y': [5, 5, 6, 8, np.NaN, 7],
        })
        funcs = ['df.merge']
        seqs = [[0]]
        self.check(inputs, output, funcs, seqs)

    def test_df_merge_2(self):
        inputs = [pd.DataFrame({
            'lkey': ['foo', 'bar', 'baz', 'foo'],
            'value': [1, 2, 3, 4],
        }), pd.DataFrame({
            'rkey': ['foo', 'bar', 'qux', 'bar'],
            'value': [5, 6, 7, 8],
        })]
        output = pd.DataFrame({
            'lkey': ['foo', 'foo', 'bar', 'bar', 'baz', np.NaN],
            'value_x': [1, 4, 2, 2, 3, np.NaN],
            'rkey': ['foo', 'foo', 'bar', 'bar', np.NaN, 'qux'],
            'value_y': [5, 5, 6, 8, np.NaN, 7],
        })[['rkey', 'lkey', 'value_x', 'value_y']]
        funcs = ['df.merge', 'df.__getitem__']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_dfgroupby_count(self):
        inputs = [pd.DataFrame({
            'A': [1, 1, 1, 1, 2, 2, 3, 3],
            'B': np.arange(8),
        }, index=pd.MultiIndex.from_arrays([['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                            ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']],
                                           names=['first', 'second']))]
        output = inputs[0].groupby(by=['A', 'second'], axis=0).count()
        funcs = ['df.groupby', 'dfgroupby.count']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_dfgroupby_first(self):
        inputs = [pd.DataFrame({
            'A': [1, 1, 1, 1, 2, 2, 3, 3],
            'B': np.arange(8),
        }, index=pd.MultiIndex.from_arrays([['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                            ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']],
                                           names=['first', 'second']))]
        output = inputs[0].groupby(by=['A', 'second'], axis=0).first()
        funcs = ['df.groupby', 'dfgroupby.first']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_dfgroupby_last(self):
        inputs = [pd.DataFrame({
            'A': [1, 1, 1, 1, 2, 2, 3, 3],
            'B': np.arange(8),
        }, index=pd.MultiIndex.from_arrays([['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                            ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']],
                                           names=['last', 'second']))]
        output = inputs[0].groupby(by=['A', 'second'], axis=0).last()
        funcs = ['df.groupby', 'dfgroupby.last']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_dfgroupby_max(self):
        inputs = [pd.DataFrame({
            'A': [1, 1, 1, 1, 2, 2, 3, 3],
            'B': np.arange(8),
        }, index=pd.MultiIndex.from_arrays([['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                            ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']],
                                           names=['max', 'second']))]
        output = inputs[0].groupby(by=['A', 'second'], axis=0).max()
        funcs = ['df.groupby', 'dfgroupby.max']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_dfgroupby_mean(self):
        inputs = [pd.DataFrame({
            'A': [1, 1, 1, 1, 2, 2, 3, 3],
            'B': np.arange(8),
        }, index=pd.MultiIndex.from_arrays([['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                            ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']],
                                           names=['mean', 'second']))]
        output = inputs[0].groupby(by=['A', 'second'], axis=0).mean()
        funcs = ['df.groupby', 'dfgroupby.mean']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_dfgroupby_median(self):
        inputs = [pd.DataFrame({
            'A': [1, 1, 1, 1, 2, 2, 3, 3],
            'B': np.arange(8),
        }, index=pd.MultiIndex.from_arrays([['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                            ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']],
                                           names=['median', 'second']))]
        output = inputs[0].groupby(by=['A', 'second'], axis=0).median()
        funcs = ['df.groupby', 'dfgroupby.median']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    def test_dfgroupby_min(self):
        inputs = [pd.DataFrame({
            'A': [1, 1, 1, 1, 2, 2, 3, 3],
            'B': np.arange(8),
        }, index=pd.MultiIndex.from_arrays([['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                                            ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']],
                                           names=['min', 'second']))]
        output = inputs[0].groupby(by=['A', 'second'], axis=0).min()
        funcs = ['df.groupby', 'dfgroupby.min']
        seqs = [[0, 1]]
        self.check(inputs, output, funcs, seqs)

    # def test_pandas_concat_1(self):
    #     inputs = [pd.DataFrame(columns=['City', 'Population'], index=[0, 1], data=[['Zagreb', 700000], ['Rijeka', 142000]]), pd.DataFrame(columns=['City', 'Area'], index=[0, 1, 2], data=[['Split', 200.0], ['Osijek', 171.0], ['Dubrovnik', 143.35]])]
    #     output = pd.concat([inputs[0], inputs[1]], ignore_index=True)
    #     funcs = ['pandas.concat']
    #     seqs = [[0]]
    #     self.check(inputs, output, funcs, seqs)
    #
    # def test_pandas_concat_2(self):
    #     columns = ['Open', 'High', 'Low']
    #     index = pd.Index(['2011-01-03', '2011-01-04', '2011-01-05', '2011-01-06', '2011-01-07'], name='Date')
    #     goog = [[21.01, 21.05, 20.78], [21.12, 21.2, 21.05], [21.19, 21.21, 20.9], [20.67, 20.82, 20.55], [20.71, 20.77, 20.27]]
    #     aapl = [[596.48, 605.59, 596.48], [605.62, 606.18, 600.12], [600.07, 610.33, 600.05], [610.68, 618.43, 610.05], [615.91, 618.25, 610.13]]
    #     inputs = [pd.DataFrame(columns=columns, index=index, data=goog), pd.DataFrame(columns=columns, index=index, data=aapl)]
    #     output = pd.concat(inputs[0], keys=['GOOG', 'AAPL'], axis=1)
    #     funcs = ['pandas.concat']
    #     seqs = [[0]]
    #     self.check(inputs, output, funcs, seqs)
    #
    # def test_pandas_concat_3(self):
    #     inputs = [pd.DataFrame(columns=['City', 'Population'], index=[0, 1], data=[['Zagreb', 700000], ['Rijeka', 142000]]), pd.DataFrame(columns=['City', 'Area'], index=[0, 1, 2], data=[['Split', 200.0], ['Osijek', 171.0], ['Dubrovnik', 143.35]])]
    #     output = pd.concat([inputs[0], inputs[1]], join='inner')
    #     funcs = ['pandas.concat']
    #     seqs = [[0]]
    #     self.check(inputs, output, funcs, seqs)
