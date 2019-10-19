import glob
import logging
import os
import unittest
from typing import List, Any

import pandas as pd

from autopandas_v2.generators.ml.networks.mocking.models import MockSelectModel, MockChainModel, MockSubsetsModel
from autopandas_v2.iospecs import IOSpec
from autopandas_v2.synthesis.search.engines.functions import BFSEngine, BaseEngine
from autopandas_v2.utils import logger


class TestBeamSearchEngine(unittest.TestCase):
    def setUp(self):
        pass

    def check(self, inputs: List[Any], output: Any, funcs: List[str], seqs: List[List[int]], model_dir: str):
        iospec: IOSpec = IOSpec(inputs, output)
        iospec.funcs = funcs
        iospec.seqs = seqs

        engine: BFSEngine = BFSEngine(iospec)
        engine.max_depth = max(len(i) for i in seqs)
        engine.silent = True
        engine.stop_first_solution = True
        engine.use_spec_funcs = True
        engine.use_spec_seqs = True
        engine.argument_engine = 'beam-search'
        engine.arg_model_dir = model_dir

        # try:
        #     engine.search()
        # except Exception as e:
        #     logging.exception(e)

        return self.check_engine(engine)

    def check_engine(self, engine: BaseEngine):
        self.assertTrue(engine.search(), msg='Did not find a solution')
        return engine.stats

    def test_pivot_selects_1(self):
        #    foo bar  baz
        # 0  one   A    1
        # 1  one   B    2
        # 2  one   C    3
        # 3  two   A    4
        # 4  two   B    5
        # 5  two   C    6
        inputs = [pd.DataFrame({
            'foo': ['one', 'one', 'one', 'two', 'two', 'two'],
            'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
            'baz': [1, 2, 3, 4, 5, 6],
        })]

        # bar  A  B  C
        # foo
        # one  1  2  3
        # two  4  5  6R
        output = inputs[0].pivot(index='foo', columns='bar', values='baz')

        mocker_columns = MockSelectModel(behavior={
            'select_columns_1': [(1.0, 'bar'), (0.0, 'foo'), (0.0, 'baz')]
        })

        mocker_index = MockSelectModel(behavior={
            'select_index_1': [(0.9, 'foo'), (0.1, 'baz')]
        })

        mocker_values = MockSelectModel(behavior={
            'select_values_1': [(1.0, 'baz'), (0.0, 'bar'), (0.0, 'foo')]
        })

        mocker_columns.save('/tmp/mock_autopandas_pivot_1/df.pivot/select_columns_1')
        mocker_index.save('/tmp/mock_autopandas_pivot_1/df.pivot/select_index_1')
        mocker_values.save('/tmp/mock_autopandas_pivot_1/df.pivot/select_values_1')
        funcs = ['df.pivot']
        seqs = [[0]]

        stats = self.check(inputs, output, funcs, seqs, '/tmp/mock_autopandas_pivot_1')
        self.assertEqual(stats.num_cands_generated[1], 1)
        self.assertEqual(stats.num_cands_error[1], 0)

    def test_pivot_selects_2(self):
        #    foo bar  baz
        # 0  one   A    1
        # 1  one   B    2
        # 2  one   C    3
        # 3  two   A    4
        # 4  two   B    5
        # 5  two   C    6
        inputs = [pd.DataFrame({
            'foo': ['one', 'one', 'one', 'two', 'two', 'two'],
            'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
            'baz': [1, 2, 3, 4, 5, 6],
        })]

        # bar  A  B  C
        # foo
        # one  1  2  3
        # two  4  5  6R
        output = inputs[0].pivot(index='foo', columns='bar', values='baz')

        mocker_columns = MockSelectModel(behavior={
            'select_columns_1': [(1.0, 'bar'), (0.0, 'foo'), (0.0, 'baz')]
        })

        mocker_index = MockSelectModel(behavior={
            'select_index_1': [(0.1, 'foo'), (0.9, 'baz')]
        })

        mocker_values = MockSelectModel(behavior={
            'select_values_1': [(1.0, 'baz'), (0.0, 'bar'), (0.0, 'foo')]
        })

        mocker_columns.save('/tmp/mock_autopandas_pivot_2/df.pivot/select_columns_1')
        mocker_index.save('/tmp/mock_autopandas_pivot_2/df.pivot/select_index_1')
        mocker_values.save('/tmp/mock_autopandas_pivot_2/df.pivot/select_values_1')
        funcs = ['df.pivot']
        seqs = [[0]]

        stats = self.check(inputs, output, funcs, seqs, '/tmp/mock_autopandas_pivot_2')
        self.assertEqual(stats.num_cands_generated[1], 2)
        self.assertEqual(stats.num_cands_error[1], 0)
