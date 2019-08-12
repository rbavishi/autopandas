import pandas as pd

from autopandas_v2.evaluation.benchmarks.base import Benchmark


class PandasTestDepth1:
    class test_df_pivot(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [pd.DataFrame({
                'foo': ['one', 'one', 'one', 'two', 'two', 'two'],
                'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
                'baz': [1, 2, 3, 4, 5, 6],
            })]
            self.output = self.inputs[0].pivot(index='foo', columns='bar', values='baz')
            self.funcs = ['df.pivot']
            self.seqs = [[0]]
