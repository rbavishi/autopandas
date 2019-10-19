from io import StringIO

import pandas as pd
import numpy as np
from autopandas_v2.evaluation.benchmarks.base import Benchmark


class PandasBenchmarks:
    # https://stackoverflow.com/questions/11881165
    class SO_11881165_depth1(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [pd.DataFrame({"a": [5, 6, 7, 8, 9], "b": [10, 11, 12, 13, 14]})]
            self.output = self.inputs[0].loc[[0, 2, 4]]
            self.funcs = ['df.loc_getitem']
            self.seqs = [[0]]

    # https://stackoverflow.com/questions/11941492/
    # same thing
    class SO_11941492_depth1(Benchmark):
        def __init__(self):
            super().__init__()
            df = pd.DataFrame({'group1': ['a', 'a', 'a', 'b', 'b', 'b'],
                               'group2': ['c', 'c', 'd', 'd', 'd', 'e'],
                               'value1': [1.1, 2, 3, 4, 5, 6],
                               'value2': [7.1, 8, 9, 10, 11, 12]
                               })
            df = df.set_index(['group1', 'group2'])
            self.inputs = [df]
            self.output = df.xs('a', level=0)
            self.funcs = ['df.xs']
            self.seqs = [[0]]

    # https://stackoverflow.com/questions/13647222
    class SO_13647222_depth1(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [
                pd.DataFrame({'series': {0: 'A', 1: 'B', 2: 'C', 3: 'A', 4: 'B', 5: 'C', 6: 'A', 7: 'B',
                                         8: 'C', 9: 'A', 10: 'B', 11: 'C', 12: 'A', 13: 'B', 14: 'C'},
                              'step': {0: '100', 1: '100', 2: '100', 3: '101', 4: '101', 5: '101', 6: '102', 7: '102',
                                       8: '102', 9: '103', 10: '103', 11: '103', 12: '104', 13: '104', 14: '104'},
                              'value': {0: '1000', 1: '1001', 2: '1002', 3: '1003', 4: '1004', 5: '1005', 6: '1006',
                                        7: '1007',
                                        8: '1008', 9: '1009', 10: '1010', 11: '1011', 12: '1012', 13: '1013',
                                        14: '1014'}})
            ]
            self.output = self.inputs[0].pivot(columns='series', values='value', index='step')
            self.funcs = ['df.pivot']
            self.seqs = [[0]]

    # https://stackoverflow.com/questions/18172851/
    class SO_18172851_depth1(Benchmark):
        def __init__(self):
            super().__init__()
            df = pd.DataFrame({'daysago': {'2007-03-31': 62, '2007-03-10': 83, '2007-02-10': 111, '2007-01-13': 139,
                                           '2006-12-23': 160, '2006-11-09': 204, '2006-10-22': 222, '2006-09-29': 245,
                                           '2006-09-16': 258, '2006-08-30': 275, '2006-02-11': 475, '2006-01-13': 504,
                                           '2006-01-02': 515, '2005-12-06': 542, '2005-11-29': 549, '2005-11-22': 556,
                                           '2005-11-01': 577, '2005-10-20': 589, '2005-09-27': 612, '2005-09-07': 632,
                                           '2005-06-12': 719, '2005-05-29': 733, '2005-05-02': 760, '2005-04-02': 790,
                                           '2005-03-13': 810, '2004-11-09': 934},
                               'line_race': {'2007-03-31': 111, '2007-03-10': 211, '2007-02-10': 29, '2007-01-13': 110,
                                             '2006-12-23': 210, '2006-11-09': 39, '2006-10-22': 28, '2006-09-29': 49,
                                             '2006-09-16': 311, '2006-08-30': 48, '2006-02-11': 45, '2006-01-13': 0,
                                             '2006-01-02': 0, '2005-12-06': 0, '2005-11-29': 0, '2005-11-22': 0,
                                             '2005-11-01': 0, '2005-10-20': 0, '2005-09-27': 0, '2005-09-07': 0,
                                             '2005-06-12': 0, '2005-05-29': 0, '2005-05-02': 0, '2005-04-02': 0,
                                             '2005-03-13': 0, '2004-11-09': 0},
                               'rw': {'2007-03-31': 0.99999, '2007-03-10': 0.97, '2007-02-10': 0.9,
                                      '2007-01-13': 0.8806780000000001, '2006-12-23': 0.793033, '2006-11-09': 0.636655,
                                      '2006-10-22': 0.581946, '2006-09-29': 0.518825, '2006-09-16': 0.48622600000000005,
                                      '2006-08-30': 0.446667, '2006-02-11': 0.16459100000000002,
                                      '2006-01-13': 0.14240899999999998, '2006-01-02': 0.1348,
                                      '2005-12-06': 0.11780299999999999, '2005-11-29': 0.113758,
                                      '2005-11-22': 0.10985199999999999, '2005-11-01': 0.098919, '2005-10-20': 0.093168,
                                      '2005-09-27': 0.083063, '2005-09-07': 0.075171, '2005-06-12': 0.04869,
                                      '2005-05-29': 0.045404, '2005-05-02': 0.039679, '2005-04-02': 0.03416,
                                      '2005-03-13': 0.030914999999999998, '2004-11-09': 0.016647}})
            df['rating'] = range(2, 28)
            df['wrating'] = df['rw'] * df['rating']
            df = df[['daysago', 'line_race', 'rating', 'rw', 'wrating']]
            self.inputs = [df, lambda a: a.line_race != 0]
            self.output = self.inputs[0].loc[lambda a: a.line_race != 0]
            self.funcs = ['df.loc_getitem']
            self.seqs = [[0]]

    # https://stackoverflow.com/questions/49583055
    class SO_49583055_depth1(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [pd.DataFrame({'ID': {0: 20, 1: 21, 2: 22, 3: 32, 4: 31, 5: 33},
                                         'admit': {0: pd.Timestamp('2018-03-04 00:00:00'),
                                                   1: pd.Timestamp('2018-02-02 00:00:00'),
                                                   2: pd.Timestamp('2018-02-05 00:00:00'),
                                                   3: pd.Timestamp('2018-01-02 00:00:00'),
                                                   4: pd.Timestamp('2018-01-15 00:00:00'),
                                                   5: pd.Timestamp('2018-01-20 00:00:00')},
                                         'discharge': {0: pd.Timestamp('2018-03-06 00:00:00'),
                                                       1: pd.Timestamp('2018-02-06 00:00:00'),
                                                       2: pd.Timestamp('2018-02-23 00:00:00'),
                                                       3: pd.Timestamp('2018-02-03 00:00:00'),
                                                       4: pd.Timestamp('2018-01-18 00:00:00'),
                                                       5: pd.Timestamp('2018-01-24 00:00:00')},
                                         'discharge_location': {0: 'Home1', 1: 'Home2', 2: 'Home3', 3: 'Home4',
                                                                4: 'Home5',
                                                                5: 'Home6'},
                                         'first': {0: 11, 1: 10, 2: 9, 3: 8, 4: 12, 5: 7}})]
            self.output = self.inputs[0].sort_values(by=['ID', 'first', 'admit'], ascending=[True, False, True])
            self.funcs = ['df.sort_values']
            self.seqs = [[0]]

    # https://stackoverflow.com/questions/49592930
    # ok I didn't uniqify the timestamps because that would change the actual output
    class SO_49592930_depth1(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [pd.DataFrame({'value': {pd.Timestamp('2014-05-21 09:30:00'): 0.0,
                                                   pd.Timestamp('2014-05-21 10:00:00'): 10.0,
                                                   pd.Timestamp('2014-05-21 10:30:00'): 3.0,
                                                   pd.Timestamp('2017-07-10 22:30:00'): 18.3,
                                                   pd.Timestamp('2017-07-10 23:00:00'): 7.6,
                                                   pd.Timestamp('2017-07-10 23:30:00'): 2.0}}),
                           pd.DataFrame({'value': {pd.Timestamp('2014-05-21 09:00:00'): 1.0,
                                                   pd.Timestamp('2014-05-21 10:00:00'): 13.0,
                                                   pd.Timestamp('2017-07-10 21:00:00'): 1.6,
                                                   pd.Timestamp('2017-07-10 22:00:00'): 32.1,
                                                   pd.Timestamp('2017-07-10 23:00:00'): 7.7}})
                           ]
            self.output = self.inputs[0].combine_first(self.inputs[1])
            self.funcs = ['df.combine_first']
            self.seqs = [[0]]

    # https://stackoverflow.com/questions/49572546
    class SO_49572546_depth1(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [
                pd.DataFrame(
                    {'C1': {1: 100, 2: 102, 3: 103, 4: 104, 5: 105, 6: 106, 7: 107},
                     'C2': {1: 201, 2: 202, 3: 203, 4: 204, 5: 205, 6: 206, 7: 207},
                     'C3': {1: 301, 2: 302, 3: 303, 4: 304, 5: 305, 6: 306, 7: 307}}),
                pd.DataFrame(
                    {'C1': {2: '1002', 3: 'v1', 4: 'v4', 7: '1007'}, 'C2': {2: '2002', 3: 'v2', 4: 'v5', 7: '2007'},
                     'C3': {2: '3002', 3: 'v3', 4: 'v6', 7: '3007'}})
            ]
            self.output = self.inputs[1].combine_first(self.inputs[0])
            self.funcs = ['df.combine_first']
            self.seqs = [[0]]

    class SO_12860421_depth1(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [
                pd.DataFrame(columns=['X', 'Y', 'Z'],
                             index=[4, 5, 6, 7],
                             data=[['X1', 'Y2', 'Z3'], ['X1', 'Y1', 'Z1'], ['X1', 'Y1', 'Z1'], ['X1', 'Y1', 'Z2']]
                             ),

                pd.Series.nunique
            ]
            self.output = self.inputs[0].pivot_table(values='X', index='Y', columns='Z', aggfunc=pd.Series.nunique)

            self.funcs = ['df.pivot_table']
            self.seqs = [[0]]

    # https://stackoverflow.com/questions/13261175
    class SO_13261175_depth1(Benchmark):
        def __init__(self):
            super().__init__()
            df = pd.DataFrame({'name': ['A', 'B', 'A', 'B'], 'type': [11, 11, 12, 12],
                               'date': ['2012-01-01', '2012-01-01', '2012-02-01', '2012-02-01'], 'value': [4, 5, 6, 7]})

            pt = df.pivot_table(values='value', index='name', columns=['type', 'date'])
            self.inputs = [df]
            self.output = pt
            self.funcs = ['df.pivot_table']
            self.seqs = [[0]]

    # https://stackoverflow.com/questions/13793321
    class SO_13793321_depth1(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [
                pd.DataFrame([[11, 12, 13]], columns=[10, 1, 2]),
                pd.DataFrame([[11, 37, 38], [34, 19, 39]], columns=[10, 3, 4])
            ]
            self.output = self.inputs[0].merge(self.inputs[1], on=10)
            self.funcs = ['df.merge']
            self.seqs = [[0]]

    class SO_14085517_depth1(Benchmark):
        def __init__(self):
            super().__init__()
            text = '''\
SEGM1\tDESC\tDistribuzione Ponderata\tRotazioni a volume
AD2\tACCADINAROLO\t74.040\t140249.693409
AD1\tZYMIL AMALAT Z\t90.085\t321529.053570
FUN\tSPECIALMALAT S\t88.650\t120711.182177
NORM5\tSTD INNAROLO\t49.790\t162259.216710
NORM4\tSTD P.NAROLO\t52.125\t1252174.695695
NORM3\tSTD PLNAROLO\t54.230\t213257.829615
NORM1\tBONTA' MALAT B\t79.280\t520454.366419
NORM6\tDA STD RILGARD\t35.290\t554927.497875
NORM7\tOVANE VT.MANTO\t15.040\t466232.639628
NORM2\tWEIGHT MALAT W\t79.170\t118628.572692
'''
            from io import StringIO
            a = pd.read_csv(StringIO(text), delimiter='\t',
                            index_col=(0, 1), )
            self.inputs = [a]
            self.output = a.sort_values(['SEGM1', 'Distribuzione Ponderata'], ascending=[True, False])
            self.seqs = [[0]]
            self.funcs = ['df.sort_values']

    class SO_11418192_depth2(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [pd.DataFrame(data=[[5, 7], [6, 8], [-1, 9], [-2, 10]], columns=['a', 'b']),
                           lambda x: x['a'] > 1, 'a > 1']
            t = self.inputs[0]
            self.output = t[t.apply(lambda x: x['a'] > 1, axis=1)]
            self.funcs = ['df.apply', 'df.__getitem__']
            self.seqs = [[0, 1]]

    class SO_49567723_depth2(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [
                pd.DataFrame({'id': {0: 255, 1: 91, 2: 347, 3: 30, 4: 68, 5: 159, 6: 32, 7: 110, 8: 225, 9: 257},
                              'valueA': {0: 1141, 1: 1130, 2: 830, 3: 757, 4: 736, 5: 715, 6: 713, 7: 683, 8: 638,
                                         9: 616}}),
                pd.DataFrame({'id': {0: 255, 1: 91, 2: 5247, 3: 347, 4: 30, 5: 68,
                                     6: 159, 7: 32, 8: 110, 9: 225, 10: 257,
                                     11: 917, 12: 211, 13: 25},
                              'valueB': {0: 1231, 1: 1170, 2: 954, 3: 870, 4: 757,
                                         5: 736, 6: 734, 7: 713, 8: 683, 9: 644,
                                         10: 616, 11: 585, 12: 575, 13: 530}}),
                'valueA != valueB'
            ]
            self.output = self.inputs[0].merge(self.inputs[1], on=['id']).query('valueA != valueB')
            self.funcs = ['df.merge', 'df.query']
            self.seqs = [[0, 1]]

    # https://stackoverflow.com/questions/49987108
    class SO_49987108_depth2(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [
                pd.DataFrame({'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                              'COL': [23, np.nan, np.nan, np.nan, np.nan, 21, np.nan, np.nan, np.nan, 25, np.nan,
                                      np.nan]}).set_index('ID'),
                int
            ]
            self.output = self.inputs[0].fillna(method='ffill').astype(int)
            self.seqs = [[0, 1]]
            self.funcs = ['df.fillna', 'df.astype']

    # https://stackoverflow.com/questions/13261691
    # (there's also another potential q/a pair in this question)
    # or this could just be done with a sort????
    class SO_13261691_depth2(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [
                pd.DataFrame(
                    {'date': ['3/9/12', '3/10/12', '4/9/12', '9/9/12', '11/9/12', '30/9/12', '31/10/12', '1/11/12'],
                     'score': [100, 99, 102, 103, 111, 98, 103, 104]}, index=pd.MultiIndex.from_tuples(
                        [('A', 'John1'), ('B', 'John2'), ('B', 'Jane'), ('A', 'Peter'), ('C', 'Josie'),
                         ('A', 'Rachel'),
                         ('B', 'Kate'), ('C', 'David')], names=['team', 'name']))
            ]
            self.output = self.inputs[0].stack().unstack()
            self.funcs = ['df.stack', 'df.unstack']
            self.seqs = [[0, 1]]

    class SO_13659881_depth2(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [
                pd.DataFrame(
                    columns=['ip', 'useragent'],
                    index=[0, 1, 2, 3],
                    data=[['192.168.0.1', 'a'], ['192.168.0.1', 'a'], ['192.168.0.1', 'b'], ['192.168.0.2', 'b']]
                )
            ]
            self.output = self.inputs[0].groupby(['ip', 'useragent']).size()
            self.funcs = ['df.groupby', 'dfgroupby.size']
            self.seqs = [[0, 1]]

    # https://stackoverflow.com/questions/13807758
    class SO_13807758_depth2(Benchmark):
        def __init__(self):
            super().__init__()
            df1 = pd.DataFrame([[10], [11], [12], [14], [16], [18]])
            df1[::3] = np.nan
            self.inputs = [
                df1
            ]
            self.output = self.inputs[0].dropna().reset_index(drop=True)
            self.funcs = ['df.dropna', 'df.reset_index']
            self.seqs = [[0, 1]]

    # http://stackoverflow.com/questions/34365578/dplyr-filtering-based-on-two-variables
    class SO_34365578_depth2(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [
                pd.DataFrame({'Group': {0: 'A', 1: 'A', 2: 'A', 3: 'B', 4: 'B', 5: 'B'},
                              'Id': {0: 11, 1: 12, 2: 13, 3: 14, 4: 15, 5: 16},
                              'Var1': {0: 'good', 1: 'good', 2: 'bad', 3: 'good', 4: 'good', 5: 'bad'},
                              'Var2': {0: 20, 1: 26, 2: 29, 3: 23, 4: 23, 5: 28}}),
                "Group == \"A\"",
                'sum',
            ]
            self.output = self.inputs[0].query('Group == "A"').pivot_table(index='Group', columns='Var1', values='Var2',
                                                                           aggfunc='sum')
            self.funcs = ['df.query', 'df.pivot_table']
            self.seqs = [[0, 1]]

      # https://stackoverflow.com/questions/10982266
    class SO_10982266_depth3(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [pd.DataFrame(
                [['08:01:08', 'C', 'PXA', 20100101, 4000, 'A', 57.8, 60],
                 ['08:01:11', 'C', 'PXA', 20100101, 4000, 'A', 58.4, 60],
                 ['08:01:12', 'C', 'PXA', 20100101, 4000, 'A', 58.0, 60],
                 ['08:01:16', 'C', 'PXA', 20100101, 4000, 'A', 58.4, 60],
                 ['08:01:16', 'C', 'PXA', 20100101, 4000, 'A', 58.0, 60],
                 ['08:01:21', 'C', 'PXA', 20100101, 4000, 'A', 58.4, 60],
                 ['08:01:21', 'C', 'PXA', 20100101, 4000, 'A', 58.0, 60]],
                columns=['time', 'contract', 'ticker', 'expiry', 'strike', 'quote', 'price', 'volume'],
                index=[0, 1, 2, 3, 4, 5, 6]
            )]
            self.output = pd.DataFrame(
                [['08:01:08', 57.8, 60], ['08:01:11', 58.4, 60], ['08:01:12', 58.0, 60], ['08:01:16', 58.2, 60],
                 ['08:01:21', 58.2, 60]],
                columns=['time', 'price', 'volume'],
                index=[0, 1, 2, 3, 4]
            )
            self.funcs = ['df.groupby', 'dfgroupby.mean', 'df.__getitem__']
            self.seqs = [[0, 1, 2]]
            # original answer for input a:
            # pd.DataFrame([{'time': k,
            #                'price': (v.price * v.volume).sum() / v.volume.sum(),
            #                'volume': v.volume.mean()}
            #               for k, v in a.groupby(['time'])],
            #              columns=['time', 'price', 'volume'])

    # https://stackoverflow.com/questions/11811392
    class SO_11811392_depth3(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [pd.DataFrame(
                columns=['one', 'two', 'three', 'four', 'five'],
                index=[0, 1],
                data=[[1, 2, 3, 4, 5], [1, 1, 1, 1, 1]]
            )]
            # original has a tolist at the end, but we don't support that
            self.output = self.inputs[0].T.reset_index().values
            self.funcs = ['df.T', 'dfgroupby.reset_index', 'df.values']
            self.seqs = [[0, 1, 2]]

    # https://stackoverflow.com/questions/49581206
    class SO_49581206_depth3(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [
                pd.DataFrame({'A': {('col1', 'no'): 2, ('col1', 'yes'): 8, ('col2', 'no'): 2, ('col2', 'yes'): 6},
                              'B': {('col1', 'no'): 0, ('col1', 'yes'): 2, ('col2', 'no'): 1, ('col2', 'yes'): 1}}).T
            ]
            self.output = self.inputs[0].div(self.inputs[0].sum(1, level=0), 1, 0).xs('yes', 1, 1)

            self.funcs = ['df.sum', 'df.div', 'df.xs']
            self.seqs = [[0, 1, 2]]

    # https://stackoverflow.com/questions/12065885
    class SO_12065885_depth3(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [
                pd.DataFrame({'RPT_Date': {0: '1980-01-01', 1: '1980-01-02', 2: '1980-01-03', 3: '1980-01-04',
                                           4: '1980-01-05', 5: '1980-01-06', 6: '1980-01-07', 7: '1980-01-08',
                                           8: '1980-01-09', 9: '1980-01-10'},
                              'STK_ID': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9},
                              'STK_Name': {0: 'Arthur', 1: 'Beate', 2: 'Cecil', 3: 'Dana', 4: 'Eric', 5: 'Fidel',
                                           6: 'George', 7: 'Hans', 8: 'Ingrid', 9: 'Jones'},
                              'sales': {0: 0, 1: 4, 2: 2, 3: 8, 4: 4, 5: 5, 6: 4, 7: 7, 8: 7, 9: 4}}),
                [[4, 2, 6]]
            ]
            self.output = self.inputs[0][self.inputs[0].STK_ID.isin([4, 2, 6])]
            self.funcs = ['df.isin', 'df.getitem', 'df.loc_getitem']
            self.seqs = [[2, 0, 1]]

    # https://stackoverflow.com/questions/13576164
    class SO_13576164_depth3(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [
                pd.DataFrame(columns=['col1', 'to_merge_on'],
                             index=pd.MultiIndex.from_arrays([[1, 1, 2, 2], ['a', 'b', 'a', 'b']],
                                                             names=['id1', 'id2']),
                             data=[[1, 2], [3, 4], [1, 2], [3, 4]]),
                pd.DataFrame(columns=['col2', 'to_merge_on'],
                             index=[0, 1, 2],
                             data=[[1, 1], [2, 3], [3, 5]])
            ]
            self.output = self.inputs[0].reset_index().merge(self.inputs[1], how='left').set_index(
                ['id1', 'id2'])
            self.funcs = ['df.reset_index', 'df.merge', 'df.set_index']
            self.seqs = [[0, 1, 2]]

    # https://stackoverflow.com/questions/14023037
    class SO_14023037_depth3(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [
                pd.DataFrame(
                    {'id': [1, 2, 3, 4, 5, 6],
                     'col1': ['A1', 'A1', 'A1', 'A1', 'A2', 'A2'],
                     'col2': ['B1', 'B1', 'B2', 'B2', 'B1', 'B2'],
                     'col3': ['before', 'after', 'before', 'after', 'before', 'after'],
                     'value': [20, 13, 11, 21, 18, 22]},
                    columns=['id', 'col1', 'col2', 'col3', 'value'])
            ]
            self.output = self.inputs[0].pivot_table(values='value',
                                                     index=['col1', 'col2'],
                                                     columns=['col3']).fillna(method='bfill').dropna()
            self.funcs = ['df.pivot_table', 'df.fillna', 'df.dropna']
            self.seqs = [[0, 1, 2]]

    # https://stackoverflow.com/questions/53762029/pandas-groupby-and-cumsum-on-a-column
    class SO_53762029_depth3(Benchmark):
        def __init__(self):
            super().__init__()
            data = """
doc_created_month   doc_created_year    speciality      doc_id_count
8                   2016                Acupuncturist   1           
2                   2017                Acupuncturist   1           
4                   2017                Acupuncturist   1           
4                   2017                Allergist       1           
5                   2018                Allergist       1           
10                  2018                Allergist       2   
"""

            df = pd.read_csv(StringIO(data), sep='\s+')
            self.inputs = [df]
            self.output = df.groupby(['doc_created_month', 'doc_created_year', 'speciality']).sum().cumsum()
            self.funcs = ['df.groupby', 'dfgroupby.sum', 'df.cumsum']
            self.seqs = [[0, 1, 2]]

            # http://stackoverflow.com/questions/21982987/mean-per-group-in-a-data-frame

    class SO_21982987_depth3(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [
                pd.DataFrame({"Name": ["Aira", "Aira", "Ben", "Ben", "Cat", "Cat"], "Month": [1, 2, 1, 2, 1, 2],
                              "Rate1": [12, 18, 53, 22, 22, 27], "Rate2": [23, 73, 19, 87, 87, 43]}),
            ]
            self.output = pd.DataFrame({'Name': {0: 'Aira', 1: 'Ben', 2: 'Cat'}, 'Rate1': {0: 15.0, 1: 37.5, 2: 24.5},
                                        'Rate2': {0: 48.0, 1: 53.0, 2: 65.0}})
            self.seqs = [[0, 1, 2]]
            self.funcs = ['df.groupby', 'dfgroupby.mean', 'df.drop']

    # http://stackoverflow.com/questions/39656670/pivot-table-on-r-using-dplyr-or-tidyr
    class SO_39656670_depth3(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [
                pd.DataFrame(
                    {"Player": ["Abdoun", "Abe", "Abidal", "Abreu"], "Team": ["Algeria", "Japan", "France", "Uruguay"],
                     "Shots": [0, 3, 0, 5], "Passes": [6, 101, 91, 15], "Tackles": [0, 14, 6, 0]}),
            ]
            self.output = self.inputs[0].melt(value_vars=["Passes", "Tackles"], var_name="Var",
                                              value_name="Mean").groupby(
                "Var", as_index=False).mean()
            self.seqs = [[0, 1, 2]]
            self.funcs = ['df.melt', 'df.groupby', 'dfgroupby.mean']

    # http://stackoverflow.com/questions/23321300/efficient-method-to-filter-and-add-based-on-certain-conditions-3-conditions-in
    class SO_23321300_depth3(Benchmark):
        def __init__(self):
            super().__init__()
            self.inputs = [
                pd.DataFrame({"a": [1, 1, 1, 1, 1, 1, 1, 1, 1], "b": [1, 1, 1, 1, 1, 2, 2, 2, 3],
                              "d": [0, 200, 300, 0, 600, 0, 100, 200, 0]}),
                'd > 0'
            ]
            self.output = self.inputs[0].query('d > 0').groupby(['a', 'b']).mean()
            self.funcs = ['df.query', 'df.groupby', 'dfgroupby.mean']
            self.seqs = [[0, 1, 2]]
