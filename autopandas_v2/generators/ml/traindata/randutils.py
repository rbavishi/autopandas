import collections
import itertools
import math
import random
import string
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Dict, Type

import pandas as pd
import numpy as np


def biased_coin(p):
    return np.random.choice([1, 0], p=[p, 1 - p])


def RandStr(alphabet=string.ascii_letters, min_len=1, max_len=8):
    yield ''.join(random.choice(alphabet) for i in range(random.randint(min_len, max_len)))


def RandSensibleString(min_len=1, max_len=8, seps=[0]):
    alphabet = string.ascii_letters
    num_seps = len(seps)
    res_str = ""
    sep = "_"
    tmps = seps + [0]
    for i in range(0, num_seps + 1):
        if i > 0:
            res_str += sep
        seg_len = random.randint(min_len, (max_len + min_len) // 2)
        alphabet = string.ascii_letters if not tmps[i] else ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        res_str += "".join(random.choice(alphabet) for j in range(seg_len))
    return res_str


class ColGen(ABC):
    @abstractmethod
    def generate(self, num_rows: int, col_id: str = "col") -> Tuple[str, Dict[int, Any]]:
        pass


class BasicColGen(ColGen):
    def __init__(self, all_distinct: bool = None, num_distinct: int = -1, duplicates: bool = None,
                 all_equal_prob=0.3):
        self.all_distinct = all_distinct
        self.num_distinct = num_distinct
        self.duplicates = duplicates
        self.all_equal_prob = all_equal_prob

    def generate(self, num_rows: int, col_id: str = "col",
                 past_cols: Dict[Type, List] = None, feeding_prob: float = 0) -> Tuple[str, Dict[int, Any]]:

        if self.all_distinct is None and self.duplicates is None:
            all_distinct = random.choice([True, False])
            duplicates = random.choice([True, False])

        elif self.all_distinct is None:
            duplicates = self.duplicates
            if not duplicates:
                all_distinct = random.choice([True, False])
            else:
                all_distinct = False

        elif self.duplicates is None:
            all_distinct = self.all_distinct
            duplicates = random.choice([True, False])

        else:
            all_distinct = False
            duplicates = False

        if all_distinct:
            pool = set()
            while len(pool) < num_rows:
                pool.add(self.get_value_wrapper(past_cols, feeding_prob))

            vals = list(pool)

        elif duplicates:
            if self.num_distinct == -1 and biased_coin(self.all_equal_prob) == 1:
                vals = [self.get_value_wrapper(past_cols, feeding_prob)] * num_rows

            else:
                pool = set()
                while len(pool) < max(self.num_distinct, (num_rows - 1), 1):
                    pool.add(self.get_value_wrapper(past_cols, feeding_prob))
                pool = list(pool)

                if self.num_distinct != -1:
                    vals = random.sample(pool, min(max(0, self.num_distinct), len(pool) - 1))
                    try:
                        vals.append(random.choice(vals))
                    except:
                        print(pool, vals, num_rows - 1)
                        raise
                else:
                    vals = []

                while len(vals) < num_rows:
                    vals.append(random.choice(pool))

        else:
            #  Anything goes
            vals = []
            for i in range(num_rows):
                vals.append(self.get_value_wrapper(past_cols, feeding_prob))

        ret_col = {}
        random.shuffle(vals)
        for i, v in enumerate(vals[:num_rows]):
            ret_col[i] = v

        return col_id + self.get_suffix(), ret_col

    def get_value_wrapper(self, past_cols: Dict[Type, List], feeding_prob: float):
        cur_type = type(self)
        if past_cols is not None and len(past_cols.get(cur_type, [])) > 0:
            if biased_coin(feeding_prob) == 1:
                #  Feed from past columns
                chosen_col = random.choice(past_cols[cur_type])
                return random.choice(chosen_col)

        return self.get_value()

    def get_value(self) -> Any:
        raise NotImplementedError

    def get_suffix(self) -> str:
        return ""


class StrColGen(BasicColGen):
    def __init__(self, all_distinct: bool = None, num_distinct: int = -1, duplicates: bool = None,
                 alphabet: str = None, min_len: int = 1, max_len: int = 15):
        super().__init__(all_distinct, num_distinct, duplicates)
        if alphabet is None:
            self.alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-,:;.!?&()"
        else:
            self.alphabet = alphabet

        self.min_len = min_len
        self.max_len = max_len

    def get_value(self) -> Any:
        ret = ""
        for i in range(random.randint(self.min_len, self.max_len)):
            ret += random.choice(self.alphabet)
        return ret

    def get_suffix(self) -> str:
        suffix = "_str"
        if self.all_distinct:
            suffix += "_d"
        if self.duplicates:
            suffix += "_dup"

        return suffix


class SensibleStrColGen(BasicColGen):
    def __init__(self, all_distinct: bool = None, num_distinct: int = -1, duplicates: bool = None,
                 alphabet: str = None, min_len: int = 1, max_len: int = 15):
        super().__init__(all_distinct, num_distinct, duplicates)
        self.seps = [random.choice([0, 0, 1]) for i in range(0, random.choice([0, 0, 0, 1, 1, 2]))]
        self.min_len = min_len
        self.max_len = max_len

    def get_value(self) -> Any:
        return RandSensibleString(self.min_len, self.max_len, self.seps)

    def get_suffix(self) -> str:
        suffix = "_sens_str"
        if self.all_distinct:
            suffix += "_d"
        if self.duplicates:
            suffix += "_dup"

        return suffix


class IntColGen(BasicColGen):

    def __init__(self, all_distinct: bool = None, num_distinct: int = -1, duplicates: bool = None,
                 min_val: int = -1000, max_val: int = 1000):
        super().__init__(all_distinct, num_distinct, duplicates)
        self.min_val = min_val
        self.max_val = max_val

    def get_value(self) -> Any:
        return random.randint(self.min_val, self.max_val)  # inclusive

    def get_suffix(self) -> str:
        suffix = "_int"
        if self.all_distinct:
            suffix += "_d"
        if self.duplicates:
            suffix += "_dup"

        return suffix


class FloatColGen(BasicColGen):
    def __init__(self, all_distinct: bool = None, num_distinct: int = -1, duplicates: bool = None,
                 min_val: int = -1000, max_val: int = 1000, nan_prob: float = 0.05):
        super().__init__(all_distinct, num_distinct, duplicates)

        self.min_val = min_val
        self.max_val = max_val
        self.nan_prob = nan_prob

    def get_value(self) -> Any:
        if self.nan_prob is not None:
            if np.random.choice([0, 1], p=[self.nan_prob, 1 - self.nan_prob]) == 0:
                return np.nan

        return random.uniform(self.min_val, self.max_val)

    def get_suffix(self) -> str:
        suffix = "_float"
        if self.all_distinct:
            suffix += "_d"
        if self.duplicates:
            suffix += "_dup"

        return suffix


class BoolColGen(BasicColGen):

    def __init__(self, all_distinct: bool = None, num_distinct: int = -1, duplicates: bool = None):
        super().__init__(all_distinct, num_distinct, duplicates)
        self.all_distinct = False
        self.duplicates = False

    def get_value(self) -> Any:
        return random.choice([True, False])

    def get_suffix(self) -> str:
        suffix = "_bool"
        if self.all_distinct:
            suffix += "_d"
        if self.duplicates:
            suffix += "_dup"

        return suffix


class RandDf:
    def __init__(self, min_width: int = 1, min_height: int = 1,
                 max_width: int = 7, max_height: int = 7,
                 column_gens: List[ColGen] = None,
                 index_levels: int = None, column_levels: int = None,
                 max_index_levels: int = 3, max_column_levels: int = 3,
                 num_rows: int = None, num_columns: int = None,
                 int_col_prob=0.2, idx_mutation_prob=0.2,
                 multi_index_prob=0.2, col_prefix='',
                 col_feeding_prob=0.2, nan_prob=0.05):

        self.min_width = min_width
        self.min_height = min_height
        self.max_width = max_width
        self.max_height = max_height

        self.index_levels = index_levels
        self.column_levels = column_levels
        self.max_index_levels = max_index_levels
        self.max_column_levels = max_column_levels

        self.num_rows = num_rows
        self.num_columns = num_columns

        self.int_col_prob = int_col_prob
        self.idx_mutation_prob = idx_mutation_prob

        self.multi_index_prob = multi_index_prob

        self.col_prefix = col_prefix
        self.col_feeding_prob = col_feeding_prob

        if column_gens is None:
            self.column_gens = [StrColGen(), SensibleStrColGen(), StrColGen(duplicates=True),
                                SensibleStrColGen(duplicates=True), FloatColGen(nan_prob=nan_prob)]
        else:
            self.column_gens = column_gens

    def create_multi_index(self, index: pd.Index, num_levels: int, column_index=False) -> pd.MultiIndex:
        num_rows = len(index)
        vals = list(index)

        level_gens: List[ColGen] = [StrColGen(max_len=8), IntColGen(), SensibleStrColGen(max_len=8),
                                    StrColGen(max_len=8, duplicates=True), IntColGen(duplicates=True)]

        levels = [vals]

        for i in range(num_levels - 1):
            col_gen = random.choice(level_gens)
            col_id, col_vals = col_gen.generate(num_rows)
            levels.append(col_vals.values())

        return pd.MultiIndex.from_tuples(zip(*reversed(levels)))

    def __iter__(self):
        yield self.generate()

    def generate(self) -> pd.DataFrame:
        num_rows = random.randint(self.min_height, self.max_height) if self.num_rows is None else self.num_rows
        num_cols = random.randint(self.min_width, self.max_width) if self.num_columns is None else self.num_columns

        if np.random.choice([0, 1], p=[self.multi_index_prob, 1 - self.multi_index_prob]) == 0:
            index_levels = random.randint(2, self.max_index_levels) if self.index_levels is None else self.index_levels
        else:
            index_levels = 1 if self.index_levels is None else self.index_levels
        if np.random.choice([0, 1], p=[self.multi_index_prob, 1 - self.multi_index_prob]) == 0:
            column_levels = random.randint(2,
                                           self.max_column_levels) if self.column_levels is None else self.column_levels
        else:
            column_levels = 1 if self.column_levels is None else self.column_levels

        df_dict = {}
        past_cols: Dict[Type, List[Any]] = collections.defaultdict(list)

        for i in range(num_cols):
            col_gen: ColGen = random.choice(self.column_gens)
            col_id, col_vals = col_gen.generate(num_rows, col_id="{}col{}".format(self.col_prefix, i),
                                                past_cols=past_cols, feeding_prob=self.col_feeding_prob)
            past_cols[type(col_gen)].append(col_vals)
            df_dict[col_id] = col_vals

        df = pd.DataFrame(df_dict)
        if biased_coin(self.int_col_prob) == 1:
            df.columns = pd.Index(range(len(df.columns)))

        if biased_coin(self.idx_mutation_prob) == 1:
            idx_col_gen: ColGen = random.choice([StrColGen(max_len=8, all_distinct=True),
                                                 IntColGen(all_distinct=True)])
            try:
                df.index = pd.Index(list(idx_col_gen.generate(num_rows)[1].values()))
            except:
                print(idx_col_gen, df.index, num_rows)
                raise

        if index_levels == 1 and column_levels == 1:
            return df

        if index_levels > 1:
            df.index = self.create_multi_index(df.index, index_levels)

        if column_levels > 1:
            df.columns = self.create_multi_index(df.columns, column_levels, column_index=True)

        return df


class ValueBag:
    def __init__(self, values, name):
        self.values = values
        self.name = name

    def get_value(self):
        return random.choice(self.values)

    def get_name(self):
        return self.name


names_bag = ValueBag(["Amy", "Joseph", "Anne", "Kennedy", "Kira", "Brian", "Christie"], "name")
baz_bag = ValueBag(["foo", "bar", "baz", "fizz", "buzz"], "funcs")
fruits_bag = ValueBag(["kiwi", "apple", "bananer", "pear", "date", "cherimoya"], "Fruits")
countried_bag = ValueBag(["Canada", "India", "Germany", "Brazil", "US", "AlienLand"], "country")
things_bag_1 = ValueBag(
    ['_'.join(x) for x in itertools.product(['3', '7', '24', '1', '2', '18', '9', '7'], baz_bag.values)], "stuff1")
things_bag_2 = ValueBag(
    ['_'.join(x) for x in itertools.product(baz_bag.values, ['32', '71', '24', '1', '2', '18', '9', '7'])], "stuff2")
things_bag_3 = ValueBag([".".join(x) for x in itertools.product(fruits_bag.values, baz_bag.values)], "stats3")
uber_things_bag = ValueBag(["_".join(x) for x in itertools.product(fruits_bag.values, things_bag_2.values)], "stats4")
string_bags = [names_bag, baz_bag, fruits_bag, countried_bag, things_bag_2, things_bag_1, things_bag_3, uber_things_bag]

small_ints_bag = ValueBag(range(0, 24), "points")
five_ints_bag = ValueBag(range(15, 55, 5), "how_much")
moar_ints_bag = ValueBag([3, 123, 532, 391, 53, 483, 85, 584, 48, 68, 49], "moar")
big_ints_bag = ValueBag(range(400, 1000, 50), "stocks")
ints_bags = [small_ints_bag, five_ints_bag, moar_ints_bag, big_ints_bag]
small_floats_bag = ValueBag([0.1, 0.234, 0.7, 0.23411, 0.54327, 0.834953, 0.4, 0.81231, 0.9, np.NaN], "prob")
even_floats_bag = ValueBag([x / 2 for x in range(-10, 10, 1)], "div_by_twos")
big_floats_bag = ValueBag([132141.124, 132186.432, 3024234.234, 4234.4, 894324.5, 23894243.7, 123.4, np.NaN], "wats")
no_nans_floats_bag = ValueBag([71.3, 123.4, 32.4, 85.5, 23.7, 23.8, 83.7], "no_nans")
moar_nans_floats_bag = ValueBag([123.4, 2324.2, 213.789, 12.54, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN], "moar_nans")
bool_bags = [ValueBag([True, False], "boooools")]
floats_bags = [small_floats_bag, big_floats_bag, no_nans_floats_bag, even_floats_bag]


# find a factoring of length factors that gives a results close to num
def find_factoring_close_to(num, factors):
    if factors == 1 or num == 1:
        return [num]
    else:
        first = random.randint(2, math.ceil(num ** (1 / factors)))
        rest = math.ceil(num / first)
        return [first] + find_factoring_close_to(rest, factors - 1)


class NaturalRandDf:
    def __init__(self, min_width: int = 1, min_height: int = 1,
                 max_width: int = 7, max_height: int = 7,
                 value_bags: List[ColGen] = None,
                 index_levels: int = None, column_levels: int = None,
                 max_index_levels: int = 3, max_column_levels: int = 3,
                 num_rows: int = None, num_columns: int = None,
                 int_col_prob=0.2, idx_mutation_prob=0.2,
                 multi_index_prob=0.2, col_prefix='',
                 col_feeding_prob=0.2, indexy_columns_prob=0.35, nan_prob=0.0):
        self.min_width = min_width
        self.min_height = min_height
        self.max_width = max_width
        self.max_height = max_height

        self.index_levels = index_levels
        self.column_levels = column_levels
        self.max_index_levels = max_index_levels
        self.max_column_levels = max_column_levels

        self.num_rows = num_rows
        self.num_columns = num_columns

        self.int_col_prob = int_col_prob
        self.idx_mutation_prob = idx_mutation_prob

        self.multi_index_prob = multi_index_prob

        self.indexy_column_prob = indexy_columns_prob

        self.col_prefix = col_prefix
        self.col_feeding_prob = col_feeding_prob

        if value_bags is None:
            self.value_bags = [*string_bags, *ints_bags, *floats_bags]
        else:
            self.value_bags = value_bags

        if nan_prob > 0:
            self.value_bags.extend([moar_nans_floats_bag] * int(nan_prob * 10))

        # print(self.value_bags)

    def create_index_tuple_like(self, length, num_levels=None, total_num_cols=10, check_value_bags=True):
        if (length < 3) and num_levels is None:
            return [("IDX%i" % i,) for i in range(length)]
        if num_levels is None:
            levels = random.randint(1, min(3, total_num_cols))
        else:
            levels = num_levels

        if length < 3 and num_levels is not None:
            factoring = find_factoring_close_to(3, levels)
        else:
            factoring = find_factoring_close_to(length, levels)

        random.shuffle(factoring)
        level_values = []
        if check_value_bags and any(i not in self.value_bags for i in [*string_bags, *ints_bags]):
            return []

        for i in range(len(factoring)):
            bag = random.choice([random.choice(string_bags), random.choice(string_bags), random.choice(ints_bags)])
            level_values.append([bag.get_value() for _ in range(factoring[i])])
        ret_list = list(itertools.product(*level_values))
        while len(ret_list) > length:
            ret_list = random.sample(ret_list, length)
            # random_idx = random.randint(0, len(ret_list) - 1)
            # ret_list.pop(random_idx)
        return ret_list

    def tuple_list_to_dict(self, tuple_list):
        vals = [list(c) for c in zip(*tuple_list)]
        ret_dict = {"NAME%i" % i: vals[i] for i in range(len(vals))}
        return ret_dict

    def __iter__(self):
        yield self.generate()

    def generate(self) -> pd.DataFrame:
        num_rows = random.randint(self.min_height, self.max_height) if self.num_rows is None else self.num_rows
        num_cols = random.randint(self.min_width, self.max_width) if self.num_columns is None else self.num_columns

        if np.random.choice([0, 1], p=[self.multi_index_prob, 1 - self.multi_index_prob]) == 0:
            index_levels = random.randint(2, self.max_index_levels) if self.index_levels is None else self.index_levels
        else:
            index_levels = 1 if self.index_levels is None else self.index_levels
        if np.random.choice([0, 1], p=[self.multi_index_prob, 1 - self.multi_index_prob]) == 0:
            column_levels = random.randint(2,
                                           self.max_column_levels) if self.column_levels is None else self.column_levels
        else:
            column_levels = 1 if self.column_levels is None else self.column_levels

        df_dict = {}
        past_cols: Dict[Type, List[Any]] = collections.defaultdict(list)

        if biased_coin(self.indexy_column_prob) == 1:
            tuples = self.create_index_tuple_like(num_rows, total_num_cols=num_cols)
            if tuples:
                df_dict = self.tuple_list_to_dict(tuples)

        for i in range(num_cols - len(df_dict)):
            val_bag: ValueBag = random.choice(self.value_bags)
            col_name = self.col_prefix + val_bag.get_name()
            col_vals = [val_bag.get_value() for _ in range(num_rows)]
            while col_name in df_dict:
                col_name += str(random.randint(0, 10))
            df_dict[col_name] = col_vals

        df = pd.DataFrame(df_dict)
        if biased_coin(self.int_col_prob) == 1:
            df.columns = pd.Index(range(len(df.columns)))

        if biased_coin(self.idx_mutation_prob) == 1:
            idx_col_gen: ColGen = random.choice([StrColGen(max_len=8, all_distinct=True),
                                                 IntColGen(all_distinct=True)])
            try:
                df.index = pd.Index(list(idx_col_gen.generate(num_rows)[1].values()))
            except:
                print(idx_col_gen, df.index, num_rows)
                raise

        if index_levels == 1 and column_levels == 1:
            return df

        if index_levels > 1:
            tuples = self.create_index_tuple_like(num_rows, num_levels=index_levels, check_value_bags=False)
            if tuples:
                df.index = pd.MultiIndex.from_tuples(tuples)

        if column_levels > 1:
            tuples = self.create_index_tuple_like(num_cols, num_levels=column_levels, check_value_bags=False)
            if tuples:
                df.columns = pd.MultiIndex.from_tuples(tuples)

        return df
