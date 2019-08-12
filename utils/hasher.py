from collections.abc import Hashable
from typing import Any

import pandas as pd
import numpy as np


class Hasher:
    @staticmethod
    def hash(val: Any) -> int:
        if not isinstance(val, Hashable):
            return hash(str(val))

        try:
            try:
                #  This returns a Series object
                #  So just take its string representation
                #  and return a hash of that
                h = pd.util.hash_pandas_object(val)
                return hash(str(h))

            except TypeError:
                pass

            if isinstance(val, np.ndarray):
                try:
                    #  This returns a ndarray object
                    #  So just take its string representation
                    #  and return a hash of that
                    h = pd.util.hash_array(val)
                    return hash(str(h))
                except TypeError:
                    pass

            return hash(val)

        except:
            try:
                return hash(str(val))
            except:
                return -1
