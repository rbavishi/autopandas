from typing import Any, Callable, Sized, Iterable, Collection
import pandas as pd
import numpy as np
from autopandas_v2.generators.dsl.values import Value

from autopandas_v2.utils import logger
pd_groupby = pd.core.groupby.GroupBy


class Checker:
    @staticmethod
    def check(v1: Any, v2: Any) -> bool:
        if isinstance(v1, Value):
            return Checker.check(v1.val, v2)

        return Checker.get_checker(v1)(v1, v2)

    @staticmethod
    def get_checker(v1: Any) -> Callable[[Any, Any], bool]:
        if isinstance(v1, pd.DataFrame):
            return Checker.check_dataframe
        if isinstance(v1, pd.Series):
            return Checker.check_series
        if isinstance(v1, pd_groupby):
            return Checker.check_groupby
        if isinstance(v1, np.ndarray):
            return Checker.check_ndarray
        if isinstance(v1, str):
            return Checker.check_default
        if isinstance(v1, Collection):
            return Checker.check_collection

        return Checker.check_default

    @staticmethod
    def check_dataframe(v1: pd.DataFrame, v2: Any) -> bool:
        if not isinstance(v2, pd.DataFrame):
            return False

        try:
            pd.testing.assert_frame_equal(v1, v2)
            return True
        except (AssertionError, TypeError, ValueError):
            return False
        except Exception as e:
            logger.warn("DataFrame Comparison Failed")
            logger.log(v1)
            logger.log(v2)
            logger.log(e)
            return False

    @staticmethod
    def check_series(v1: pd.Series, v2: Any) -> bool:
        if not isinstance(v2, pd.Series):
            return False

        try:
            pd.testing.assert_series_equal(v1, v2)
            return True
        except (AssertionError, TypeError, ValueError):
            return False
        except Exception as e:
            logger.warn("Series Comparison Failed")
            logger.log(v1)
            logger.log(v2)
            logger.log(e)
            return False

    @staticmethod
    def check_groupby(v1: pd_groupby, v2: Any) -> bool:
        if not isinstance(v2, pd_groupby):
            return False

        try:
            return all(v1.apply(lambda x: x.equals(v2.get_group(x.name)) if x.name in v2.groups else False))
        except (AssertionError, TypeError, ValueError, NameError):
            return False
        except Exception as e:
            logger.warn("GroupBy Comparison Failed")
            logger.log(v1)
            logger.log(v2)
            logger.log(e)
            return False

    @staticmethod
    def check_index(v1: pd.Index, v2: Any) -> bool:
        if not isinstance(v2, pd.Index):
            return False

        try:
            pd.testing.assert_index_equal(v1, v2)
            return True
        except (AssertionError, TypeError, ValueError):
            return False
        except Exception as e:
            logger.warn("Index Comparison Failed")
            logger.log(v1)
            logger.log(v2)
            logger.log(e)
            return False

    @staticmethod
    def check_ndarray(v1: np.ndarray, v2: Any) -> bool:
        if not isinstance(v2, np.ndarray):
            return False

        try:
            return np.array_equal(v1, v2)
        except Exception as e:
            logger.warn("NDArray Comparison Failed")
            logger.log(v1)
            logger.log(v2)
            logger.log(e)
            return False

    @staticmethod
    def check_collection(v1: Collection, v2: Any) -> bool:
        if (not isinstance(v2, Collection)) or isinstance(v2, str):
            return False

        if len(v1) != len(v2):
            return False

        try:
            for i, j in zip(v1, v2):
                if not Checker.check(i, j):
                    return False

        except Exception as e:
            logger.warn("Collection Comparison Failed")
            logger.log(v1)
            logger.log(v2)
            logger.log(e)
            return False

        return True

    @staticmethod
    def check_default(v1: Any, v2: Any) -> bool:
        try:
            if v1 == v2:
                #  I know what I'm doing
                #  v1 == v2 is not guaranteed to return a bool
                #  This is to capture that
                return True
            else:
                return False

        except (AssertionError, TypeError, ValueError, NameError):
            return False
        except Exception as e:
            logger.warn("Default Comparison Failed")
            logger.log(v1)
            logger.log(v2)
            logger.log(e)
            return False


