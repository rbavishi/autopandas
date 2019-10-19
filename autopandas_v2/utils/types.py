import numpy as np


class DType:
    def __init__(self, dtype):
        self.dtype = dtype

    def hasinstance(self, other):
        if isinstance(self.dtype, list):
            for t in self.dtype:
                if isinstance(other, t):
                    return True

            return False

        return isinstance(other, self.dtype)

    def is_superclass_of(self, other: 'DType'):
        try:
            return issubclass(other.dtype, self.dtype)
        except:
            return False

    def is_subclass_of(self, other: 'DType'):
        try:
            return issubclass(self.dtype, other.dtype)
        except:
            return False


class FType(DType):
    def hasinstance(self, other):
        return self.dtype(other)

    def is_subclass_of(self, other: 'DType'):
        raise NotImplementedError("Can't subclass for FType")

    def is_superclass_of(self, other: 'DType'):
        raise NotImplementedError("Can't superclass for FType")


class Lambda:
    def __init__(self, fn: str = None):
        self.fn = fn

    def __str__(self):
        return self.fn

    def __repr__(self):
        return self.fn

    def __call__(self, *args, **kwargs):
        return eval(self.fn)(*args, **kwargs)


def is_float(val):
    return np.issubdtype(type(val), np.floating)


def is_int(val):
    return np.issubdtype(type(val), np.signedinteger) or np.issubdtype(type(val), np.unsignedinteger)
