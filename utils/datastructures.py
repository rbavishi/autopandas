import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import orderedset


def oset(vals=None):
    """
    Behaves like a set, but preserves the order in vals
    Does this by exploiting the insertion order preservation in sets from Python 3.6+
    YES, I don't care about backward compatibility.
    Complexity should be O(nlogn) but that's not going to be the critical path anyway
    """
    if vals is not None:
        return orderedset.OrderedSet(vals)
    else:
        return orderedset.OrderedSet()
