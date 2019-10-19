from typing import Any

from autopandas_v2.iospecs import SearchSpec


class Value:
    def __init__(self, val: Any):
        self.val = val


class Default(Value):
    def __eq__(self, o: object) -> bool:
        try:
            if not isinstance(o, Value):
                return self.val == o

            if isinstance(o, Value):
                return self.val == o.val

            return False

        except:
            return False

    def __ne__(self, o: object) -> bool:
        return not self.__eq__(o)

    def __hash__(self) -> int:
        return hash(self.val)

    def __init__(self, val: Any):
        super().__init__(val)


class Inactive(Value):
    def __init__(self, val: Any):
        super().__init__(val)


class AnnotatedVal(Value):
    def __init__(self, val: Any, **kwargs):
        super().__init__(val)
        self.annotations = kwargs


class Fetcher(Value):
    def __init__(self, val: Any, source: str, idx: int):
        super().__init__(val)
        self.source = source
        self.idx = idx
        if source == 'inps':
            self.repr = 'inps[{}]'.format(idx)
        elif source == 'intermediates':
            self.repr = 'v{}'.format(idx)
        else:
            raise NotImplementedError("Source {} is not supported right now".format(source))

    def __call__(self, spec: SearchSpec):
        if self.source == 'inputs':
            return spec.inputs[self.idx]

        elif self.source == 'intermediates':
            return spec.intermediates[self.idx]


class RandomColumn(Value):
    """
    TODO : Come up with a better, more general solution. Ideally we should also track the randomness
    TODO : while generating training data so we can invert that as well
    """

    def __ne__(self, o: object) -> bool:
        return not self.__eq__(o)

    def __hash__(self) -> int:
        return hash(self.val)

    def __init__(self, val: Any):
        super().__init__(val)

    def __eq__(self, o: object) -> bool:
        try:
            if not isinstance(o, Value):
                return self.val == o

            if isinstance(o, Value):
                return self.val == o.val

            return False

        except:
            return False
