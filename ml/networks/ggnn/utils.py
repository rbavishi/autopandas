import argparse


class ParamsNamespace(argparse.Namespace):
    @staticmethod
    def from_namespace(namespace: argparse.Namespace):
        res = ParamsNamespace()
        for k, v in namespace.__dict__.items():
            setattr(res, k, v)

        return res

    def get(self, key, default):
        return self.__dict__.get(key, default)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def update(self, other):
        if isinstance(other, argparse.Namespace):
            self.__dict__.update(other.__dict__)

        elif isinstance(other, dict):
            self.__dict__.update(other)

        else:
            raise NotImplementedError("Cannot update ParamsNamespace with {}".format(type(other)))
