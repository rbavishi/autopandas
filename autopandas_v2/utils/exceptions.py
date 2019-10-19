import sys


def quiet_hook(kind, message, traceback):
    if SilentException in kind.__bases__:
        print('{0}: {1}'.format(kind.__name__, message))  # Only print Error Type and Message
    else:
        sys.__excepthook__(kind, message, traceback)  # Print Error Type, Message and Traceback


sys.excepthook = quiet_hook


class AutoPandasException(Exception):
    pass


class AutoPandasInversionFailedException(AutoPandasException):
    pass


class SilentException(Exception):
    pass
