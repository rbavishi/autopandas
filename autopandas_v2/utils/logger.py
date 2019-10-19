from timeit import default_timer as timer

#  Colors
from typing import Set

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
DEFAULT = "\033[39m"

TAG_INFO = '[{}INFO{}]'.format(BLUE, DEFAULT)
TAG_ERR = '[{}ERROR{}]'.format(RED, DEFAULT)
TAG_WARN = '[{}WARN{}]'.format(YELLOW, DEFAULT)

warn_cache: Set[str] = set()


def debug(*args, **kwargs):
    print(*args, **kwargs)


def log(*args, **kwargs):
    print(*args, **kwargs)


def info(*args, **kwargs):
    print(TAG_INFO, *args, **kwargs)


def err(*args, **kwargs):
    print(TAG_ERR, *args, **kwargs)


def warn(*args, use_cache=False, **kwargs):
    if use_cache:
        if len(args) == 1 and args[0] in warn_cache:
            return

        warn_cache.add(args[0])

    print(TAG_WARN, *args, **kwargs)


def log_result(msg, *args, **kwargs):
    print("[{}{}{}]".format(GREEN, msg, DEFAULT), *args, **kwargs)


def get_time():
    return timer()


def get_time_elapsed(end_time, start_time=0):
    return round(end_time - start_time, 2)
