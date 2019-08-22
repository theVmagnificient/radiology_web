""" Contains useful functions that are used by RadIO backend. """

import logging
import dill
import Pyro4
from .workitem import Workitem


Pyro4.config.SERIALIZER = 'pickle'
Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')


def map_logging_level(log_level: str) -> 'loglevel':
    """ Map string representation of logging level to logging constants. """
    if log_level == 'info':
        return logging.INFO
    elif log_level == 'debug':
        return logging.DEBUG
    elif log_level == 'warning':
        return logging.WARNING
    elif log_level == 'error':
        return logging.ERROR
    else:
        return logging.DEBUG


def get_prediction(client_id: str, scan_path: str,
                   fmt: str, timeout: float = None,
                   config: dict = None) -> 'Future':
    dispatcher = Pyro4.Proxy('PYRONAME:dispatcher')
    dispatcher.put_work(Workitem(client_id, scan_path,
                                 fmt=fmt, timeout=timeout,
                                 config=config))
    get_result = Pyro4.Future(dispatcher.get_result)
    return get_result(client_id)


def load_pkl(path):
    """ Load pickled object."""
    with open(path, 'r+b') as f:
        data = dill.load(f)
    return data
