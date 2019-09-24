""" Contains implementation of metaclass for debugging. """

from functools import wraps
import logging
from toolz import curry
from .utils import get_config
from .utils import DEFAULT_CONFIG as CONFIG


@curry
def debug_method(method):
    @wraps(method)
    def decorated(self, *args, **kwargs):
        logger = logging.getLogger('RadIO.preprocessing')

        logger.debug(
            "\tPipeline: {}".format(self.pipeline) + '\n' +
            "\tBatch: {}".format(self) + '\n' +
            "\tMethod: {method}(args={args}, kwargs={kwargs})".format(method=method.__name__,
                                                                      args=args, kwargs=kwargs) + "\n")
        return method(self, *args, **kwargs)
    return decorated


class DebugActions(type):

    def __new__(mtls, name, bases, attrs):  # noqa: N804
        new_attrs = {}
        for name, attr in attrs.items():
            if not name.startswith('_') and callable(attr):
                new_attrs[name] = debug_method(attr)
            else:
                new_attrs[name] = attr
        return super(DebugActions, mtls).__new__(mtls, name, bases, new_attrs)
