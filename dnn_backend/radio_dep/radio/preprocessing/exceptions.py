""" Contains errors and exceptions classes associated with RadIO. """

from functools import wraps
from toolz import curry
import numpy as np


INT_TYPES = (int, np.int, np.int8, np.uint8, np.int16,
             np.uint16, np.int32, np.uint32, np.int64, np.uint64)


FLOAT_TYPES = (float, np.float, np.float16, np.float32, np.float64)



class RadIOBatchException(Exception):
    """ Base class for Exceptions in RadIO framework. """
    pass


class RadIOComponentNotFound(RadIOBatchException):

    def __init__(self, component, batch):
        super().__init__("Component '{}' not found".format(component)
                         + " in batch of type {}".format(type(batch)))


class RadIOBatchComponentNotLoaded(RadIOBatchException):

    def __init__(self, component, batch):
        super().__init__("Component '{}' must be loaded".format(component)
                         + " before calling this method.")


def assert_component_is_loaded(component):

    def _decorator(method):

        @wraps(method)
        def _wrapped_method(self, *args, **kwargs):
            if not hasattr(self, component):
                raise RadIOComponentNotFound(component, self)
            elif getattr(self, component) is None:
                raise RadIOBatchComponentNotLoaded(component, self)

            return method(self, *args, **kwargs)

        return _wrapped_method

    return _decorator
