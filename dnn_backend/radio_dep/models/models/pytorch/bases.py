""" Contains implementation of MetaModule, Module, Sequential and Layer classes."""

import sys
import math
from functools import reduce
from operator import mul
from collections import defaultdict
import numpy as np
import torch

from .utils import transform_to_int_tuple, INT_TYPES, LIST_TYPES
from .utils import crop_as, pad_as


class MetaModule(type):

    """ Metaclass for partial constructor calling. """

    _counter = defaultdict(int)

    def partial(cls, *args, **kwargs):
        """ Build partialy applied module. """

        if hasattr(cls, '_is_partial_module'):
            _kwargs = {**cls._partial_kwargs, **kwargs}
            _args = cls._partial_args if len(args) == 0 else args
            cls = cls._base_cls
        else:
            _kwargs = kwargs
            _args = args

        cls._counter[cls.__name__] += 1
        class PartialModule(cls):
            _is_partial_module = True
            _partial_args = _args
            _partial_kwargs = _kwargs
            _options = cls._options
            _options_params = cls._options_params
            _base_cls = cls
        PartialModule.__name__ = cls.__name__ + '_' + str(cls._counter[cls.__name__])
        return PartialModule

    def __new__(mtcls, name, bases, attrs):
        attrs = {'_options': {}, '_options_params': {}, **attrs}
        cls = super(MetaModule, mtcls).__new__(mtcls, name, bases, attrs)
        return cls

    def __call__(cls, *args, **kwargs):
        if hasattr(cls, '_is_partial_module'):
            args = cls._partial_args

            prev_keys = set(cls._partial_kwargs.keys())
            keys = set(kwargs.keys())

            new_kwargs = {}
            for key in keys & prev_keys:
                if (isinstance(cls._partial_kwargs[key], dict)
                    and isinstance(kwargs[key], dict)):

                    new_kwargs[key] = {**cls._partial_kwargs[key],
                                       **kwargs[key]}
                else:
                    new_kwargs[key] = kwargs[key]

            kwargs = {**cls._partial_kwargs, **kwargs, **new_kwargs}
            return cls._base_cls(*args, **kwargs)

        return super().__call__(*args, **kwargs)


class Module(torch.nn.Module):

    def __init__(self, input_shape):
        super().__init__()
        self._input_shape = transform_to_int_tuple(input_shape,
                                                   'input_shape',
                                                   len(input_shape))
    @property
    def ndims(self):
        return len(self._input_shape)

    @property
    def input_shape(self):
        """ Get shape of the input tensor.

        Returns
        -------
        ndarray(int)
            shape of the input tensor.
        """
        return np.array(self._input_shape, dtype=np.int)

    @property
    def output_shape(self):
        """ Get shape of the output tensor.

        Returns
        -------
        ndarray(int)
            shape of the output tensor.
        """
        return self.input_shape

    @property
    def in_channels(self):
        """ Get number of channels in the input tensor of operation.

        Returns
        -------
        int
            size of channels dimension of the input tensor.
        """
        return int(self.input_shape[0])

    @property
    def out_channels(self):
        """ Get number of channels in the output tensor of operation.

        Returns
        -------
        int
            size of channels dimension in the output tensor.
        """
        return int(self.output_shape[0])

    def to_int_array(self, parameter, name, length):
        """ Transform input parameter value to int list of given length.

        Parameters
        ----------
        parameter : int, tuple(int), list(int) or ndarray(int)
            input parameter value.
        name : str
            name of parameter. Required by exception raising part of function.
        length : int
            length of output list with parameter values.

        Returns
        -------
        ndarray(int)

        Raises
        ------
        ValueError
            If input parameter has wrong type or has improper length(if list-like).
        """
        if isinstance(parameter, INT_TYPES):
            parameter = np.asarray([parameter] * length, dtype=np.int)
        elif isinstance(parameter, LIST_TYPES):
            parameter = np.asarray(parameter, dtype=np.int).flatten()
            if len(parameter) != length:
                raise ValueError("Argument {} has inproper lenght.".format(name)
                                 + " Must have {}, got {}.".format(length,
                                                                   len(parameter)))
        else:
            raise ValueError("Argument {} must be int or ".format(name)
                             + "tuple, list, ndarray "
                             + "containing {} int values.".format(length))
        return parameter

    @classmethod
    def merge(cls, x, y, how='+'):
        """ Merge tensors according given rule.

        Parameters
        ----------
        x : Tensor
            first tensor.
        y : Tensor
            second tensor.
        how : str
            how to merge input tensors. Can be on of following values:
            '+' for sum, '*' for product or '.' for concatenation along first
            dimension. Default is '+'.

        Returns
        -------
        Tensor
            result of merging operation.

        Raises
        ------
        ValueError
            if argument 'how' has value diverging from '+', '*' or '.'.
        """
        if how not in ('+', '*', '.'):
            raise ValueError("Argument 'how' must be one of "
                             + "following values: ('+', '.', '*'). "
                             + "Got {}.".format(how))
        if how == '.':
            return torch.cat([x, y], dim=1)
        elif how == '+':
            return x + y
        elif how == '*':
            return x * y

    @classmethod
    def crop_as(cls, x: 'Tensor', y: 'Tensor') -> 'Tensor':
        """ Crop first tensor to have the same shape as the second.

        This method affects only spatial dimensions of tensor, so
        batch size and channels dimension remain unchanged.

        Parameters
        ----------
        x : Tensor
            tensor to crop.
        y : Tensor
            tensor whose shape will be used for cropping.

        Returns
        -------
        Tensor
        """
        return crop_as(x, y)

    @classmethod
    def pad_as(cls, x: 'Tensor', y: 'Tensor') -> 'Tensor':
        """ Add padding to first tensor to have the same shape as the second.

        This method affects only spatial dimensions of tensor, so
        batch size and channels dimension remain unchanged.


        Parameters
        ----------
        x : Tensor
            tensor to pad.
        y : Tensor
            tensor whose shape will be used for padding size computation.

        Returns
        -------
        Tensor
        """
        return pas_as(x, y)


class Layer(Module):

    def forward(self, inputs):
        return self.layer.forward(inputs)

    def __repr__(self):
        return self.layer.__repr__()


class Sequential(torch.nn.Sequential):
    """ Base class for Sequential models. """

    @property
    def input_shape(self) -> 'Tensor':
        """ Get shape of the input tensor. """
        first, *_ = list(self.children())
        return first.input_shape

    @property
    def output_shape(self) -> 'Tensor':
        """ Get shape of the output tensor. """
        *_, last = list(self.children())
        return last.output_shape

    @property
    def ndims(self):
        """ Number of dimensions in the input tensor. """
        return len(self.input_shape)

    @property
    def in_channels(self):
        """ Get number of channels in the input tensor of operation.

        Returns
        -------
        int
            size of channels dimension of the input tensor.
        """
        return int(self.input_shape[0])

    @property
    def out_channels(self):
        """ Get number of channels in the output tensor of operation.

        Returns
        -------
        int
            size of channels dimension in the output tensor.
        """
        return int(self.output_shape[0])

    def to_int_array(self, parameter, name, length):
        """ Transform input parameter value to int list of given length.

        Parameters
        ----------
        parameter : int, tuple(int), list(int) or ndarray(int)
            input parameter value.
        name : str
            name of parameter. Required by exception raising part of function.
        length : int
            length of output list with parameter values.

        Returns
        -------
        ndarray(int)

        Raises
        ------
        ValueError
            If input parameter has wrong type or has improper length(if list-like).
        """
        if isinstance(parameter, INT_TYPES):
            parameter = np.asarray([parameter] * length, dtype=np.int)
        elif isinstance(parameter, LIST_TYPES):
            parameter = np.asarray(parameter, dtype=np.int).flatten()
            if len(parameter) != length:
                raise ValueError("Argument {} has inproper lenght.".format(name)
                                 + " Must have {}, got {}.".format(length,
                                                                   len(parameter)))
        else:
            raise ValueError("Argument {} must be int or ".format(name)
                             + "tuple, list, ndarray "
                             + "containing {} int values.".format(length))
        return parameter

    @classmethod
    def merge(cls, x, y, how='+'):
        """ Merge tensors according given rule.

        Parameters
        ----------
        x : Tensor
            first tensor.
        y : Tensor
            second tensor.
        how : str
            how to merge input tensors. Can be on of following values:
            '+' for sum, '*' for product or '.' for concatenation along first
            dimension. Default is '+'.

        Returns
        -------
        Tensor
            result of merging operation.

        Raises
        ------
        ValueError
            if argument 'how' has value diverging from '+', '*' or '.'.
        """
        if how not in ('+', '*', '.'):
            raise ValueError("Argument 'how' must be one of "
                             + "following values: ('+', '.', '*'). "
                             + "Got {}.".format(how))
        if how == '.':
            return torch.cat([x, y], dim=1)
        elif how == '+':
            return x + y
        elif how == '*':
            return x * y

    @classmethod
    def crop_as(cls, x: 'Tensor', y: 'Tensor') -> 'Tensor':
        """ Crop first tensor to have the same shape as the second.

        This method affects only spatial dimensions of tensor, so
        batch size and channels dimension remain unchanged.

        Parameters
        ----------
        x : Tensor
            tensor to crop.
        y : Tensor
            tensor whose shape will be used for cropping.

        Returns
        -------
        Tensor
        """
        return crop_as(x, y)

    @classmethod
    def pad_as(cls, x: 'Tensor', y: 'Tensor') -> 'Tensor':
        """ Add padding to first tensor to have the same shape as the second.

        This method affects only spatial dimensions of tensor, so
        batch size and channels dimension remain unchanged.


        Parameters
        ----------
        x : Tensor
            tensor to pad.
        y : Tensor
            tensor whose shape will be used for padding size computation.

        Returns
        -------
        Tensor
        """
        return pas_as(x, y)


class ConvModule(Module):
    """ Base class for all Convolutional modules.

    Following layers are considered convolutional:
    Conv1d, Conv2d, Conv3d, ConvTranspose1d,
    ConvTranspose2d, ConvTranspose3d,
    MaxPool1d, MaxPool2d, MaxPool3d, MaxUnpool.
    """
    _repr_attributes = []

    @property
    def output_shape(self):
        """ Get shape of the output tensor.

        Returns
        -------
        tuple(int)
            shape of the output tensor.
        """
        return self._output_shape

    def __init__(self, input_shape, kernel_size=3, stride=1, dilation=1):
        super().__init__(input_shape)
        if self.ndims not in (2, 3, 4):
            raise ValueError("Input tensor must be 2, 3 or 4 dimensional "
                             + " with zero axis meaning number of channels.")
        self.kernel_size = transform_to_int_tuple(kernel_size,
                                                  'kernel_size',
                                                  self.ndims - 1)
        self.stride = transform_to_int_tuple(stride,
                                             'stride',
                                             self.ndims - 1)
        self.dilation = transform_to_int_tuple(dilation,
                                               'dilation',
                                               self.ndims - 1)
