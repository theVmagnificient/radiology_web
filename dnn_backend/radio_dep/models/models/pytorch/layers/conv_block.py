""" Contains ConvBlock, Branches, NoOperation and Merge classes. """

import inspect
import copy
from functools import reduce
import operator
from collections import Counter, OrderedDict
import re
import numpy as np
import torch

from ..bases import MetaModule, Module, Sequential, transform_to_int_tuple
from ..utils import transform_to_int_tuple, INT_TYPES, FLOAT_TYPES, LIST_TYPES
from ..utils import unpack_dict_recursive, addindent


class ConvBlock(Sequential, metaclass=MetaModule):

    @classmethod
    def get_options(cls) -> dict:
        """ Get registered options.

        Returns
        -------
        dict
            dictionary with options: keys are shortcuts, values are corresponding
            modules' classes.
        """
        return cls._options.copy()

    @classmethod
    def _unify_parameter(cls, values, vectorize, num_layers, ndims):
        """ Unify parameter values passed to ConvBlock. """

        if isinstance(values, LIST_TYPES):

            # This is a kind of a hook: if there is only one layer of this type
            # you won't need to write smth like
            # c=dict(kernel_size=[(1, 3)]) if layout='c'
            if num_layers == 1 and ((len(values) == ndims and vectorize) or not vectorize):
                if len(values) == 1:
                    return values
                else:
                    return [values]

            # Here is just a regular vectorized logics
            if len(values) != num_layers:
                raise ValueError("Length of param {}".format(len(values))
                                 + " must match number"
                                 + " of layers in layout"
                                 + " which is {}".format(num_layers))

            if all(isinstance(v, INT_TYPES + FLOAT_TYPES) for v in values) and vectorize:
                return [(v, ) * ndims for v in values]
            else:
                return values

        if isinstance(values, bool):
            values = [values] * num_layers
        elif isinstance(values, INT_TYPES):
            values = [int(values)] * num_layers
        elif isinstance(values, FLOAT_TYPES):
            values = [float(values)] * num_layers
        else:
            values = [values] * num_layers

        if vectorize:
            return [(value, ) * ndims for value in values]

        return values

    @classmethod
    def register_option(cls, name, vectorized_params=()):
        """ Decorator used to register options for ConvBlock.

        Parameters
        ----------
        name : str
            name of shortcut for registered option.
            For example, 'c' for Conv layer.
        vectorized_params : Tuple[str]
            names of arguments for option that will be
            expanded depending on number of dimensions.

        Returns
        -------
        Callable
            decorator for module class.
        """

        def decorator(module_cls):
            cls._options[name] = module_cls

            params_description = []

            if isinstance(module_cls, type) and type(module_cls) != type:
                params_info = inspect.getfullargspec(module_cls.__init__)
            else:
                params_info = inspect.getfullargspec(module_cls)
            args_names = params_info.args[2:]

            if params_info.defaults:
                args_defaults = dict(zip(args_names[::-1],
                                         params_info.defaults[::-1]))
            else:
                args_defaults = {}
            for i in range(len(args_names)):
                arg_name = args_names[i]
                param_dict = {'name': arg_name}
                if arg_name in vectorized_params:
                    param_dict['vectorize'] = True
                else:
                    param_dict['vectorize'] = False

                if arg_name in args_defaults:
                    param_dict['default'] = args_defaults[arg_name]

                params_description.append(param_dict)

            cls._options_params[name] = params_description
            return module_cls

        return decorator

    @classmethod
    def map_layout_to_options(cls, layout):
        return list(re.sub(r'\s+', '', layout))

    def __init__(self, input_shape, layout, **kwargs):
        """ Create convolutional block module.

        Covnolutional block is a subclass of pytorch sequential model.
        User can dynamically register custom pytorch modules
        as options of ConvBlock and use shortcuts corresponding
        to these modules in layout of ConvBlock using
        ConvBlock.register_option decorator. Calling get_options
        classmethod will return all registered options that can be used in
        ConvBlock layout.

        Many layers contain vectorized parameters like 'kernel_size',
        'stride' or 'dilation' that means that these parameters can have
        different values for different dimensions. Also each created block
        can contain several operations of given type(in our case
        two convolutions, two activations and two batch normalizations).
        Usually it is required to pass the same value for specific parameter
        for each dimension or/and for all operations of given type in block.
        ConvBlock gives an ability to avoid explicit copying of parameters values
        for similar operations in block. For example, c=dict(kernel_size=3)
        in ConvBlock will mean that all convolutions will
        have (3, 3) kernel size (in 2D case). If one wants to pass different
        kernel_size parameter for each convolution then it's possible to do that
        passing list of parameters c=dict(kernel_size=[3, 5]). Note that in this
        case length of list or tuple must be the same as number of
        corresponding operations in block. If it's required
        to have non-symmetric kernel_size for convolutions along xy-dims
        but shared by all operations in block then passing
        c=dict(kernel_size=(3, 5)) will solve the problem.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor. Note that
            batch dimension is not taken in account.
        layout : str
            compressed description of the block
            operations sequence using shortcuts for registered options.
            For instance, 'cna cna' will represent two convolution operations
            with batch normalization before activation.
        **kwargs
            each argument must be a dict with keys representing parameters of
            corresponding operation(For example, c=dict(kernel_size=3, filters=16))

        Note
        ----
        This module also have partial(...) method that allows to split
        parameters passing for __init__ or __new__
        constructor into several steps.


        Examples
        --------
        Creation of block of two convolutions with batch normalization before
        activation followed by max pooling operation will look like:

        >>> x = ConvBlock(
        ... input_shape=(3, 128, 128), layout='cna cna p',
        ... c=dict(kernel_size=3, filters=(16, 32)),
        ... p=dict(kernel_size=2, stride=2)
        ... a=dict(activation=)
        ... )

        # TODO: Add 'Raises' section in docstring
        """
        super().__init__()

        self.name = kwargs.get('name', self.__class__.__name__)
        self.layout = self.map_layout_to_options(layout)
        self.layers_counter = Counter(self.layout)

        input_shape = self.to_int_array(input_shape,
                                        'input_shape',
                                        len(input_shape))
        ndims = len(input_shape)
        layers_params = {layer_name: {} for layer_name in self.layers_counter}
        for layer_name, layer_counts in self.layers_counter.items():
            layer_kwargs = kwargs.get(layer_name, {})
            for param in self._options_params[layer_name]:
                if param['name'] in layer_kwargs:
                    raw_value = layer_kwargs[param['name']]
                elif 'default' in param:
                    raw_value = param['default']
                else:
                    raise ValueError("Argument {} ".format(param['name'])
                                     + "has no default value")

                values = self._unify_parameter(raw_value, param['vectorize'],
                                               layer_counts, ndims - 1)
                layers_params[layer_name][param['name']] = values

        self.layers_params = copy.deepcopy(layers_params)
        shape = transform_to_int_tuple(input_shape, 'input_shape', ndims)
        for i, layer in enumerate(self.layout):
            layer_class = self._options[layer]
            params_dict = {}
            for param in self._options_params[layer]:
                param_values = layers_params[layer][param['name']]
                params_dict[param['name']] = param_values[0]
                layers_params[layer][param['name']] = param_values[1:]

            module = layer_class(shape, **params_dict)
            self.add_module('Module_{}'.format(i), module)

            shape = transform_to_int_tuple(module.output_shape,
                                           'output_shape',
                                           len(module.output_shape))

    def __repr__(self) -> str:
        """ String representation of ConvBlock. """
        tmpstr = self.name + '(\n'
        for key, module in self._modules.items():
            modstr = module.__repr__()
            modstr = addindent(modstr, 2)
            tmpstr = tmpstr + '  (' + key + '): ' + modstr + '\n'
        tmpstr = tmpstr + ')'
        return tmpstr
