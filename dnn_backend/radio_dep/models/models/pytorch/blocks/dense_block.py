""" Contains implementation of Dense Block. """

import math
import numpy as np
import torch
import torch.nn.functional as F

from ..bases import Module
from ..layers import ConvBlock


class DenseBlock(Module):
    """ Dense Block implementation used in DenseNet architecture. """

    def __init__(self, input_shape, layout='nac',
                 use_bottleneck=False, kernel_size=3,
                 bottleneck_factor=4, growth_rate=32,
                 num_layers=12, block=None):
        """ Parametrized dense block for DenseNet architecture.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor.
        use_bottleneck : bool
            whether to use bottleneck or not. Default is False.
        kernel_size : int, Tuple[int], List[int] or NDArray[int]
            kernel size of convolution opertion. Default is 3.
        bottleneck_factor : int
        growth_rate : int
            number of output_channels in convolutional operation.
            Default is 32.
        num_layers : int
            number of convolutions apllied one by one. Bottleneck 1x...x1
            convolutions are not taken into account.
        block : ConvBlock or None
            basic block class that will be used
            for internal operations description.
            Can be partially applied ConvBlock or None. If None than ConvBlock
            module will be used. Default is None.
        """
        super().__init__(input_shape)
        block = ConvBlock if block is None else block
        self.module_list = torch.nn.ModuleList()
        shape = input_shape
        for i in range(num_layers):
            if use_bottleneck:
                x = block(
                    input_shape=shape, layout=layout * 2,
                    c=dict(filters=(growth_rate * bottleneck_factor, growth_rate),
                           kernel_size=(1, kernel_size))
                )
            else:
                x = block(
                    input_shape=shape, layout=layout,
                    c=dict(kernel_size=kernel_size, filters=growth_rate)
                )
            self.module_list.append(x)
            in_channels = shape[0]
            shape = np.array(x.output_shape)
            shape[0] += in_channels
        self._output_shape = shape
        self._output_shape[0] = growth_rate

    @property
    def output_shape(self) -> 'NDArray[int]':
        """ Get shape of the output tensor. """
        return np.asarray(self._output_shape, dtype=np.int)

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        """ Forward pass method for dense block module. """
        x = inputs
        for i, module in enumerate(self.module_list):
            if i == (len(self.module_list) - 1):
                return module(x)
            x = torch.cat([x, module(x)], dim=1)
        return x
