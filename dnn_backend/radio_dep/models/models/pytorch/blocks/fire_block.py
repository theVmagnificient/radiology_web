""" Contains implementation of fire block of SqueezeNet model. """

import numpy as np
import pandas as pd
import torch
from ..layers import ConvBlock, Identity
from ..bases import Module, MetaModule
from ..utils import transform_to_int_tuple


class FireModule(Module, metaclass=MetaModule):

    def __init__(self, input_shape, filters, expand_kernel=3,
                 layout='cna', how='+', **kwargs):
        super().__init__(input_shape)
        block = ConvBlock if not kwargs.get('block') else kwargs.get('block')

        self.suqeeze_layer = block(
            input_shape=input_shape, layout=layout,
            c=dict(kernel_size=1, filters=filters),
        )
        self.fire_x1 = block(
            input_shape=self.squeeze_layer.output_shape, layout=layout,
            c=dict(kernel_size=expand_kernel, filters=filters * 4),
        )
        self.fire_x3 = block(
            input_shape=self.squeeze_layer.output_shape, layout=layout,
            c=dict(kernel_siz=1, filters=filters * 4)
        )

        self.how = how

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        """ Forward pass method for FireModule of SqueezeNet model. """
        x = self.squeeze_layer(inputs)
        return self.merge(self.fire_x1(x), self.fire_x3(x), how=self.how)
