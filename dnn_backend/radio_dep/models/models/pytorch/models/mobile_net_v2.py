""" Contains implementation of parametrized MobileNetV2 model. """

import torch
from ..layers import ConvBlock
from ..blocks import VanillaResBlock
from ..utils import INT_TYPES, FLOAT_TYPES
from ..bases import Sequential
from .base_model import BaseModel


class MobileNetV2(BaseModel):

    @classmethod
    def default_config(cls):
        config = BaseModel.default_config()

        config['conv_block'] = {
            'a': dict(activation='relu6')
        }

        config['input_block'] = {
            'layout': 'cna',
            'c': dict(kernel_size=3, stride=2, filters=32),
            'a': dict(activation='relu')
        }

        config['body_block'] = {
            'filters': (16, 24, 32, 64, 96, 160, 320),
            'num_blocks': (1, 2, 3, 4, 3, 3, 1),
            'factor': (1, 6, 6, 6, 6, 6, 6),
            'downsample': (False, True, True, True, False, True, False)
        }

        config['head_block'] = {
            'layout': 'cna > fa',
            'c': dict(kernel_size=1, filters=1028),
            '>': dict(mode='avg'),
            'f': dict(out_features=10),
            'a': dict(activation=('relu', 'linear'))
        }
        return config

    @classmethod
    def block(cls, input_shape, filters, factor=4,
              downsample=False, layout='cna cna cn', block=None, **kwargs):

        if not isinstance(filters, INT_TYPES):
            raise TypeError("Argument 'filters' must be of 'int' type.")

        if not isinstance(factor, INT_TYPES + FLOAT_TYPES):
            raise TypeError("Argument 'factor' must be of 'int' or 'float' type.")

        expand_filters = int(round(input_shape[0] * factor))
        if downsample:
            conv_block = ConvBlock if block is None else block
            return conv_block(
                input_shape=input_shape, layout=layout,
                c=dict(
                    filters=(expand_filters, expand_filters, filters),
                    kernel_size=(1, 3, 1),
                    stride=(1, 2, 1),
                    groups=(1, expand_filters, 1)
                )
            )
        else:
            return VanillaResBlock(
                input_shape=input_shape,
                filters=(expand_filters, expand_filters, filters),
                layout=layout,
                kernel_size=(1, 3, 1),
                groups=(1, expand_filters, 1),
                block=block, how='+',
                post_activation=True
            )

    @classmethod
    def body_block(cls, input_shape, block, config):
        filters = config.get('filters')
        downsample = config.get('downsample')
        factor = config.get('factor')
        num_blocks = config.get('num_blocks')
        shape = input_shape
        body = Sequential()
        for i, ifilters in enumerate(filters):
            for j, repeats in enumerate(range(num_blocks[i])):
                iconfig = {
                    'input_shape': shape,
                    'filters': ifilters,
                    'downsample': downsample[i] and (j == 0),
                    'factor': factor[i],
                    'block': block
                }
                x = cls.block(**iconfig)

                body.add_module("Block-{}-{}".format(i, j), x)
                shape = x.output_shape
        return body
