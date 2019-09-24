""" Contains implementation of parametrized ResNet model. """

import math
from functools import reduce
import operator
import numpy as np
import torch

from ..blocks import BottleneckResBlock, SimpleResBlock
from ..layers import ConvBlock
from ..bases import Sequential
from .base_model import BaseModel


class ResNet(BaseModel):

    @classmethod
    def default_config(cls):
        config = BaseModel.default_config()

        config['input_block'] = {
            'layout': 'cnap',
            'c': dict(kernel_size=7, stride=2, filters=32),
            'p': dict(kernel_size=3, stride=2)
        }

        config['body_block'] = {
            'filters': (64, 128, 256, 512),
            'num_blocks': (3, 4, 6, 3),
            'use_bottleneck': False,
            'factor': 4,
            'how': '+'
        }

        config['head_block'] = {
            'layout': '> fa',
            '>': dict(mode='avg'),
            'f': dict(out_features=10),
            'a': dict(activation='linear')
        }
        return config

    @classmethod
    def body_block(cls, input_shape, block, config):
        use_bottleneck = config.get('use_bottleneck')
        factor, how = config.get('factor'), config.get('how')
        filters, num_blocks = config.get('filters'), config.get('num_blocks')
        post_activation = config.get('post_activation', default=True)
        shape = input_shape
        body = Sequential()
        for i, ifilters in enumerate(filters):
            for j, repeats in enumerate(range(num_blocks[i])):
                if use_bottleneck:
                    x = BottleneckResBlock(shape, ifilters,
                                           downsample=(j == 0 and i > 0),
                                           factor=factor,
                                           post_activation=post_activation,
                                           how=how, block=block)
                else:
                    x = SimpleResBlock(shape, ifilters,
                                       downsample=(j == 0 and i > 0),
                                       post_activation=post_activation,
                                       how=how, block=block)
                body.add_module("Block-{}-{}".format(i, j), x)
                shape = x.output_shape
        return body


class ResNet18(ResNet):

    @classmethod
    def default_config(cls):
        config = ResNet.default_config()
        body_config = {
            'num_blocks': (2, 2, 2, 2),
            'use_bottleneck': False,
        }
        return config + {'body_block': body_config}


class ResNet34(ResNet):

    @classmethod
    def default_config(cls):
        config = ResNet.default_config()
        body_config = {
            'num_blocks': (3, 4, 6, 3),
            'use_bottleneck': False,
        }
        return config + {'body_block': body_config}


class ResNet50(ResNet):

    @classmethod
    def default_config(cls):
        config = ResNet.default_config()
        body_config = {
            'num_blocks': (3, 4, 6, 3),
            'use_bottleneck': True,
        }
        return config + {'body_block': body_config}


class ResNet101(ResNet):

    @classmethod
    def default_config(cls):
        config = ResNet.default_config()
        body_config = {
            'num_blocks': (3, 4, 23, 3),
            'use_bottleneck': True,
        }
        return config + {'body_block': body_config}


class ResNet152(ResNet):

    @classmethod
    def default_config(cls):
        config = ResNet.default_config()
        body_config = {
            'num_blocks': (3, 8, 36, 3),
            'use_bottleneck': True,
        }
        return config + {'body_block': body_config}
