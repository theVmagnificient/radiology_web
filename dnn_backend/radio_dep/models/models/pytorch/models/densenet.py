""" Contains implementation of DenseNet architecture. """

from functools import reduce
import operator
import numpy as np
import torch

from ..blocks import DenseBlock
from ..layers import ConvBlock
from ..bases import Sequential
from .base_model import BaseModel


class DenseNet(BaseModel):

    @classmethod
    def default_config(cls):
        """ Get default config for DenseNet model. """
        config = BaseModel.default_config()

        config['input_block'] = {
            'layout': 'cnap',
            'c': dict(kernel_size=7, stride=2, filters=32),
            'p': dict(kernel_size=3, stride=2)
        }

        config['body_block/dense'] = {
            'num_layers': (6, 12, 24, 16),
            'layout': 'nac',
            'kernel_size': 3,
            'use_bottleneck': True,
            'bottleneck_factor': 4,
            'growth_rate': 32
        }

        config['body_block/transition'] = {
            'layout': 'cnap',
            'c': dict(kernel_size=1, stride=1),
            'p': dict(kernel_size=2, stride=2, mode='avg')
        }

        config['head_block'] = {
            'layout': '> fa',
            'f': dict(out_features=10),
            'a': dict(activation='linear')
        }
        return config

    @classmethod
    def body_block(cls, input_shape, block, config):
        """ Body block of densenet model. """
        transition_config = config.get('transition')
        dense_block = DenseBlock
        dense_config = config.get('dense')
        use_bottleneck = dense_config.get('use_bottleneck')
        bottleneck_factor = dense_config.get('bottleneck_factor')
        growth_rate = dense_config.get('growth_rate')
        num_layers = dense_config.get('num_layers')
        kernel_size = dense_config.get('kernel_size')

        shape = input_shape
        body = Sequential()
        for i, inum_layers in enumerate(num_layers):
            x = dense_block(
                input_shape=shape, block=block,
                use_bottleneck=use_bottleneck,
                bottleneck_factor=bottleneck_factor,
                growth_rate=growth_rate, num_layers=inum_layers,
                kernel_size=kernel_size
            )
            body.add_module("Block-{}".format(i), x)

            transition_layer = block.partial(
                input_shape=x.output_shape,
                **transition_config
            )
            transition_layer = transition_layer(c=dict(filters=growth_rate))
            body.add_module("Transition-{}".format(i), transition_layer)
            shape = transition_layer.output_shape

        return body


class DenseNet121(DenseNet):

    @classmethod
    def default_config(cls):
        config = DenseNet.default_config()
        body_config = {
            'num_layers': (6, 12, 24, 16),
        }
        return config + {'body_block': {'dense': body_config}}


class DenseNet169(DenseNet):

    @classmethod
    def default_config(cls):
        config = DenseNet.default_config()
        body_config = {
            'num_layers': (6, 12, 32, 32),
        }
        return config + {'body_block': {'dense': body_config}}


class DenseNet201(DenseNet):

    @classmethod
    def default_config(cls):
        config = DenseNet.default_config()
        body_config = {
            'num_layers': (6, 12, 48, 32),
        }
        return config + {'body_block': {'dense': body_config}}


class DenseNet161(DenseNet):

    @classmethod
    def default_config(cls):
        config = DenseNet.default_config()
        body_config = {
            'num_layers': (6, 12, 36, 24),
            'growth_rate': 48
        }
        return config + {'body_block': {'dense': body_config}}
