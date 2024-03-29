""" Contains implementation of parametrized ShuffleNet model. """

import torch

from ..blocks import CSplitAndShuffleUnit
from ..layers import ConvBlock
from ..bases import Sequential
from .base_model import BaseModel


class ShuffleNetV2(BaseModel):

    @classmethod
    def default_config(cls):
        config = BaseModel.default_config()

        config['input_block'] = {
            'layout': 'cnap',
            'c': dict(kernel_size=3, stride=2, filters=24),
            'p': dict(kernel_size=3, stride=2, mode='max')
        }

        config['body_block'] = {
            'filters': (116, 232, 464),
            'num_blocks': (4, 8, 4)
        }

        config['head_block'] = {
            'layout': 'cna > fa',
            'c': dict(kernel_size=1, filters=1024),
            '>': dict(mode='avg'),
            'f': dict(out_features=10),
            'a': dict(activation=('relu', 'linear'))
        }
        return config

    @classmethod
    def body_block(cls, input_shape, block, config):
        filters = config.get('filters')
        num_blocks = config.get('num_blocks')
        shape = input_shape
        body = Sequential()
        for i, ifilters in enumerate(filters):
            for j, repeats in enumerate(range(num_blocks[i])):
                iconfig = {
                    'downsample': j == 0,
                    'filters': ifilters,
                    'block': block
                }
                x = CSplitAndShuffleUnit(shape, **iconfig)

                body.add_module("Block-{}-{}".format(i, j), x)
                shape = x.output_shape
        return body
