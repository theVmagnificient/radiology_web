import math
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F

from ..bases import Sequential
from ..layers import ConvBlock

from .base_model import BaseModel
from ..blocks.gcn import GCNBlock
from ..blocks.nonlocal_block import NonLocalBlock
from ..blocks import BaseDecoder, VanillaUNetDecoder, VanillaVNetDecoder
from ..blocks import BaseEncoder, VanillaUNetEncoder, VanillaVNetEncoder



class EncoderDecoder(BaseModel):

    @classmethod
    def default_config(cls):
        config = BaseModel.default_config()
        config['body_block'] = {
            'encoder': {
                'kernel_size': 3,
                'levels/filters': (64, 128, 256, 512, 1024),
                'levels/layout': ('cna', 'cna cna',
                                  'cna cna cna',
                                  'cna cna cna',
                                  'cna cna cna')
            },
            'decoder': {
                'kernel_size': 3,
                'levels/filters': (512, 256, 128, 64),
                'levels/layout': ('cna cna cna',
                                  'cna cna cna',
                                  'cna cna', 'cna')
            }
        }
        return config

    @staticmethod
    def get_level(config, level):
        return {key: value[level] for key, value in config.items()}

    @staticmethod
    def get_levels_num(config):
        levels_num = len(config[list(config.keys())[0]])
        if not np.all(np.array([len(v) for k, v in config.items()]) == levels_num):
            raise ValueError("Values in 'levels' part of config "
                             + "must have the same lenght")
        return levels_num

    @classmethod
    def encoder(cls, input_shape, **kwargs):
        print('Calling encoder with params {}'.format(kwargs))
        x = torch.nn.Module()
        x.output_shape = input_shape
        x.input_shape = input_shape
        return x

    @classmethod
    def decoder(cls, input_shape, skip_shape, **kwargs):
        print('Calling decoder with params {}'.format(kwargs))
        x = torch.nn.Module()
        x.output_shape = input_shape
        x.input_shape = input_shape
        return x

    @classmethod
    def body_block(cls, input_shape, block, config):
        encoder_config = {'block': block}
        for key, value in config.get('encoder').items():
            if key != 'levels':
                encoder_config[key] = value

        decoder_config = {'block': block}
        for key, value in config.get('decoder').items():
            if key != 'levels':
                decoder_config[key] = value

        encoders = Sequential()
        decoders = Sequential()

        encoders_output_shapes = []
        shape = input_shape
        for i in range(cls.get_levels_num(config.get('encoder').get('levels'))):
            iconfig = {**encoder_config,
                       **cls.get_level(config.get('encoder').get('levels'), i),
                       'input_shape': shape}

            iencoder = cls.encoder(**iconfig)
            encoders.add_module('Encoder-{}'.format(i), iencoder)
            shape = iencoder.output_shape
            encoders_output_shapes.append(shape)

        for i in range(cls.get_levels_num(config.get('decoder').get('levels'))):
            iconfig = {**decoder_config,
                       **cls.get_level(config.get('decoder').get('levels'), i),
                       'input_shape': shape, 'skip_shape': encoders_output_shapes[-i-2]}
            idecoder = cls.decoder(**iconfig)
            shape = idecoder.output_shape
            decoders.add_module('Decoder-{}'.format(i), idecoder)

        return encoders, decoders


    def build(self, *args, **kwargs):
        config = self.build_config()
        layers = OrderedDict()
        if config.get('conv_block'):
            conv_block = ConvBlock.partial(**config.get('conv_block').config)
        else:
            conv_block = ConvBlock

        input_shape = config.get('input_shape')
        self.input_module = self.input_block(
            input_shape=input_shape, block=conv_block,
            config=config.get('input_block')
        )

        self.encoders, self.decoders = self.body_block(
            input_shape=(self.input_module.output_shape
                         if self.input_module else input_shape),
            block=conv_block, config=config.get('body_block')
        )

        self.head_module = self.head_block(
            input_shape=self.decoders.output_shape,
            block=conv_block, config=config.get('head_block')
        )

        self.output_shape = input_shape

    def forward_encoders(self, inputs):
        x = inputs
        encoder_outputs = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            encoder_outputs.append(x)
        return encoder_outputs

    def forward_decoders(self, encoder_outputs):
        x = encoder_outputs[-1]
        decoder_outputs = []
        for i, decoder in enumerate(self.decoders):
            x = decoder([x, encoder_outputs[-i-2]])
            decoder_outputs.append(x)
        return decoder_outputs

    def forward(self, inputs):
        x = self.input_module(x) if self.input_module else inputs
        x = self.forward_decoders(self.forward_encoders(x))[-1]
        x = self.head_module(x) if self.head_module else x
        return x
