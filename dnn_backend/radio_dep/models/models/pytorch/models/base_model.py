from collections import OrderedDict
import numpy as np
import torch
from torch.autograd import Variable
from .config import Config
from ..layers import ConvBlock


class BaseModel(torch.nn.Module):
    """ Base class for all models

    Attributes
    ----------
    name : str
        a model name
    config : dict
        configuration parameters

    Notes
    -----

    **Configuration**:

    * build : bool
        whether to build a model by calling `self.build()`. Default is True.
    * load : dict
        parameters for model loading. If present, a model will be loaded
        by calling `self.load(**config['load'])`.

    """
    def __init__(self, config=None, *args, **kwargs):
        super().__init__()
        self.data_format = 'channels_first'
        self.config = Config(config) or Config()
        load = self.config.get('load', default=False)
        if load:
            self.load(**load)
        if self.config.get('build', default=True):
            self._model = self.build(*args, **kwargs)

        self.input_shape = self.config.get('input_shape')

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def as_torch(cls, data, dtype='float32', device=None, grad=False, async=False):
        if isinstance(data, np.ndarray):
            x = torch.from_numpy(data)
        elif torch.is_tensor(data):
            x = data
        elif isinstance(data, Variable):
            x = data.data

        if dtype == 'float16':
            x = x.type(torch.HalfTensor)
        elif dtype == 'float32':
            x = x.type(torch.FloatTensor)
        elif dtype == 'float64' or dtype == 'float':
            x = x.type(torch.DoubleTensor)

        elif dtype == 'int8':
            x = x.type(torch.ByteTensor)
        elif dtype == 'int16':
            x = x.type(torch.ShortTensor)
        elif dtype == 'int32':
            x = x.type(torch.IntTensor)
        elif dtype == 'int64' or dtype == 'int':
            x = x.type(torch.LongTensor)
        else:
            raise ValueError("Argument 'dtype' must be str.")

        if device is None:
            return Variable(x, requires_grad=grad)
        elif isinstance(device, int):
            return Variable(x.cuda(device=device, async=async), requires_grad=grad)

    @classmethod
    def as_numpy(cls, data, dtype='float32'):
        if isinstance(data, np.ndarray):
            return data.astype(dtype)
        elif torch.is_tensor(data):
            x = data
        elif isinstance(data, Variable):
            x = data.data
        return x.cpu().numpy()

    @property
    def default_name(self):
        """: str - the class name (serve as a default for a model name) """
        return self.__class__.__name__

    @classmethod
    def pop(cls, variables, config, **kwargs):
        """ Return variables and remove them from config"""
        return Config().pop(variables, config, **kwargs)

    @classmethod
    def get(cls, variables, config, default=None):
        """ Return variables from config """
        return Config().get(variables, config, default=default)

    @classmethod
    def put(cls, variable, value, config):
        """ Put a new variable into config """
        return Config().put(variable, value, config)

    @classmethod
    def default_config(cls):
        """ Define model defaults. """
        config = {}
        config['conv_block'] = {}
        config['input_block'] = {}
        config['body_block'] = {}
        config['head_block'] = {}
        return Config(config)

    def build_config(self, names=None):
        """ Define a model architecture configuration. """
        return self.default_config() + self.config

    @classmethod
    def input_block(cls, input_shape, block, config):
        if config:
            return block(input_shape=input_shape, **config)

    @classmethod
    def head_block(cls, input_shape, block, config):
        if config:
            return block(input_shape=input_shape, **config)

    @classmethod
    def body_block(cls, input_shape, **kwargs):
        raise NotImplementedError("This method must be "
                                  + "implemented in ancestor class")

    def build(self, *args, **kwargs):
        config = self.build_config()
        layers = OrderedDict()

        if config.get('conv_block'):
            conv_block = ConvBlock.partial(**config.get('conv_block').config)
        else:
            conv_block = ConvBlock

        input_shape = config.get('input_shape')
        x = self.input_block(input_shape=input_shape,
                             block=conv_block,
                             config=config.get('input_block'))

        if x is not None:
            layers['InputBlock'] = x
            input_shape = x.output_shape

        x = self.body_block(input_shape=input_shape,
                            block=conv_block,
                            config=config.get('body_block'))
        if x is not None:
            layers['Body'] = x
            input_shape = x.output_shape

        x = self.head_block(input_shape=input_shape,
                            block=conv_block,
                            config=config.get('head_block'))
        if x is not None:
            layers['Head'] = x
            input_shape = x.output_shape

        self.output_shape = input_shape
        return torch.nn.Sequential(layers)

    @classmethod
    def load(cls, path, **kwargs):
        """ Load the model """
        return torch.load(path)

    def save(self, path, **kwargs):
        """ Save the model """
        torch.save(self, path)

    def forward(self, inputs):
        return self._model.forward(inputs)
