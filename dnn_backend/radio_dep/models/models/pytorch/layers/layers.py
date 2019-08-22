""" Contains pytorch modules compatible with ConvBlock interface. """

import math
from functools import reduce
import operator
import numpy as np
import torch
from torch.autograd.function import Function
import torch.nn.functional as F

from ..utils import INT_TYPES, transform_to_int_tuple
from ..bases import Module, Layer
from ..bases import ConvModule

from .conv_block import ConvBlock


class FlattenFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return input.reshape(input.shape[0], -1)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.input
        return grad_output.reshape(*[int(v) for v in input.shape])

    @staticmethod
    def symbolic(g, input):
        r = g.op("Flatten", input)
        return r


class Identity(Module):

    def __init__(self, input_shape, **kwargs):
        """ Identity mapping layer.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor. Note that
            batch dimension is not taken into account.
        """
        super().__init__(input_shape, **kwargs)

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        """ Forward pass method for identity layer.

        Parameters
        ----------
        inputs : Tensor
            input tensor.

        Returns
        -------
        Tensor
            flattened input tensor.
        """
        return inputs


@ConvBlock.register_option(name='<')
class Flatten(Module):
    """ Flatten input tensor. """

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        """ Forward pass method for flatten layer.

        Parameters
        ----------
        inputs : Tensor
            input tensor.

        Returns
        -------
        Tensor
            flattened input tensor.
        """
        return FlattenFunction.apply(inputs)

    @property
    def output_shape(self) -> 'NDArray[int]':
        """ Get shape of the output tensor. """
        return np.array([np.prod(self.input_shape)])

    def __repr__(self) -> str:
        """ String representtion of flatten layer. """
        s = "{name}(input_shape={input_shape}, output_shape={output_shape})"
        return s.format(name=self.__class__.__name__,
                        input_shape=tuple(self.input_shape),
                        output_shape=self.output_shape[0])


@ConvBlock.register_option(name='u')
class Upsample(Module):

    def __init__(self, input_shape: 'ArrayLike',
                 scale: int = 2, kernel_size: 'ArrayLike[int]' = 3,
                 mode: str = 'linear'):
        """ Generalized upsampling layer.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor. Note that
            batch dimension is not taken into account.
        scale : int
            scale factor.
        kernel_size : int, Tuple[int], List[int] or NDArrray[int]
            kernel_size required by unpooling operation.
        mode : upsampling mode
            can be 'linear', 'nearest' ('bilinear' and 'trilinear'
            values are also supported).
        """
        super().__init__(input_shape)
        self.scale = int(scale)
        self.kernel_size = transform_to_int_tuple(kernel_size,
                                                  'kernel_size',
                                                  self.ndims - 1)
        if mode == 'linear':
            if self.ndims == 2:
                self.mode = 'linear'
            elif self.ndims == 3:
                self.mode = 'bilinear'
            elif self.ndims == 4:
                self.mode = 'trilinear'
        else:
            self.mode = mode

    @property
    def output_shape(self) -> 'NDArray[int]':
        """ Get shape of the output tensor. """
        return np.array([self.input_shape[0], *(self.scale * self.input_shape[1:])])

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        """ Forward pass method for upsampling layer. """
        if self.mode == 'unpool':
            raise NotImplementedError("Mode 'unpool' is not implemented yet.")
        else:
            return F.upsample(
                inputs, mode=self.mode,
                scale_factor=self.scale
                )

    def __repr__(self) -> str:
        """ String representation of upsampling layer. """
        s = "{name}(input_shape={input_shape}, "
        s += "output_shape={output_shape}, "
        s += "mode='{mode}')"
        return s.format(name=self.__class__.__name__, mode=self.mode,
                        input_shape=tuple(self.input_shape),
                        output_shape=tuple(self.output_shape))


@ConvBlock.register_option('n')
class BatchNorm(Layer):

    def __init__(self, input_shape: 'ArrayLike[int]',
                 eps: float = 1e-05, momentum: float = 0.1,
                 affine: bool = True):
        """ Generalized batch normalization layer.

        Parameters
        ----------
        input_shape : int, Tuple[int], List[int] or NDArray[int]
            shape of the input tensor. Note that batch dimension is not
            taken into account.
        eps : float
            epsilon value, required by batch normalization operation.
            Default is 1e-05.
        momentum : float
            value of momentum for exponential moving average computation.
            Default is 0.1.
        affine : bool
            this parameter is required by native pytorch batch normalization
            layer. Default is True.

        Raises
        ------
        ValueError
            if input_shape argument has length greater than 4.
        """
        super().__init__(input_shape)
        if self.ndims <= 2:
            self.layer = torch.nn.BatchNorm1d(self.in_channels, eps,
                                              momentum, affine)
        elif self.ndims == 3:
            self.layer = torch.nn.BatchNorm2d(self.in_channels, eps,
                                              momentum, affine)
        elif self.ndims == 4:
            self.layer = torch.nn.BatchNorm3d(self.in_channels, eps,
                                              momentum, affine)
        else:
            raise ValueError("Incorrect input shape.")


@ConvBlock.register_option(name='i')
class InstanceNorm(Layer):

    def __init__(self, input_shape: 'ArrayLike[int]',
                 eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True):
        """ Generalized instance normalization layer.

        Parameters
        ----------
        input_shape : int, Tuple[int], List[int] or NDArray[int]
            shape of the input tensor. Note that batch dimension is not
            taken into account.
        eps : float
            epsilon value, required by batch normalization operation.
            Default is 1e-05.
        momentum : float
            value of momentum for exponential moving average computation.
            Default is 0.1.
        affine : bool
            this parameter is required by native pytorch batch normalization
            layer. Default is True.

        Raises
        ------
        ValueError
            if input_shape argument has length greater than 4.
        """
        super().__init__(input_shape)
        if self.ndims <= 2:
            layer = torch.nn.InstanceNorm1d(self.in_channels, eps,
                                            momentum, affine)
        elif self.ndims == 3:
            layer = torch.nn.InstanceNorm2d(self.in_channels, eps,
                                            momentum, affine)
        elif self.ndims == 4:
            layer = torch.nn.InstanceNorm3d(self.in_channels, eps,
                                            momentum, affine)
        else:
            raise ValueError("Incorrect input shape.")

        self.layer = layer


@ConvBlock.register_option(name='f')
class Linear(Module):

    def __init__(self, input_shape: 'ArrayLike[int]',
                 out_features: int, bias: bool = True, **kwargs):
        """ Dense layer.

        Parameters
        ----------
        input_shape : int, Tuple[int], List[int] or NDArray[int]
            shape of the input tensor. Note that batch dimension is not
            taken into account.
        out_features : int
            number of features in the output tensor.
            Output tensor is considered to be 2D.
        bias : bool
            whether to use bias or not. Default is False.
        kwargs : dict
            these keyword arguments will be ignored.

        Note that ndimage tensor must be first flatten before linear layer
        can be correctly applied.
        """
        super().__init__(input_shape)
        self.out_features = out_features
        self.in_features = int(self.input_shape[0])
        self.weight = torch.nn.Parameter(torch.Tensor(self.out_features,
                                                      self.in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    @property
    def output_shape(self) -> 'NDArray[int]':
        """ Get shape of the output tensor. """
        return np.array([self.out_features])

    def reset_parameters(self) -> None:
        """ Parameters initialization. """
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input: 'Tensor') -> 'Tensor':
        """ Forward pass method for fully-connected layer. """
        if len(input.size()) != 2:
            input = input.view(input.size(0), -1)
        return F.linear(input, self.weight, self.bias)

    def __repr__(self) -> str:
        """ String representation of the fully-connected layer. """
        s = (self.__class__.__name__ + '('
            + 'in_features=' + str(self.in_features)
            + ', out_features=' + str(self.out_features))

        if self.bias is None:
            s += ', bias=False'
        else:
            s += ', bias=True'
        s += ')'
        return s


@ConvBlock.register_option(name='l')
class Lambda(Module):

    def __init__(self, input_shape: 'ArrayLike[int]', op: 'callable',
                 output_shape: 'ArrayLike[int]' = None,
                 annotation=None):
        super().__init__(input_shape)
        if not callable(op):
            raise TypeError("Argument 'op' must be callable.")
        if output_shape is None:
            self._output_shape = self.input_shape
        else:
            self._output_shape = np.array(output_shape, dtype=np.int)
        self.op = op
        self.annotation = annotation

    @property
    def output_shape(self) -> 'NDArray[int]':
        """ Get shape of the output tensor. """
        return self._output_shape

    def forward(self, inputs):
        """ Forward pass method for Lambda layer. """
        return self.op(inputs)

    def __repr__(self) -> str:
        """ String representation of Lambda layer. """
        if self.annotation is None:
            return super().__repr__()
        else:
            return self.__class__.__name__ + '(' + self.annotation + ')'


@ConvBlock.register_option(name='s')
class ChannelsShuffle(Module):

    def __init__(self, input_shape: 'ArrayLike[int]'):
        """ Choose random permutation for channels shuffle. """
        super().__init__(input_shape)
        permutation = np.random.permutation(input_shape[0])
        permutation = torch.LongTensor(permutation)
        self.register_buffer('permutation', torch.LongTensor(permutation))

    @property
    def output_shape(self) -> 'NDArray[int]':
        """ Get shape of the output tensor. """
        return self.input_shape

    def forward(self, inputs):
        """ Forward pass method for ChannelsShuffle layer. """
        return inputs[:, self.permutation, ...]


@ConvBlock.register_option(name='d')
class Dropout(Layer):

    def __init__(self, input_shape: 'ArrayLike[int]', p: float = 0.35,
                 inplace: bool = True, mode: str = 'pixels'):
        """" Generalized dropout layer.

        Parameters
        ----------
        input_shape : int, Tuple[int], List[int] or NDArray[int]
            shape of the input tensor. Note that batch dimension is not
            taken into account.
        p : float
            probability of dropout. Default is 0.35.
        inplace : bool
            whether to use dropout inplace. Default is True.
        mode : str
            dropout mode. Can be 'pixels' or 'channels'. If pixels, then
            dropout is applied pixelwise, otherwise channelwise
            version of dropout is used. Default is 'pixels'.
        ndims : int
            number of spatial dimensions in the input tensor.
        """
        super().__init__(input_shape)
        if not(isinstance(mode, str) and mode in ('pixels', 'channels')):
            raise ValueError("Argument mode must be one of 'pixels' or 'channels'")

        if mode == 'pixels':
            layer = torch.nn.Dropout(p=p, inplace=inplace)

        if self.ndims == 4:
            layer = torch.nn.Dropout3d(p=p, inplace=inplace)
        elif self.ndims == 3:
            layer = torch.nn.Dropout2d(p=p, inplace=inplace)
        elif self.ndims <= 1 :
            layer = torch.nn.Dropout(p=p, inplace=inplace)
        else:
            raise ValueError("Argument 'ndims' must be  one of "
                             + "(1, 2, 3, 4) int values, "
                             + "but got {}.".format(self.ndims))

        self.layer = layer
