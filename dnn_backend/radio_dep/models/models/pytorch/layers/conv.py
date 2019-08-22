""" Contains pytorch convolutional layers compatible with ConvBlock interface. """

import math
import numpy as np
import torch
import torch.nn.functional as F

from ..bases import transform_to_int_tuple
from ..bases import ConvModule

from ..utils import compute_direct_output_shape, compute_transposed_output_shape
from ..utils import compute_direct_same_padding, compute_transposed_same_cropping
from ..utils import crop, pad
from ..utils import INT_TYPES, FLOAT_TYPES

from .conv_block import ConvBlock


class BaseConvLayer(ConvModule):

    def __init__(self, input_shape, filters, kernel_size=3,
                 stride=1, dilation=1, groups=1, bias=False):
        """ Base class for convolutional layers generalized for different dims.

        All convolutional layers from this module slightly
        extends functionality of original torch.nn.Conv* modules in four
        main aspects:
        1) Shape of the input tensor is passed as argument of constructor.

        2) Shape of the output tensor can be accessed by 'output_shape'
        property of the Conv* module.

        3) Different padding modes for 'same' mode. It means that there is
        no need to compute padding size for operation to make output tensor
        shape match input tensor's shape.

        4) All arguments of Conv* operations can be lists, tuples or
        ndarrays of int, np.int, np.int32 or int64 type.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor. Note that
            batch dimension is not taken in account.
        filters : int
            number of channels in the output tensor.
        kernel_size : int, Tuple[int], List[int] or NDArray[int]
            size of convolution or deconvolution kernel along each dimension.
        stride : int, Tuple[int], List[int] or NDArray[int]
            size of stride along each dimension. Default is 1.
        dilation : int, Tuple[int], List[int] or NDArray[int]
            dilation rate along each dimension. Default is 1.
        groups : int
            number of groups. Default is 1.
        bias : bool
            whether to use bias or not. Default is False.
        """
        super().__init__(input_shape, kernel_size, stride, dilation)
        self.filters = int(filters)
        self.groups = int(groups)
        self.weight = self._create_conv_weight_tensor()
        if bias:
            self.bias = self._create_conv_bias_tensor()
        else:
            self.register_parameter('bias', None)
        self._reset_parameters()

    def _create_conv_bias_tensor(self) -> 'torch.nn.Parameter':
        """ Create bias tensor for conv operation and wrap it as Parameter. """
        return torch.nn.Parameter(torch.Tensor(self.filters))

    def _create_conv_weight_tensor(self) -> 'torch.nn.Parameter':
        """ Create weigth tensor for conv operation and wrap is as Parameter. """
        weight = torch.Tensor(self.filters, self.in_channels // self.groups,
                              *self.kernel_size)
        return torch.nn.Parameter(weight)

    def _reset_parameters(self) -> None:
        """ Initialize parameters of convolutional layer. """
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


@ConvBlock.register_option(name='c', vectorized_params=('kernel_size',
                                                        'stride', 'dilation'))
class Conv(BaseConvLayer):

    def __init__(self, input_shape, filters, kernel_size=3, stride=1,
                 dilation=1, groups=1, padding='constant', bias=False):
        """ Direct convolution layer generalized for different dims.

        This layer slightly extends functionality of original
        torch.nn.Conv* modules in four
        main aspects:

        1) Shape of the input tensor is passed as argument of constructor.

        2) Shape of the output tensor can be accessed by 'output_shape'
        property of the Conv* module.

        3) Different padding modes for 'same' mode. It means that there is
        no need to compute padding size for operation to make output tensor
        shape match input tensor's shape.

        4) All arguments of Conv* operations can be lists, tuples or
        ndarrays of int, np.int, np.int32 or int64 type.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor. Note that
            batch dimension is not taken in account.
        filters : int
            number of channels in the output tensor.
        kernel_size : int, Tuple[int], List[int] or NDArray[int]
            size of convolution kernel along each dimension.
        stride : int, Tuple[int], List[int] or NDArray[int]
            size of stride along each dimension. Default is 1.
        dilation : int, Tuple[int], List[int] or NDArray[int]
            dilation rate along each dimension. Default is 1.
        groups : int
            number of groups. Default is 1.
        padding : str or int
            padding mode. Can be 'constant', 'reflect',
            'replicate', int or None. Default is 'constant'. If has int type
            then value will be used as 'value' argument in 'constant' mode.
        bias : bool
            whether to use bias or not. Default is False.
        """
        super().__init__(input_shape, filters, kernel_size,
                         stride, dilation, groups, bias)

        if isinstance(padding, (*INT_TYPES, *FLOAT_TYPES)):
            self.padding_mode = 'constant'
            self._padding_value = padding
        else:
            self.padding_mode = padding
            self._padding_value = 0.0

        if self.padding_mode == 'valid' or self.padding_mode is None:
            self.pad_sizes = [0] * (self.ndims - 1) * 2
        else:
            self.pad_sizes = compute_direct_same_padding(self.kernel_size,
                                                         self.stride,
                                                         self.dilation)

        _shape = compute_direct_output_shape(self.input_shape[1:],
                                             self.kernel_size,
                                             self.stride,
                                             self.dilation,
                                             self.pad_sizes)

        self._output_shape = np.array([self.filters, *_shape], dtype=np.int)

    def __repr__(self) -> str:
        """ String representation of convolutional layer. """
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if tuple(self.pad_sizes) != (0,) * len(self.pad_sizes):
            s += ', padding={padding}'
            s += ', padding_mode={padding_mode}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        values_dict = {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'padding': tuple(self.pad_sizes),
            'padding_mode': self.padding_mode,
            'groups': self.groups,
            'bias': self.bias,
            'dilation': self.dilation,
            'stride': self.stride
        }
        return s.format(name=self.__class__.__name__, **values_dict)

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        """ Forward pass method for tranposed convolution layer.

        Parameters
        ----------
        inputs : Tensor
            input tensor for transposed convolution layer.

        Returns
        -------
        Tensor
            result of convolutional operation applied to the input tensor.
        """
        x = pad(inputs, self.pad_sizes,
                mode=self.padding_mode,
                value=self._padding_value)
        conv_args = (x, self.weight, self.bias, self.stride, 0,
                     self.dilation, self.groups)

        if self.ndims == 2:
            return F.conv1d(*conv_args)
        elif self.ndims == 3:
            return F.conv2d(*conv_args)
        elif self.ndims == 4:
            return F.conv3d(*conv_args)


@ConvBlock.register_option(name='t', vectorized_params=('kernel_size',
                                                        'stride',
                                                        'dilation'))
class ConvTransposed(BaseConvLayer):

    def _create_conv_weight_tensor(self) -> 'torch.nn.Parameter':
        """ Create weight tensor for transposed convolution. """
        weight = torch.Tensor(self.in_channels // self.groups,
                              self.filters, *self.kernel_size)
        return torch.nn.Parameter(weight)

    def __init__(self, input_shape, filters, kernel_size=3,
                 stride=1, dilation=1, groups=1, crop=True, bias=False):
        """ Transposed convolution layer generalized for different dims.

        This layer slightly extends functionality of original
        torch.nn.TransposedConv* modules in four
        main aspects:

        1) Shape of the input tensor is passed as argument of constructor.

        2) Shape of the output tensor can be accessed by 'output_shape'
        property of the Conv* module.

        3) Different padding modes for 'same' mode. It means that there is
        no need to compute padding size for operation to make output tensor
        shape match input tensor's shape.

        4) All arguments of Conv* operations can be lists, tuples or
        ndarrays of int, np.int, np.int32 or int64 type.

        Parameters
        ----------
        input_shape : Tuple[int], List[int] or NDArray[int]
            shape of the input tensor. Note that
            batch dimension is not taken in account.
        filters : int
            number of channels in the output tensor.
        kernel_size : int, Tuple[int], List[int] or NDArray[int]
            size of deconvolution kernel along each dimension.
        stride : int, Tuple[int], List[int] or NDArray[int]
            size of stride along each dimension. Default is 1.
        dilation : int, Tuple[int], List[int] or NDArray[int]
            dilation rate along each dimension. Default is 1.
        groups : int
            number of groups. Default is 1.
        crop : bool
            whether to crop output tensor to have the same spatial
            shape as input tensor or not. Default is True.
        bias : bool
            whether to use bias or not. Default is False.
        """
        super().__init__(input_shape, filters, kernel_size,
                         stride, dilation, groups, bias)
        self.crop = crop
        if self.crop:
            self.crop_sizes = compute_transposed_same_cropping(self.kernel_size,
                                                               self.stride,
                                                               self.dilation)
        else:
            self.crop_sizes = [0] * (self.ndims - 1) * 2

        _shape = compute_transposed_output_shape(self.input_shape[1:],
                                                 self.kernel_size,
                                                 self.stride,
                                                 self.dilation,
                                                 self.crop_sizes)

        self._output_shape = np.array([self.filters, *_shape], dtype=np.int)

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        """ Forward pass method for transposed convolution layer.

        Parameters
        ----------
        inputs : torch Tensor
            input tensor for transposed convolution layer.

        Returns
        -------
        torch Tensor
            result of convolutional operation applied to the input tensor.
        """
        conv_args = (inputs, self.weight, self.bias,
                     self.stride, 0, 0, self.groups, self.dilation)

        if self.ndims == 2:
            x = F.conv_transpose1d(*conv_args)
        elif self.ndims == 3:
            x = F.conv_transpose2d(*conv_args)
        elif self.ndims == 4:
            x = F.conv_transpose3d(*conv_args)

        if self.crop:
            x = crop(x, self.crop_sizes)
        return x

    def __repr__(self) -> 'str':
        """ String representation of transposed convolution layer. """
        s = ('{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if tuple(self.crop_sizes) != (0,) * len(self.crop_sizes):
            s += ', cropping={cropping}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        values_dict = {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'cropping': tuple(self.crop_sizes),
            'groups': self.groups,
            'bias': self.bias,
            'dilation': self.dilation,
            'stride': self.stride
        }
        return s.format(name=self.__class__.__name__, **values_dict)
