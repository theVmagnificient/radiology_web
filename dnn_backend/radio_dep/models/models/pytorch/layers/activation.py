""" Contains various custom pytorch activation functions. """

import torch
import torch.nn.functional as F

from ..bases import Layer
from .conv_block import ConvBlock


def softmax1d(inputs: 'Tensor') -> 'Tensor':
    """ Apply softmax along channels dimension to batch of 1D signals.

    Parameters
    ----------
    inputs : Tensor
        3D input tensor. First dimension is considered to be associated with
        batch items, second with channels.

    Returns
    -------
    Tensor
        tensor containing result of softmax operation.
    """
    batch_size, num_channels = int(inputs.size(0)), int(inputs.size(1))
    size = int(inputs.size(2))
    x = (
        inputs
        .permute(0, 2, 1)
        .contiguous()
        .view(-1, num_channels)
    )
    x = (
        F.softmax(x, dim=1)
        .view(batch_size, size, num_channels)
        .permute(0, 2, 1)
    )
    return x.contiguous()


def softmax2d(inputs: 'Tensor') -> 'Tensor':
    """ Apply softmax along channels dimension to batch of 2D images.

    Parameters
    ----------
    inputs : Tensor
        4D input tensor. First dimension is considered to be associated with
        batch items, second with channels.

    Returns
    -------
    Tensor
        tensor containing result of softmax operation.
    """
    batch_size, num_channels = int(inputs.size(0)), int(inputs.size(1))
    shape = (int(inputs.size(2)), int(inputs.size(3)))
    x = (
        inputs
        .permute(0, 2, 3, 1)
        .contiguous()
        .view(-1, num_channels)
    )
    x = (
        F.softmax(x, dim=1)
        .view(batch_size, *shape, num_channels)
        .permute(0, 3, 1, 2)
    )
    return x.contiguous()


def softmax3d(inputs: 'Tensor') -> 'Tensor':
    """ Apply softmax along channels dimension to batch of 3D images.

    Parameters
    ----------
    inputs : Tensor
        5D input tensor. First dimension is considered to be associated with
        batch items, second with channels.

    Returns
    -------
    Tensor
        tensor containing result of softmax operation.
    """
    batch_size, num_channels = int(inputs.size(0)), int(inputs.size(1))
    shape = (int(inputs.size(2)), int(inputs.size(3)), int(inputs.size(4)))
    x = (
        inputs
        .permute(0, 2, 3, 4, 1)
        .contiguous()
        .view(-1, num_channels)
    )
    x = (
        F.softmax(x, dim=1)
        .view(batch_size, *shape, num_channels)
        .permute(0, 4, 1, 2, 3)
    )
    return x.contiguous()


def softmax(inputs: 'Tensor') -> 'Tensor':
    """ Apply softmax to input ndimage represented by torch Tensor. """
    if len(inputs.size()) == 2:
        return F.softmax(inputs, dim=1)
    elif len(inputs.size()) == 3:
        return softmax1d(inputs)
    elif len(inputs.size()) == 4:
        return softmax2d(inputs)
    elif len(inputs.size()) == 5:
        return softmax3d(inputs)


class Softmax(torch.nn.Module):
    """ Softmax activation function generalized for 1D, 2D, 3D images. """

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        """ Forward pass method for Softmax activation function.

        Parameters
        ----------
        inputs : Tensor
            input tensor.

        Returns
        -------
        Tensor
        """
        return softmax(inputs)


class Softmin(torch.nn.Module):
    """ Softmin activation function generalized for 1D, 2D, 3D images. """

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        """ Forward pass method for Softmin activation function.

        Parameters
        ----------
        inputs : Tensor
            input Tensor

        Returns
        -------
        Tensor
        """
        return softmax(-inputs)


@ConvBlock.register_option(name='a')
class Activation(Layer):
    """ Generalized activation layer. """

    def __init__(self, input_shape, activation='relu', alpha=1.0, inplace=True,
                 init=0.1, negative_slope=0.01, num_parameters=1, **kwargs):
        """ Generalized activation layer.

        Parameters
        ----------
        input_shape : int, Tuple[int], List[int] or NDArray[int]
            shape of the input tensor. Note that batch dimension is not
            taken into account.
        activation : str
            possible values: 'relu', 'prelu', 'elu',
            'selu', 'sigmoid', 'softmax' or 'leaky_relu'.
            Default is 'relu'.
        alpha : float
            alpha value for 'elu' activation function. Default is 1.0.
        inplace : bool
            put activation inplace. This parameter
            exists only for 'relu', 'leaky_relu',
            'elu' and 'selu' activation functions. Default is True.
        init : float
            init argument required by 'prelu' activation. Default is 0.1.
        negative_slope : float
            slope in negative halfspace. This argument required by 'leaky_relu'.
            Default is 0.01.
        num_parameters : int
            required by 'prelu' activation. Default is 1.
        **kwargs : dict
            these parameters will be ignored.

        Raises
        ------
        ValueError
            if argument 'activation' is not str or None value.
        """
        super().__init__(input_shape)
        if not (isinstance(activation, str) or activation is None):
            raise ValueError("Argument 'activation' must have "
                             + "type 'str' or be None.")
        activation = 'linear' if activation is None else activation
        activation = activation.lower()
        activation = activation.strip()

        inplace = kwargs.get('inplace', True)
        init = kwargs.get('init', 0.1)
        alpha = kwargs.get('alpha', 1.0)
        beta = kwargs.get('beta', 1.0)
        threshold = kwargs.get('threhshold', 20)
        negative_slope = kwargs.get('negative_slope', 0.01)
        num_parameters = kwargs.get('num_parameters', 1)
        dim = kwargs.get('dim', None)
        if activation == 'relu':
            self.layer = torch.nn.ReLU(inplace)
        elif activation == 'relu6':
            self.layer = torch.nn.ReLU6(inplace)
        elif activation == 'sigmoid':
            self.layer = torch.nn.Sigmoid()
        elif activation == 'leaky_relu':
            self.layer = torch.nn.LeakyReLU(negative_slope, inplace)
        elif activation == 'elu':
            self.layer = torch.nn.ELU(alpha, inplace)
        elif activation == 'selu':
            self.layer = torch.nn.SELU(inplace)
        elif activation == 'prelu':
            self.layer = torch.nn.PReLU(num_parameters, init)
        elif activation == 'softmax':
            self.layer = Softmax()
        elif activation == 'softmin':
            self.layer = Softmin()
        elif activation == 'softplus':
            self.layer = torch.nn.Softplus(beta, threshold)
        elif activation == 'linear':
            self.layer = None
        else:
            raise ValueError("Argument 'activation' must be one "
                             + "of following values: 'relu', 'leaky_relu', "
                             + "'sigmoid', 'linear', 'elu' or None.")

    def forward(self, inputs: 'Tensor') -> 'Tensor':
        """ Forward pass method.

        Parameters
        ----------
        inputs : Tensor
            input tensor.

        Returns
        -------
        Tensor
            result of activaton function application.
        """
        if self.layer is None:
            return inputs
        return self.layer.forward(inputs)
