""" Contains various useful functions used in different submodules. """

import math
import numpy as np
import torch
import torch.nn.functional as F


INT_TYPES = (int, np.int, np.int32, np.int64)
FLOAT_TYPES = (float, np.float, np.float32, np.float64)
LIST_TYPES = (list, tuple, np.ndarray)


def unpack_dict_recursive(inputs: dict) -> list:
    """ Unpack values from dict in a recurrsive manner.

    Parameters
    ----------
    inputs : dict
        input dictionary.

    Returns
    -------
    list
        list containing values unpacked from dictionary.
    """
    outputs = []
    for value in inputs.values():
        if isinstance(value, dict):
            outputs.extend(unpack_dict_recursive(value))
            continue
        outputs.append(value)
    return outputs


def addindent(s_, num_spaces):
    """ Function that add given number of spaces instead '\n'."""
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(num_spaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


def transform_to_int_tuple(parameter: 'ArrayLike[int]',
                           name: str, length: int) -> 'ArrayLike[int]':
    """ Transform input parameter value to tuple of ints of given length.

    Parameters
    ----------
    parameter : int, Tuple[int], List[int] or NDArray[int]
        input parameter value.
    name : str
        name of parameter. Required by exception raising part of function.
    length : int
        length of output list with parameter values.

    Returns
    -------
    Tuple[int]

        If input parameter is list-like then all its values
        are transformed to int type and tuple of ints is returned.

        If input parameter is instance of int type then its
        value is transformed to int type and tuple containing
        this value repeated 'length' times is returned.

    Raises
    ------
    ValueError
        If input parameter has wrong type or has improper length (if list-like).
    """
    if isinstance(parameter, INT_TYPES):
        parameter = [int(parameter)] * length
    elif isinstance(parameter, LIST_TYPES):
        parameter = [int(item) for item in np.asarray(parameter).flatten()]
        if len(parameter) != length:
            raise ValueError("Argument {} has inproper lenght.".format(name)
                             + " Must have {}. ".format(length)
                             + "Got {}.".format(len(parameter)))
    else:
        raise ValueError("Argument {} must be int or ".format(name)
                         + "tuple, list, ndarray "
                         + "containing {} int values.".format(length))
    return tuple(parameter)


def transform_to_float_tuple(parameter: 'ArrayLike[float]',
                             name: str, length: int) -> 'ArrayLike[float]':
    """ Transform input parameter value to tuple of floats of given length.

    Parameters
    ----------
    parameter : float, Tuple[float], List[float] or NDArray[float]
        input parameter value.
    name : str
        name of parameter. Required by exception raising part of function.
    length : int
        length of output list with parameter values.

    Returns
    -------
    Tuple[float]

        If input parameter is list-like then all its values
        are transformed to int type and tuple of ints is returned.

        If input parameter is instance of int type then its
        value is transformed to int type and tuple containing
        this value repeated 'length' times is returned.

    Raises
    ------
    ValueError
        If input parameter has wrong type or has improper length (if list-like).
    """
    if isinstance(parameter, FLOAT_TYPES):
        parameter = [float(parameter)] * length
    elif isinstance(parameter, LIST_TYPES):
        parameter = [float(item) for item in np.asarray(parameter).flatten()]
        if len(parameter) != length:
            raise ValueError("Argtuemnt {} has inproper lenght.".format(name)
                             + " Must have {}.".format(length)
                             + "Got {}.".format(len(parameter)))
    else:
        raise ValueError("Argument {} must be float or ".format(name)
                         + "tuple, list, ndarray "
                         + "containing {} float values.".format(length))
    return tuple(parameter)


def compute_direct_same_padding(kernel_size: 'ArrayLike[int]',
                                stride: 'ArrayLike[int]',
                                dilation: 'ArrayLike[int]') -> 'List[int]':
    """ Compute padding size for 'SAME' mode in case of direct operations.

    Word 'transposed' in this context means the oposite to 'direct'.
    For instance, 'Convolution' and 'MaxPooling' are 'direct' operations
    while 'TransposedConvolution' and 'MaxUnpooling' are considered to
    be 'tranposed'.

    Parameters
    ----------
    kernel_size : Tuple[int], List[int] or NDArray[int]
        kernel_size of direct operation.
    stride : Tuple[int], List[int] or NDArray[int]
        stride of direct operation along different dimensions.
    dilation : Tuple[int], List[int] or NDArray[int]
        dilation of direct operation along different dimensions.

    Returns
    List[int]
        list containing padding sizes for both left and right sides for
        each spatial dimension.
    """
    pad_sizes = []
    kernel_size = [(k - 1) * d + 1 for k, d in zip(kernel_size, dilation)]
    for k, s in zip(kernel_size, stride):
        pad_sizes.append(math.ceil((k - 1) / 2))
        pad_sizes.append(math.floor((k - 1) / 2))
    return pad_sizes


def compute_transposed_same_cropping(kernel_size: 'ArrayLike[int]',
                                     stride: 'ArrayLike[int]',
                                     dilation: 'ArrayLike[int]') -> 'List[int]':
    """ Compute cropping size for 'SAME' mode in case of transposed operations.

    Word 'transposed' in this context means the oposite to 'direct'.
    For instance, 'Convolution' and 'MaxPooling' are 'direct' operations
    while 'TransposedConvolution' and 'MaxUnpooling' are considered to
    be 'tranposed'.

    Parameters
    ----------
    kernel_size : Tuple[int], List[int] or NDArray[int]
        kernel_size of direct operation.
    stride : Tuple[int], List[int] or NDArray[int]
        stride of direct operation along different dimensions.
    dilation : Tuple[int], List[int] or NDArray[int]
        dilation of direct operation along different dimensions.

    Returns
    -------
    List[int]
        list containing number of pixels to crop from both sides for each
        spatial dimension.
    """
    crop_sizes = []
    kernel_size = [(k - 1) * d + 1 for k, d in zip(kernel_size, dilation)]
    for k, s in zip(kernel_size, stride):
        overshoot = k - s
        crop_sizes.append(math.ceil(overshoot / 2))
        crop_sizes.append(math.floor(overshoot / 2))
    return crop_sizes


def compute_direct_output_shape(input_shape: 'ArrayLike[int]',
                                kernel_size: 'ArrayLike[int]',
                                stride: 'ArrayLike[int]',
                                dilation: 'ArrayLike[int]',
                                padding: 'ArrayLike[int]') -> 'NDArray[int]':
    """ Compute shape of the result tensor of direct operation.

    Word 'transposed' in this context means the oposite to 'direct'.
    For instance, 'Convolution' and 'MaxPooling' are 'direct' operations
    while 'TransposedConvolution' and 'MaxUnpooling' are considered to
    be 'tranposed'.

    Parameters
    ----------
    input_shape : List[int], Tuple[int] or NDArray[int]
        shape of the input tensor. Note that only spatial dimensions
        are taken into account, meaning that len(kernel_size) must be equal
        to len(input_shape).
    kernel_size : List[int], Tuple[int] or NDArray[int]
        kernel size of direct operation along different dimensions.
    stride : List[int], Tuple[int] or NDArray[int]
        stride of direct operation along different dimensions.
    dilation : List[int], Tuple[int] or NDArray[int]
        dilation of direct operation along different dimensions.
    padding : List[int], Tuple[int] or NDArray[int]
        padding sizes for both 'left' and 'right' sides for each dimension.
        Padding sizes must be provided in the following
        format: [Left_1, Right_1, Left_2, Right_2, ...]. Total length
        of padding argument must be equal to double length of 'kernel_size',
        'stride' or 'dilation' arguments.

    Returns
    -------
    NDArray[int]
        shape of the output tensor.
    """
    kernel_size = np.array(kernel_size)
    stride, dilation = np.array(stride), np.array(dilation)
    padding = np.array(padding).reshape(len(kernel_size), 2).sum(axis=1)
    output_shape = input_shape - (kernel_size - 1) * dilation - 1 + padding
    output_shape = np.floor(output_shape / stride) + 1
    return output_shape.astype(np.int)


def compute_transposed_output_shape(input_shape: 'ArrayLike[int]',
                                    kernel_size: 'ArrayLike[int]',
                                    stride: 'ArrayLike[int]',
                                    dilation: 'ArrayLike[int]',
                                    cropping: 'ArrayLike[int]') -> 'NDArray[int]':
    """ Compute shape of the result tensor of transposed operation.

    Word 'transposed' in this context means the oposite to 'direct'.
    For instance, 'Convolution' and 'MaxPooling' are 'direct' operations
    while 'TransposedConvolution' and 'MaxUnpooling' are considered to
    be 'tranposed'.

    Parameters
    ----------
    input_shape : List[int], Tuple[int] or NDArray[int]
        shape of the input tensor. Note that only spatial dimensions
        are taken into account, meaning that len(kernel_size) must be equal
        to len(input_shape).
    kernel_size : List[int], Tuple[int] or NDArray[int]
        kernel size of transposed operation along different dimensions.
    stride : List[int], Tuple[int] or NDArray[int]
        stride of transposed operation along different dimensions.
    dilation : List[int], Tuple[int] or NDArray[int]
        dilation of transposed operation along different dimensions.
    cropping : List[int], Tuple[int] or NDArray[int]
        cropping sizes for both 'left' and 'right' sides for each dimension.
        Cropping sizes must be provided in the following
        format: [Left_1, Right_1, Left_2, Right_2, ...]. Total length
        of padding argument must be equal to double length of 'kernel_size',
        'stride' or 'dilation' arguments.

    Returns
    -------
    NDAarray[int]
        shape of the output tensor.
    """
    kernel_size = np.array(kernel_size)
    stride, dilation = np.array(stride), np.array(dilation)
    cropping = np.array(cropping).reshape(len(kernel_size), 2).sum(axis=1)

    output_shape = stride * (input_shape - 1) + (kernel_size - 1) * dilation + 1
    output_shape -= cropping
    return output_shape.astype(np.int)


def crop(inputs: 'Tensor', sizes: 'ArrayLike[int]') -> 'Tensor':
    """ Crop input tensor.

    Parameters
    ----------
    inputs : Tensor
        input tensor.
    sizes : Tuple[int], List[int] or NDArray[int]
        crops sizes in format [Left_1, Right_1, Left_2, Right_2, ...]
        where index reflects spatial dimensions (first two are considered to
        be batch and channels dimensions).

    Returns
    -------
    Tensor
        result of cropping.
    """
    ndims = len(inputs.size()) - 2
    sizes = transform_to_int_tuple(sizes, 'sizes', 2 * ndims)
    if ndims == 1:
        slices = [slice(sizes[0], inputs.size(2) - sizes[1])]
        return inputs[:, :, slices[0]].contiguous()
    if ndims == 2:
        slices = [
            slice(sizes[0], inputs.size(2) - sizes[1]),
            slice(sizes[2], inputs.size(3) - sizes[3])
        ]
        return inputs[:, :, slices[0], slices[1]].contiguous()
    if ndims == 3:
        slices = [
            slice(sizes[0], inputs.size(2) - sizes[1]),
            slice(sizes[2], inputs.size(3) - sizes[3]),
            slice(sizes[4], inputs.size(4) - sizes[5])
        ]
        return inputs[:, :, slices[0], slices[1], slices[2]].contiguous()


def crop_as(x: 'Tensor', y: 'Tensor') -> 'Tensor':
    """ Crop first tensor to the same shape as second tensor.

    Parameters
    ----------
    x : Tensor
        input tensor to crop.
    y : Tensor
        tensor whose shape defines shape of first tensor after cropping.

    Returns
    -------
    Tensor
        first tensor cropped to have the same shape as second tensor.
    """
    diff = np.array(x.shape[2:]) - np.array(y.shape[2:])
    crop_size = []
    for i in range(len(diff)):
        crop_size.append(math.ceil(int(diff[i]) / 2))
        crop_size.append(math.floor(int(diff[i]) / 2))
    crop_size = np.array(crop_size, dtype=np.int)
    return crop(x, crop_size)


def pad(inputs: 'Tensor', sizes: 'ArrayLike[int]',
        mode: str = 'constant', value: float = 0.0) -> 'Tensor':
    """ Add padding to tensor.

    Parameters
    ----------
    inputs : Tensor
        input tensor.
    sizes : Tuple[int], List[int] or NDArray[int]
        padding sizes in format [Left_1, Right_1, Left_2, Right_2, ...]
        where index reflects spatial dimensions (first two are considered to
        be batch and channels dimensions).
    mode : str
        padding mode. Can be 'constant', 'reflect' or 'replicate'.
        Default is 'constant'. If mode is 'constant' then padding value
        can be specified by 'value' argument (default is 0.0), otherwise
        'value' argument will be ignored.

    Returns
    -------
    Tensor
        result of padding operation.
    """
    if mode in ('constant', 'reflect', 'replicate'):
        _sizes = np.array(sizes).reshape(len(sizes) // 2, 2)
        _sizes = tuple(int(value) for value
                       in _sizes[::-1, :].reshape(len(sizes)))
        return F.pad(inputs, _sizes, mode=mode, value=value)
    else:
        return inputs

def pad_as(x: 'Tensor', y: 'Tensor',
           mode: str = 'constant',
           value: float = 0.0) -> 'Tensor':
    """ Pad first tensor to the same shape as second tensor.

    Parameters
    ----------
    x : Tensor
        input tensor to pad.
    y : Tensor
        tensor whose shape defines shape of first tensor after adding padding.

    Returns
    -------
    Tensor
        first tensor with padding added to have the same shape as second tensor.
    """
    diff = np.array(y.shape[2:]) - np.array(x.shape[2:])
    pad_size = []
    for i in range(len(diff)):
        pad_size.append(math.ceil(int(diff[i]) / 2))
        pad_size.append(math.floor(int(diff[i]) / 2))
    pad_size = np.array(pad_size, dtype=np.int)
    return pad(x, pad_size, mode, value)
