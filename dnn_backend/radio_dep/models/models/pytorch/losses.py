''' Losses on PyTorch for various tasks '''

import torch


def dice_loss(y_true: 'Tensor', y_pred: 'Tensor') -> 'Tensor':
    """ Dice loss function implemented via pytorch.

    Parameters
    ----------
    y_true : Tensor
        tensor representing true mask.
    y_pred : Tensor
        tensor representing predicted mask.

    Returns
    -------
    Tensor
        dice loss value.
    """
    smooth = 1e-7
    batch_size = y_true.size(0)
    y_true, y_pred = y_true.view(-1).float(), y_pred.view(-1).float()
    return -2 * torch.sum(y_true * y_pred) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)


def tversky_loss(y_true: 'Tensor', y_pred: 'Tensor',
                 alpha=0.3, beta=0.7, smooth=1e-10) -> 'Tensor':
    """ Tversky loss function implemented via pytorch.

    Parameters
    ----------
    y_true : Tensor
        tensor representing true mask.
    y_pred : Tensor
        tensor representing predicted mask.
    alpha : float
        alpha coefficient used in alpha * torch.sum(y_true * (1 - y_pred)) part.
    beta : float
        beta coefficient used in beta * torch.sum((1 - y_pred) * y_true) part.
    smooth : float
        small value used to avoid division by zero error.

    Returns
    -------
    Tensor
        tversky loss value.
    """
    y_true, y_pred = y_true.view(-1).float(), y_pred.view(-1).float()
    truepos = torch.sum(y_true * y_pred)
    fp_and_fn = (alpha * torch.sum((1 - y_true) * y_pred)
                 + beta * torch.sum((1 - y_pred) * y_true))

    return -(truepos + smooth) / (truepos + smooth + fp_and_fn)


def log_loss(y_true: 'Tensor', y_pred: 'Tensor') -> 'Tensor':
    """ Log loss function implemented via pytorch.

    Parameters
    ----------
    y_true : Tensor
        tensor representing true mask.
    y_pred : Tensor
        tensor representing predicted mask.

    Returns
    -------
    Tensor
        log loss value.
    """
    smooth = 1e-7
    y_true, y_pred = y_true.view(-1).float(), y_pred.view(-1).float()
    return -torch.mean(y_true * torch.log(y_pred + smooth)
                       + (1 - y_true) * torch.log(1 - y_pred + smooth))


def dice_and_log_loss(y_true: 'Tensor', y_pred: 'Tensor', alpha=0.2) -> 'Tensor':
    """ Dice and log combination loss function implemented via pytorch.

    Parameters
    ----------
    y_true : Tensor
        tensor representing true mask.
    y_pred : Tensor
        tensor representing predicted mask.
    alpha : float
        alpha coefficient used as (1 - alpha) * dice_loss + alpha * log_loss.

    Returns
    -------
    Tensor
        dice + log loss value.
    """
    return ((1 - alpha) * dice_loss(y_true, y_pred)
            + alpha * log_loss(y_true, y_pred))
