import numpy as np
import torch

from ..layers import ConvBlock
from ..utils import LIST_TYPES


class ASPPBase(torch.nn.Module):

    def __init__(self, input_shape, filters, rates, layout,
                 body_block, input_block=None, head_block=None, how='+'):
        super().__init__()

        if not isinstance(rates, LIST_TYPES):
            raise ValueError("Argument rates must be list-like. "
                             + "Got type {}.".format(type(rates)))

        if how not in ('.', '+'):
            raise ValueError("Argument 'how' supports only '+' "
                             + "and '.' values. Got {}.".format(how))

        self._input_shape = input_shape
        self.how = how
        self.branches = torch.nn.ModuleList()
        for i, (ifilters, irate) in enumerate(zip(filters, rates)):
            x = body_block(
                input_shape=input_shape, layout=layout,
                c=dict(filters=ifilters, dilations=irate)
            )
            self.branches.append(x)

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        pass

    def forward(self, inputs):
        outputs = []
        for module in self.branches:
            outputs.append(module(inputs))
        if self.how == '.':
            x = torch.cat(outputs, dim=1)
        else:
            x = sum(outputs)
        return x
