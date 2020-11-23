#!/usr/bin/env python

import torch
import torch.nn as nn
import numpy as np

from .kernel_machine import KernelMachine


class CouplingLayer(nn.Module):
    def __init__(self, dim, num_features, transf_index, length=1):
        super(CouplingLayer, self).__init__()

        self.dim_ = dim

        # Even elements
        inx_even = np.array([i for i in range(self.dim_) if i % 2 == 0])

        # Odd elements
        inx_odd = inx_even + 1
        if inx_odd[-1] > self.dim_ - 1:
            inx_odd = np.delete(inx_odd, -1)

        if transf_index == 0:
            self.scaling_ = KernelMachine(
                np.ceil(self.dim_/2).astype('int'), num_features, np.floor(self.dim_/2).astype('int'), length)
            self.translation_ = KernelMachine(
                np.ceil(self.dim_/2).astype('int'), num_features, np.floor(self.dim_/2).astype('int'), length)
            self.inx_za = inx_even
            self.inx_zb = inx_odd
        else:
            self.scaling_ = KernelMachine(
                np.floor(self.dim_/2).astype('int'), num_features, np.ceil(self.dim_/2).astype('int'), length)
            self.translation_ = KernelMachine(
                np.floor(self.dim_/2).astype('int'), num_features, np.ceil(self.dim_/2).astype('int'), length)
            self.inx_za = inx_odd
            self.inx_zb = inx_even

    def forward(self, x):
        result = x.clone()

        result[:, self.inx_zb] = x[:, self.inx_zb] * \
            torch.exp(self.scaling_(x[:, self.inx_za])) + \
            self.translation_(x[:, self.inx_za])

        return result

    def invert(self, x):
        result = x.clone()

        result[:, self.inx_zb] = (x[:, self.inx_zb] - self.translation_(
            x[:, self.inx_za]))*torch.exp(-self.scaling_(x[:, self.inx_za]))

        return result
