#!/usr/bin/env python

import torch
import torch.nn as nn
import numpy as np

from .kernel_machine import KernelMachine
from time import time

# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")


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
        # z_a = x[:, self.inx_za]
        # z_b = x[:, self.inx_zb]

        # z_b = z_b*torch.exp(self.scaling_(z_a)) + self.translation_(z_a)

        # result = torch.empty(z_b.size(0), self.dim_).to(device)
        # result[:, self.inx_za] = z_a
        # result[:, self.inx_zb] = z_b
        result = x.clone()
        result[:, self.inx_zb] = x[:, self.inx_zb] * \
            torch.exp(self.scaling_(x[:, self.inx_za])) + \
            self.translation_(x[:, self.inx_za])

        return result
