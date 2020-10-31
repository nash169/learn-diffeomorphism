#!/usr/bin/env python

import torch
import torch.nn as nn

from torch.autograd.functional import jacobian
from .diffeomorphism import Diffeomorphism


class Dynamics(nn.Module):
    def __init__(self, dim, num_samples, num_diff, length=1):
        super(Dynamics, self).__init__()

        self.dim_ = dim

        self.diffeomorphism_ = Diffeomorphism(
            dim, num_samples, num_diff,  length)

    def forward(self, x):
        result = torch.empty_like(x)

        y = self.diffeomorphism_(x)

        jac = jacobian(self.diffeomorphism_, x)

        J = torch.empty(self.dim_, self.dim_)

        for i in range(x.size(0)):
            for j in range(x.size(1)):
                J[j, :] = jac[i, j, i]
            # print(J)
            # print(J.size())
            G = torch.mm(J, J)
            result[i, :] = -torch.mv(torch.inverse(G), y[i, :])

        return result
