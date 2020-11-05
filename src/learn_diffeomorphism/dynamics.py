#!/usr/bin/env python

import torch
import torch.nn as nn

from .diffeomorphism import Diffeomorphism
from src.learn_diffeomorphism.utils import blk_matrix

from time import time


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class Dynamics(nn.Module):
    def __init__(self, dim, num_features, num_diff, attractor, length=1):
        super(Dynamics, self).__init__()

        self.dim_ = dim

        self.attractor_ = attractor.unsqueeze(0)

        self.diffeomorphism_ = Diffeomorphism(
            dim, num_features, num_diff,  length)

    def forward(self, x):
        m = x.size(0)
        y = self.diffeomorphism_(x)

        attractor = self.diffeomorphism_(self.attractor_)[0]

        jac = torch.empty(m*self.dim_, self.dim_).to(device)

        for i in range(self.dim_):
            grad = torch.autograd.grad(
                y[:, i], x, grad_outputs=torch.ones_like(y[:, i]), retain_graph=True)[0]
            jac[:, i] = grad.reshape(1, -1)

        jac = jac.reshape(m, self.dim_, self.dim_).permute(0, 2, 1)

        result = torch.empty(m, self.dim_).to(device)

        for i in range(m):
            result[i, :] = -torch.mv(torch.inverse(torch.mm(jac[i].transpose(
                1, 0), jac[i])), y[i, :]-attractor)

        return result
