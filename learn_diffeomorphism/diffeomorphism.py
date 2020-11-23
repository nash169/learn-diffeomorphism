#!/usr/bin/env python

import torch
import torch.nn as nn
from .coupling_layer import CouplingLayer


class Diffeomorphism(nn.Module):
    def __init__(self, dim, num_features, num_diff, length=1):
        super(Diffeomorphism, self).__init__()

        self.dim_ = dim

        self.prediction_ = nn.Sequential(*(CouplingLayer(dim, num_features, i % 2, length)
                                           for i in range(num_diff)))

    def forward(self, x):
        return self.prediction_(x)

    def jacobian(self, x):
        y = self.prediction_(x)

        jac = torch.empty(x.size(0)*self.dim_, self.dim_).to(x.device)

        for i in range(self.dim_):
            grad = torch.autograd.grad(
                y[:, i], x, grad_outputs=torch.ones_like(y[:, i]), create_graph=True)[0]
            jac[:, i] = grad.reshape(1, -1)

        return jac.reshape(-1, self.dim_, self.dim_).permute(0, 2, 1), y

    def invert(self, x):
        for m in reversed(self.prediction_):
            x = m.invert(x)

        return x
