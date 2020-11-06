#!/usr/bin/env python

import torch
import torch.nn as nn

from .diffeomorphism import Diffeomorphism


class Dynamics(nn.Module):
    def __init__(self, dim, num_features, num_diff, attractor, length=1):
        super(Dynamics, self).__init__()

        self.dim_ = dim

        self.attractor_ = attractor.unsqueeze(0)

        self.diffeomorphism_ = Diffeomorphism(
            dim, num_features, num_diff,  length)

    def forward(self, x):
        # Calculate attactor location in the linear space
        attractor = self.diffeomorphism_(self.attractor_)[0]

        # Calculate diffeomorphism and jacobian
        J, y = self.diffeomorphism_.jacobian(x)

        # Calculate minus gradient of the potential function in linear space
        _, dy = self.potential(attractor - y)

        # Return DS in the original space
        return torch.bmm(torch.inverse(J), dy.unsqueeze(2)).squeeze()

    def potential(self, y):
        return torch.bmm(y.unsqueeze(1), y.unsqueeze(2)), y
