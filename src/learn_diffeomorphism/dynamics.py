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
        # Calculate diffeomorphism and jacobian
        J, y = self.diffeomorphism_.jacobian(x)

        # Calculate minus gradient of the potential function in linear space
        _, dy = self.potential(y)

        # Return DS in the original space
        return torch.bmm(torch.inverse(J), -dy.unsqueeze(2)).squeeze()

    def potential(self, y):
        # Calculate attactor location in the linear space
        attractor = self.diffeomorphism_(self.attractor_)[0]

        # Centered poisition
        y_hat = y - attractor

        return torch.bmm(y_hat.unsqueeze(1), y_hat.unsqueeze(2)), y_hat
