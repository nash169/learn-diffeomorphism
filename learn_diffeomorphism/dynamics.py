#!/usr/bin/env python

import torch
import torch.nn as nn

from .diffeomorphism import Diffeomorphism


class Dynamics(nn.Module):
    def __init__(self, dim, num_features, num_diff, attractor, length=1):
        super(Dynamics, self).__init__()

        self.attractor_ = attractor.unsqueeze(0)

        self.diffeomorphism = Diffeomorphism(
            dim, num_features, num_diff,  length)

    # Attractor setter/getter
    @property
    def attractor(self):
        return self.attractor_

    @attractor.setter
    def attractor(self, value):
        self.attractor_ = value

    # Diffeomorphism setter/getter
    @property
    def diffeomorphism(self):
        return self.diffeomorphism_

    @diffeomorphism.setter
    def diffeomorphism(self, value):
        self.diffeomorphism_ = value

    # Forward network pass
    def forward(self, x):
        # Calculate diffeomorphism and jacobian
        J, y = self.diffeomorphism.jacobian(x)

        # Calculate minus gradient of the potential function in linear space
        _, dy = self.potential(y)

        # Return DS in the original space
        return torch.bmm(torch.inverse(J), -dy.unsqueeze(2)).squeeze()

    # Potential function
    def potential(self, y):
        # Calculate attactor location in the linear space
        attractor = self.diffeomorphism(self.attractor_)[0]

        # Centered poisition
        y_hat = y - attractor

        return torch.bmm(y_hat.unsqueeze(1), y_hat.unsqueeze(2)), y_hat
