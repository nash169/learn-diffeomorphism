#!/usr/bin/env python

import torch
import torch.nn as nn
import numpy as np

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform


class KernelMachine(nn.Module):  # inheriting from nn.Module!

    def __init__(self, dim, num_features, match_dim, length=1):
        super(KernelMachine, self).__init__()

        # Number of samples
        self.num_features_ = num_features

        # Fourier features parameters
        a = MultivariateNormal(torch.zeros(self.num_features_, dim), torch.eye(
            dim)/np.power(length, 2)).sample()
        b = Uniform(torch.zeros(self.num_features_), torch.ones(
            self.num_features_)*2*np.pi).sample()

        # Linear layer with fixed parameters
        self.linear_clamped_ = torch.nn.Linear(dim,  self.num_features_)

        # Set weights and biases for the clamped layer
        self.linear_clamped_.weight = nn.Parameter(a, requires_grad=False)
        self.linear_clamped_.bias = nn.Parameter(b, requires_grad=False)

        # Linear layer matching the output dimension
        self.prediction_ = torch.nn.Linear(
            self.num_features_,  match_dim, bias=False)

        # Init weights to zero (identity diffeomorphism)
        # nn.init.zeros_(self.prediction_.weight.data)

    def fourier_features(self, x):
        return torch.sqrt(torch.tensor(2/self.num_features_))*torch.cos(self.linear_clamped_(x))

    def forward(self, x):
        return self.prediction_(self.fourier_features(x))
