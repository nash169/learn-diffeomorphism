#!/usr/bin/env python

import torch
import torch.nn as nn
import numpy as np

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform


class KernelMachine(nn.Module):  # inheriting from nn.Module!

    def __init__(self, dim, num_samples, match_dim, length=1):
        super(KernelMachine, self).__init__()

        # Sample dimension
        self.dim_ = dim

        # Number of samples
        self.num_samples_ = num_samples

        # Matching dimension
        self.match_dim_ = match_dim

        # Kernel length
        self.length_ = length

        # Fourier features
        a = MultivariateNormal(torch.zeros(
            self.num_samples_, self.dim_), torch.eye(self.dim_)/np.power(self.length_, 2))
        b = Uniform(torch.zeros(self.num_samples_),
                    torch.ones(self.num_samples_)*2*np.pi)
        a_samples = a.sample()
        b_samples = b.sample()

        self.fourier_features_ = []

        for i in range(self.num_samples_):
            self.fourier_features_.append(torch.nn.Sequential(
                torch.nn.Linear(self.dim_, 1)
            ))
            self.fourier_features_[i].weights = a_samples[i, :]
            self.fourier_features_[i].bias = b_samples[i]

        self.prediction_ = torch.nn.Linear(
            self.num_samples_*self.match_dim_, 1, bias=False)

    def kronecker(self, A, B):
        return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))

    def phi(self, x):
        features = torch.cat([torch.unsqueeze(b(x), 1)
                              for b in self.fourier_features_], 1)

        results = torch.empty(
            (features.size(0), features.size(1) * self.match_dim_, self.match_dim_))

        for i in range(features.size(0)):
            results[i, :, :] = self.kronecker(
                np.sqrt(2/self.num_samples_)*torch.cos(features[i, :, :]), torch.eye(self.match_dim_))

        return results

    def forward(self, x):
        return self.prediction_(self.phi(x).permute(0, 2, 1)).squeeze(2)
