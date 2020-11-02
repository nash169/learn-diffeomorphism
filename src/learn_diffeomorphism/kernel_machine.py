#!/usr/bin/env python

import torch
import torch.nn as nn
import numpy as np

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class KernelMachine(nn.Module):  # inheriting from nn.Module!

    def __init__(self, dim, num_features, match_dim, length=1):
        super(KernelMachine, self).__init__()

        # Sample dimension
        self.dim_ = dim

        # Number of samples
        self.num_features_ = num_features

        # Matching dimension
        self.match_dim_ = match_dim

        # Kernel length
        self.length_ = length

        # Fourier features parameters
        self.a_ = MultivariateNormal(torch.zeros(
            self.num_features_, self.dim_), torch.eye(self.dim_)/np.power(self.length_, 2)).sample().to(device)
        self.b_ = Uniform(torch.zeros(self.num_features_),
                          torch.ones(self.num_features_)*2*np.pi).sample().to(device)

        self.prediction_ = torch.nn.Linear(
            self.num_features_*self.match_dim_, 1, bias=False).to(device)

    def fourier_features(self, x):
        num_points = x.size(0)

        fourier_features = torch.sqrt(torch.tensor(2/self.num_features_))*torch.cos(torch.sum(self.a_.repeat(
            num_points, 1)*x.repeat_interleave(self.num_features_, 0), axis=1).unsqueeze(1) + self.b_.unsqueeze(1).repeat(num_points, 1))

        index = torch.cat([torch.arange(num_points).repeat_interleave(self.num_features_*self.match_dim_).unsqueeze(0), torch.arange(
            self.match_dim_, dtype=torch.uint8).repeat(num_points*self.num_features_).unsqueeze(0), torch.arange(self.num_features_*self.match_dim_, dtype=torch.uint8).repeat(num_points).unsqueeze(0)], dim=0).to(device)

        return torch.sparse_coo_tensor(index, fourier_features.repeat_interleave(
            self.match_dim_, 0).squeeze(1), (num_points, self.match_dim_, self.num_features_*self.match_dim_))

    def forward(self, x):
        # check sparse tensor problem
        return self.prediction_(self.fourier_features(x).to_dense()).squeeze(2)
