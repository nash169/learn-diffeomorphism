#!/usr/bin/env python

import torch.nn as nn
from .coupling_layer import CouplingLayer
from time import time


class Diffeomorphism(nn.Module):
    def __init__(self, dim, num_features, num_diff, length=1):
        super(Diffeomorphism, self).__init__()

        self.prediction_ = nn.Sequential(*(CouplingLayer(dim, num_features, i % 2, length)
                                           for i in range(num_diff)))

    def forward(self, x):
        return self.prediction_(x)
