#!/usr/bin/env python

# import torch
import torch.nn as nn
from .coupling_layer import CouplingLayer
from time import time
# import torch.nn.functional as F
# import torch.optim as optim
# import torch.utils.data as Data
# import matplotlib.pyplot as plt
# import numpy as np

# from torch.autograd import Variable

# from torch.distributions.multivariate_normal import MultivariateNormal
# from torch.distributions.uniform import Uniform
# from torch.autograd import grad
# from torch.autograd.functional import jacobian


class Diffeomorphism(nn.Module):
    def __init__(self, dim, num_features, num_diff, length=1):
        super(Diffeomorphism, self).__init__()

        self.prediction_ = nn.Sequential(*(CouplingLayer(dim, num_features, i % 2, length)
                                           for i in range(num_diff)))

    def forward(self, x):
        return self.prediction_(x)
