#!/usr/bin/env python

import torch
import torch.nn as nn

from torch.autograd.functional import jacobian
from .diffeomorphism import Diffeomorphism
from src.learn_diffeomorphism.utils import blk_matrix


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class Dynamics(nn.Module):
    def __init__(self, dim, num_features, num_diff, length=1):
        super(Dynamics, self).__init__()

        self.dim_ = dim

        self.diffeomorphism_ = Diffeomorphism(
            dim, num_features, num_diff,  length)

    def forward(self, x):
        jac = blk_matrix(jacobian(self.diffeomorphism_, x).sum(
            2).reshape(x.size(0)*self.dim_, self.dim_, 1).squeeze(2))

        return -torch.mv(torch.inverse(torch.sparse.mm(jac.transpose(
            1, 0), jac.to_dense())), self.diffeomorphism_(x).reshape(-1, 1).squeeze(1)).reshape(-1, self.dim_)
