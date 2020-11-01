#!/usr/bin/env python

import torch

from torch.autograd.functional import jacobian
from torch.autograd import grad
from src.learn_diffeomorphism import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

point = torch.rand(2, 5).to(device)

net = Dynamics(point.size(1), 10, 2).to(device)

out = net.forward(point)

# print(out)

# jac = jacobian(net, test_point)

# print(jacobian(net, test_point))

# test_point = torch.rand(4, 2)


# def test_fun(x):
#     result = torch.empty(x.size(0), 2)
#     result[:, 0] = x[:, 0]*x[:, 1]
#     result[:, 1] = x[:, 0]*x[:, 1]
#     return result


# jac = jacobian(test_fun, test_point)

# print(grad(outputs=out, inputs=test_point,
#            grad_outputs=torch.ones_like(out), retain_graph=True))

# print(test_point)
# print(out)
