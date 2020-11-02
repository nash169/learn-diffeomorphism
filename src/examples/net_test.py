#!/usr/bin/env python

import torch

from torch.autograd.functional import jacobian
from torch.autograd.functional import vjp
from torch.autograd.functional import jvp
from torch.autograd import grad
from src.learn_diffeomorphism import *
from src.learn_diffeomorphism.utils import blk_matrix

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# point = torch.rand(2, 5).to(device)

# net = KernelMachine(point.size(1), 10, 3).to(device)
# # net2 = KernelMachineVec(point.size(1), 10, 2).to(device)

# out1 = net.forward(point)

# # out3 = net2.phi(point)


# dim = 2
# num_points = 2
# num_feature = 3

# test = torch.rand(num_points*num_feature, 1)

# index = torch.cat([torch.arange(num_points).repeat_interleave(num_feature*dim).unsqueeze(0), torch.arange(num_feature*dim, dtype=torch.uint8).repeat(num_points).unsqueeze(0), torch.arange(
#     dim, dtype=torch.uint8).repeat(num_points*num_feature).unsqueeze(0)], dim=0)

# s = torch.sparse_coo_tensor(index, test.repeat_interleave(
#     dim, 0).squeeze(1), (num_points, num_feature*dim, dim))
# print(out)

# jac = jacobian(net, test_point)

# print(jacobian(net, test_point))

test_point = torch.rand(5, 5, requires_grad=True).to(device)


# def test_fun(x):
#     result = torch.empty(x.size(0), 2)
#     result[:, 0] = x[:, 0]*x[:, 1]
#     result[:, 1] = x[:, 0]**2*x[:, 1]
#     return result


# jac = jacobian(test_fun, test_point).sum(2)

net = Dynamics(5, 10, 3)

out1 = net.forward(test_point)

# tensor = test_fun(test_point)

# u, v = torch.split(tensor, 1, dim=1)
# du = torch.autograd.grad(
#     u, test_point, grad_outputs=torch.ones_like(u), retain_graph=True)[0]
# dv = torch.autograd.grad(
#     v, test_point, grad_outputs=torch.ones_like(v))[0]

# print(grad(outputs=out, inputs=test_point,
#            grad_outputs=torch.ones_like(out), retain_graph=True))

# print(test_point)
# print(out)
