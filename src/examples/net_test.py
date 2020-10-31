#!/usr/bin/env python

import torch

from torch.autograd.functional import jacobian
from torch.autograd import grad
from src.learn_diffeomorphism import Dynamics

net = Dynamics(5, 10, 2)

test_point = torch.rand(2, 5, requires_grad=True)
out = net.forward(test_point)

jac = jacobian(net, test_point)

print(jacobian(net, test_point))

test_point = torch.rand(4, 2)


def test_fun(x):
    result = torch.empty(x.size(0), 2)
    result[:, 0] = x[:, 0]*x[:, 1]
    result[:, 1] = x[:, 0]*x[:, 1]
    return result


jac = jacobian(test_fun, test_point)

print(grad(outputs=out, inputs=test_point,
           grad_outputs=torch.ones_like(out), retain_graph=True))

print(test_point)
print(out)
