#!/usr/bin/env python

import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def linear_map(x, xmin, xmax, ymin, ymax):
    m = (ymin - ymax)/(xmin-xmax)
    q = ymin - m*xmin

    y = m*x + q

    return y


def blk_matrix(a):
    m, n = a.size()

    assert m % n == 0, "Dimension incorrect"

    x, y = torch.meshgrid(torch.arange(m, dtype=torch.uint8).to(device),
                          torch.arange(n, dtype=torch.uint8).to(device))

    y = y + torch.arange(0, m, n,
                         dtype=torch.uint8).repeat_interleave(n).unsqueeze(1).to(device)

    return torch.sparse_coo_tensor(torch.cat([x.reshape(-1, 1).transpose(1, 0), y.reshape(-1, 1).transpose(1, 0)], dim=0), a.reshape(-1, 1).squeeze(1), (m, m))
