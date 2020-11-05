#!/usr/bin/env python

import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from src.learn_diffeomorphism import *

# CPU/GPU setting
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Load data
data = np.loadtxt("data/Leaf_2_ref.csv")
pos = data[:, 0:2]
vel = data[:, 2:4]
pos = torch.from_numpy(pos).float().to(device)
vel = torch.from_numpy(vel).float().to(device)

# Model options
dim = pos.size(1)
fourier_features = 200
coupling_layers = 10
kernel_length = 0.45

# Create model
net = Dynamics(dim, fourier_features, coupling_layers,
               pos[-1, :], kernel_length).to(device)

# Load params
net.load_state_dict(torch.load(os.path.join(
    'models', '{}.pt'.format("learned_model"))))

# Calculate test point solution
resolution = 100
x, y = np.meshgrid(np.linspace(-0.5, 0.5, resolution),
                   np.linspace(-0.5, 0.5, resolution))
pos_test = np.array([x.ravel(order="F"), y.ravel(order="F")]).transpose()

pos_test_tensor = torch.from_numpy(
    pos_test).float().to(device).requires_grad_(True)

vel_test = net(pos_test_tensor).cpu().detach().numpy()

dx = vel_test[:, 0].reshape(resolution, -1, order="F")
dy = vel_test[:, 1].reshape(resolution, -1, order="F")

# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(pos[:, 0], pos[:, 1], s=5, marker='o', c='r')
ax.streamplot(x, y, dx, dy, density=[0.5, 1])
plt.show()
