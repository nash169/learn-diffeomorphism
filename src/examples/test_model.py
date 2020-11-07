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

# Meshgrid for test points
resolution = 100
x, y = np.meshgrid(np.linspace(-0.5, 0.5, resolution),
                   np.linspace(-0.5, 0.5, resolution))
pos_test = np.array([x.ravel(order="F"), y.ravel(order="F")]).transpose()

pos_test_tensor = torch.from_numpy(
    pos_test).float().to(device).requires_grad_(True)

# Vector field
vel_test = net(pos_test_tensor).cpu().detach().numpy()
dx = vel_test[:, 0].reshape(resolution, -1, order="F")
dy = vel_test[:, 1].reshape(resolution, -1, order="F")

# Diffeomorphism
psi_test = net.diffeomorphism_(pos_test_tensor)

# Potential function
phi_test, _ = net.potential(psi_test)

psi_test = psi_test.cpu().detach().numpy()
phi_test = phi_test.cpu().detach().numpy()

psi_x = psi_test[:, 0].reshape(resolution, -1, order="F")
psi_y = psi_test[:, 1].reshape(resolution, -1, order="F")

phi_test = phi_test.reshape(resolution, -1, order="F")

# Plot
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(pos[:, 0], pos[:, 1], s=5, marker='o', c='r')
ax1.streamplot(x, y, dx, dy, density=[0.5, 1])

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
contour = ax2.contourf(x, y, phi_test, 500, cmap="viridis")
ax2.contour(x, y, phi_test, 10, cmap=None, colors='#f2e68f')
fig2.colorbar(contour)


fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection="3d")
surf = ax3.plot_surface(x, y, phi_test, cmap="viridis",
                        linewidth=0, antialiased=False)
fig3.colorbar(surf)

plt.show()
