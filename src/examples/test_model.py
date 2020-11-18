#!/usr/bin/env python

import argparse
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from src.learn_diffeomorphism import *
from src.learn_diffeomorphism.utils import linear_map


# Parse arguments
parser = argparse.ArgumentParser(
    description='Diffeomorphic mapping for learning Dynamical System')

parser.add_argument('--data', type=str, default='Leaf_2_ref',
                    help='Name of the dataset/model')

args = parser.parse_args()

# CPU/GPU setting
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Load data
data = np.loadtxt(os.path.join('rsc', '{}.csv'.format(args.data)))
pos_train = data[:, 0:2]
vel = data[:, 2:4]

# Normalization
lower, upper = -0.5, 0.5
pos_train[:, 0] = linear_map(pos_train[:, 0], np.min(pos_train[:, 0]),
                             np.max(pos_train[:, 0]), lower, upper)
pos_train[:, 1] = linear_map(pos_train[:, 1], np.min(pos_train[:, 1]),
                             np.max(pos_train[:, 1]), lower, upper)

# Model options
dim = pos_train.shape[1]
fourier_features = 200
coupling_layers = 10
kernel_length = 0.45

# Create model
net = Dynamics(dim, fourier_features, coupling_layers,
               torch.from_numpy(pos_train[-1, :]).float().to(device), kernel_length).to(device)

# Load params
net.load_state_dict(torch.load(os.path.join(
    'models', '{}.pt'.format(args.data)), map_location=torch.device(device)))

# Meshgrid for test points
resolution = 100
x, y = np.meshgrid(np.linspace(-0.5, 0.5, resolution),
                   np.linspace(-0.5, 0.5, resolution))
pos_test = np.array(
    [x.ravel(order="F"), y.ravel(order="F")]).transpose()

pos_test_tensor = torch.from_numpy(
    pos_test).float().to(device).requires_grad_(True)

# Vector field
vel_test = net(pos_test_tensor).cpu().detach().numpy()
dx = vel_test[:, 0].reshape(resolution, -1, order="F")
dy = vel_test[:, 1].reshape(resolution, -1, order="F")

# Diffeomorphism
psi = net.diffeomorphism_(pos_test_tensor)

# Potential function
phi, _ = net.potential(psi)

psi = psi.cpu().detach().numpy()
phi = phi.cpu().detach().numpy()

psi_x = psi[:, 0].reshape(resolution, -1, order="F")
psi_y = psi[:, 1].reshape(resolution, -1, order="F")

phi = phi.reshape(resolution, -1, order="F")

# Plot
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(pos_train[:, 0], pos_train[:, 1], s=5, marker='o', c='r')
ax1.streamplot(x, y, dx, dy, density=[0.5, 1])

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
contour = ax2.contourf(x, y, phi, 500, cmap="viridis")
ax2.contour(x, y, phi, 10, cmap=None, colors='#f2e68f')
fig2.colorbar(contour)


fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection="3d")
surf = ax3.plot_surface(x, y, phi, cmap="viridis",
                        linewidth=0, antialiased=False)
fig3.colorbar(surf)

plt.show()
