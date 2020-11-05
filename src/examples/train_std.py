#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt

from src.learn_diffeomorphism import Dynamics
from src.learn_diffeomorphism.utils import linear_map


torch.manual_seed(1)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

data = np.loadtxt("data/GShape.csv")

pos = data[:, 0:2]
vel = data[:, 2:4]
acc = data[:, 4:6]
time = data[:, -2]
steps = data[:, -1]

# Normalization
# lower, upper = -0.5, 0.5
# pos[:, 0] = linear_map(pos[:, 0], np.min(pos[:, 0]),
#                        np.max(pos[:, 0]), lower, upper)
# pos[:, 1] = linear_map(pos[:, 1], np.min(pos[:, 1]),
#                        np.max(pos[:, 1]), lower, upper)
pos = (pos - pos.mean(axis=0)) / pos.std(axis=0)

plt.scatter(pos[:, 0], pos[:, 1])
plt.show()

# Convert data
pos = torch.from_numpy(pos).float().to(device)
vel = torch.from_numpy(vel).float().to(device)

# mu, std = pos.mean(), pos.std()
# pos.sub_(mu).div_(std)
pos.requires_grad = True

# Shuffle data
idx = torch.randperm(pos.size(0))
pos = pos[idx, :]
vel = vel[idx, :]

dim = pos.size(1)
fourier_features = 200
coupling_layers = 10
kernel_length = 0.45

ds = Dynamics(dim, fourier_features, coupling_layers,
              pos[-1, :], kernel_length).to(device)

# Set optimizer
optimizer = optim.Adam(ds.parameters(), lr=1e-4, weight_decay=1e-8)
# optimizer = optim.SGD(ds.parameters(), lr=1e-4,  weight_decay=1e-8)

# Set loss function
loss_func = nn.MSELoss()

# Batch and number of epochs
BATCH_SIZE = 100
EPOCH = 100

# start training
for epoch in range(EPOCH):
    print("EPOCH: ", epoch)
    for batch in range(0, pos.size(0), BATCH_SIZE):  # for each training step
        x = pos.narrow(0, batch, BATCH_SIZE)
        y = vel.narrow(0, batch, BATCH_SIZE)

        # input x and predict based on x
        prediction = ds(x)

        # must be (1. nn output, 2. target)
        loss = loss_func(prediction, y)

        print(loss)

        # clear gradients for next train
        optimizer.zero_grad()

        # backpropagation, compute gradients
        loss.backward(retain_graph=True)

        # apply gradients
        optimizer.step()
