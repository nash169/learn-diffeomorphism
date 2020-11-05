#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt

from torch.autograd import Variable
from src.learn_diffeomorphism import Dynamics
from src.learn_diffeomorphism.utils import linear_map


torch.manual_seed(1)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

data = np.loadtxt("data/Leaf_2.csv")

pos = data[:, 0:2]
vel = data[:, 2:4]
acc = data[:, 4:6]
time = data[:, -2]
steps = data[:, -1]

# Normalization
lower, upper = -0.5, 0.5
pos[:, 0] = linear_map(pos[:, 0], np.min(pos[:, 0]),
                       np.max(pos[:, 0]), lower, upper)
pos[:, 1] = linear_map(pos[:, 1], np.min(pos[:, 1]),
                       np.max(pos[:, 1]), lower, upper)

plt.scatter(pos[:, 0], pos[:, 1])
plt.show()

pos = torch.from_numpy(pos).float().to(device)
vel = torch.from_numpy(vel).float().to(device)

dim = pos.size(1)
fourier_features = 200
coupling_layers = 10
kernel_length = 0.45

ds = Dynamics(dim, fourier_features, coupling_layers,
              pos[-1, :], kernel_length).to(device)

# Set optimizer
optimizer = optim.Adam(ds.parameters(), lr=1e-4,  weight_decay=1e-10)

# Set loss function
loss_func = nn.SmoothL1Loss()

# Batch and number of epochs
BATCH_SIZE = pos.size(0)
EPOCH = 100

# Create dataset
torch_dataset = Data.TensorDataset(pos, vel)

loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

# start training
for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
        b_x = Variable(batch_x, requires_grad=True)
        b_y = Variable(batch_y)

        # input x and predict based on x
        prediction = ds(b_x)

        # must be (1. nn output, 2. target)
        loss = loss_func(prediction, b_y)

        print(loss)

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
