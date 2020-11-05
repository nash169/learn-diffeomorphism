#!/usr/bin/env python

import torch
import numpy as np

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

# Create trainer
trainer = Trainer(net, pos, vel)

# Set trainer optimizer (this is not very clean)
trainer.optimizer = torch.optim.Adam(
    trainer.model.parameters(), lr=1e-4,  weight_decay=1e-8)

# Set trainer loss
trainer.loss = torch.nn.SmoothL1Loss()

# Set trainer options
trainer.options(normalize=False, shuffle=True, print_loss=True, epochs=100)

# Train model
trainer.train()

# Save model
trainer.save()
