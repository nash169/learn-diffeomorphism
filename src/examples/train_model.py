#!/usr/bin/env python

import argparse
import torch
import numpy as np
import os

from src.learn_diffeomorphism import *

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
data = np.loadtxt(os.path.join('data', '{}.csv'.format(args.data)))
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
    trainer.model.parameters(), lr=1e-4,  weight_decay=1e-10)

# Set trainer loss
trainer.loss = torch.nn.SmoothL1Loss()

# Set trainer options
trainer.options(normalize=False, shuffle=True, print_loss=True,
                epochs=1000, load_model=args.data)

# Train model
trainer.train()

# Save model
trainer.save(args.data)
