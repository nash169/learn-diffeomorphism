#!/usr/bin/env python

import torch
import os
import numpy as np


class Trainer:
    __options__ = ['epochs', 'batch', 'normalize',
                   'shuffle', 'record_loss', 'print_loss', 'clip_grad', 'load_model']

    def __init__(self, model, input, target):
        # Set the model
        self.model = model

        # Set the input
        self.input = input

        # Set the target
        self.target = target

        # Set deault optimizer
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=1e-4,  weight_decay=1e-8)

        # Set default loss function
        self.loss = torch.nn.MSELoss()

        # Set default options
        self.options_ = dict(
            epochs=5,
            batch=None,
            normalize=False,
            shuffle=True,
            record_loss=False,
            print_loss=True,
            clip_grad=None,
            load_model=None)

    def options(self, **kwargs):
        for _, option in enumerate(kwargs):
            if option in self.__options__:
                self.options_[option] = kwargs.get(option)
            else:
                print("Warning: option not found -", option)

    def train(self):
        # Load model if requested
        if self.options_['load_model'] is not None:
            self.load(self.options_['load_model'])

        # Set batch to dataset size as default
        if self.options_['batch'] is None:
            self.options_['batch'] = self.input.size(0)

        # Normalize dataset
        if self.options_['normalize']:
            mu, std = self.input.mean(0), self.input.std(0)
            self.input.sub_(mu).div_(std)

        # Create dataset
        torch_dataset = torch.utils.data.TensorDataset(self.input, self.target)

        # Create loader
        loader = torch.utils.data.DataLoader(
            dataset=torch_dataset,
            batch_size=self.options_['batch'],
            shuffle=self.options_['shuffle'],
            num_workers=0
        )

        # Open file
        if self.options_["record_loss"]:
            loss_log = np.empty([1,3])

        # start training
        for epoch in range(self.options_['epochs']):
            for iter, (batch_x, batch_y) in enumerate(loader):  # for each training step
                b_x = torch.autograd.Variable(batch_x, requires_grad=True)
                b_y = torch.autograd.Variable(batch_y)

                # input x and predict based on x
                prediction = self.model(b_x)

                # must be (1. nn output, 2. target)
                loss = self.loss(prediction, b_y)

                # Print loss
                if self.options_["print_loss"]:
                    print("EPOCH: ", epoch, "ITER: ",
                          iter, "LOSS: ", loss.item())
                
                # Record loss (EPOCH,ITER,LOSS)
                if self.options_["record_loss"]:
                    # print(np.array([epoch,iter,loss.item()])[:,np.newaxis])
                    loss_log = np.append(loss_log,np.array([epoch,iter,loss.item()])[np.newaxis,:],axis=0)
                    
                # clear gradients for next train
                self.optimizer.zero_grad()

                # backpropagation, compute gradients
                loss.backward()

                # Clip grad if requested
                if self.options_["clip_grad"] is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.options_["clip_grad"])

                # apply gradients
                self.optimizer.step()

        # Close file
        if self.options_["record_loss"]:
            if not os.path.exists('models'):
                os.makedirs('models')

            np.savetxt(os.path.join('models', '{}.csv'.format(self.options_["record_loss"])), loss_log)

    def save(self, file):
        if not os.path.exists('models'):
            os.makedirs('models')

        torch.save(self.model.state_dict(), os.path.join(
            'models', '{}.pt'.format(file)))

    def load(self, file):
        self.model.load_state_dict(torch.load(
            os.path.join('models', '{}.pt'.format(file)), map_location=torch.device(self.input.device)))

    # Model
    @property
    def model(self):
        return self.model_

    @model.setter
    def model(self, value):
        self.model_ = value

    # Input
    @property
    def input(self):
        return self.input_

    @input.setter
    def input(self, value):
        self.input_ = value

    # Target
    @property
    def target(self):
        return self.target_

    @target.setter
    def target(self, value):
        self.target_ = value

    # Optimizer
    @property
    def optimizer(self):
        return self.optimizer_

    @optimizer.setter
    def optimizer(self, value):
        self.optimizer_ = value

    # Loss function
    @property
    def loss(self):
        return self.loss_

    @loss.setter
    def loss(self, value):
        self.loss_ = value
