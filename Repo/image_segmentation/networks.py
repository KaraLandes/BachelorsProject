import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam, SGD

class BaseNet(nn.Module):
    def __init__(self, n_in_channels: int = 1, n_hidden_layers: int = 3, n_kernels: int = 8, kernel_size: int = 7):
        """Simple CNN with `n_hidden_layers`, `n_kernels`, and `kernel_size` as hyperparameters"""
        super(BaseNet, self).__init__()

        self.cnn = []
        for i in range(n_hidden_layers):
            self.cnn.append(torch.nn.Conv2d(in_channels=n_in_channels, out_channels=n_kernels, kernel_size=kernel_size,
                                       bias=True, padding=int(kernel_size / 2)))
            self.cnn.append(torch.nn.ReLU())
            n_in_channels = n_kernels
        self.hidden_layers = torch.nn.Sequential(*self.cnn)
        self.output_layer = torch.nn.Conv2d(in_channels=n_in_channels, out_channels=1,
                                            kernel_size=kernel_size, bias=True, padding=int(kernel_size / 2))

    def forward(self, x):
        x = torch.tensor(x, dtype=self.cnn[0].weight.dtype)
        x = x.reshape(x.size()[0], 1, x.size()[1], x.size()[2])
        """Apply CNN to input `x` of shape (N, n_channels, X, Y), where N=n_samples and X, Y are spatial dimensions"""
        cnn_out = self.hidden_layers(x)  # apply hidden layers (N, n_in_channels, X, Y) -> (N, n_kernels, X, Y)
        pred = self.output_layer(cnn_out)  # apply output layer (N, n_kernels, X, Y) -> (N, 1, X, Y)
        return pred