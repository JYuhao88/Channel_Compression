#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.layers import GDN
from compressai.models import CompressionModel
from compressai.models import CompressionModel
from compressai.models.utils import conv, deconv
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

class Network(CompressionModel):
    def __init__(self, N=128):
        super().__init__(entropy_bottleneck_channels=N)
        self.encode = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
        )
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.decode = nn.Sequential(
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )
    
    def forward(self, x):
        y = self.encode(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.decode(y_hat)
        return x_hat, y_likelihoods
        
    def compress(self, x):
        y = self.encode(x)
        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": y.size()[-2:]}
net = Network()    
x = torch.rand(1, 3, 64, 64)
y = net.compress(x)






