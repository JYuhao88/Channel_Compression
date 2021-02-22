import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import sys
import math

import torchsnooper

# x = torch.rand(128, 2, 24, 16)
# r = torch.sqrt(x[:,0,:,:]**2+x[:,1,:,:]**2).unsqueeze(1)
# theta = torch.arctan(x[:,0,:,:]/x[:,1,:,:]).unsqueeze(1)
# x = torch.cat((x, r, theta), 1)
# print(x.size())

x = torch.rand(128, 2, 24, 16)
y = torch.rand(128, 2, 24, 16)
xy = x*y
print(xy.size())