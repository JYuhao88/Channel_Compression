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


x = torch.rand(3, 2, 4, 16)
x = x - 0.5
r = torch.sqrt(x[:,0,:,:]**2+x[:,1,:,:]**2).unsqueeze(1)
# theta = torch.atan2(x[:,0,:,:],x[:,1,:,:]).unsqueeze(1)
# x = torch.cat((x, r, theta), 1)

(r_max, _) = torch.max(r, 3)
r_mask = r_max<0.6
r_mask = r_mask.int()
r_mask = r_mask.unsqueeze(3)

mask = r_mask.expand(-1, -1, -1, 4)
# print(r_mask[1])
print(r_mask)
print(mask)

# x = torch.rand(128, 24)
# x = x.unsqueeze(1)
# x = x.unsqueeze(3)
# print(x.size())