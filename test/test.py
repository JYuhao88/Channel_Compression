import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models
import random
import numpy as np
import os

import sys
import math

import torchsnooper


# x = torch.rand(3, 2, 4, 16)
# x = x - 0.5
# r = torch.sqrt(x[:,0,:,:]**2+x[:,1,:,:]**2).unsqueeze(1)
# # theta = torch.atan2(x[:,0,:,:],x[:,1,:,:]).unsqueeze(1)
# # x = torch.cat((x, r, theta), 1)

# (r_max, _) = torch.max(r, 3)
# r_mask = r_max<0.6
# r_mask = r_mask.int()
# r_mask = r_mask.unsqueeze(3)

# mask = r_mask.expand(-1, -1, -1, 4)
# # print(r_mask[1])
# print(r_mask)
# print(mask)

# x = torch.rand(128, 24)
# x = x.unsqueeze(1)
# x = x.unsqueeze(3)
# print(x.size())

# x = torch.randn(2, 3, requires_grad = True)
# y = torch.randn(2, 3, requires_grad = False)
# z = torch.randn(2, 3, requires_grad = False)
# m=x+y+z
# with torch.no_grad():
#     w = x + y + z
# print(w)
# print(m)
# print(w.requires_grad)
# print(w.grad_fn)
# print(m.requires_grad)

# x = torch.tensor([1.], requires_grad=True)

# @torch.no_grad()
# def doubler(x):
#     return x * 2
# z = doubler(x)
# print(z.requires_grad)
# y = z**2 + x
# print(y.requires_grad)

SEED = 42
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(SEED)
print(torch.randperm(4))
# a=torch.rand(3,5)
# print(a)

# a=a[torch.randperm(a.size(0))]
# print(a)

# a=a[:,torch.randperm(a.size(1))]
# print(a)