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

x = torch.rand(128, 24, 16, 2)
x_hat = torch.rand(128, 24, 16, 2)
x_real = x[:, :, :, 0].view(len(x),-1) - 0.5
x_imag = x[:, :, :, 1].view(len(x),-1) - 0.5
x_hat_real = x_hat[:, :, :, 0].view(len(x_hat), -1) - 0.5
x_hat_imag = x_hat[:, :, :, 1].view(len(x_hat), -1) - 0.5
power = torch.sum(x_real**2 + x_imag**2, axis=1)
mse = torch.sum((x_real-x_hat_real)**2 + (x_imag-x_hat_imag)**2, axis=1)
nmse = mse/power

print(mse.size())