import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# def Num2Bit(Num, B):
#     Num_ = Num.type(torch.uint8)
#     def integer2bit(integer, num_bits=B * 2):
#         dtype = integer.type()
#         exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
#         exponent_bits = exponent_bits.repeat(integer.shape + (1,))
#         out = integer.unsqueeze(-1) // 2 ** exponent_bits
#         return (out - (out % 1)) % 2
#     bit = integer2bit(Num_)
#     bit = (bit[:, :, B:]).reshape(-1, Num_.shape[1] * B)
#     return bit.type(torch.float32)

# Num = torch.abs(torch.randn(2,1)) * 10
# print(Num)
# print(Num2Bit(Num, 3))

# x = torch.tensor([1, 2, 3, 4])
# y = torch.unsqueeze(x, 1)
# print(torch.unsqueeze(y, 2))
# print('Epoch: [{0}][{1}/{2}]\t' 'Loss {loss:.4f}\t'.format(5, 6, 7, loss=8))

# import numpy as np
# import torch
# from modelDesign import AutoEncoder,DatasetFolder #*
# import os
# import torch.nn as nn
# import scipy.io as sio

# mat = sio.loadmat('channelData/H_4T4R.mat')
# data = mat['H_4T4R']
# data = data.astype('float32')

# m = nn.Conv1d(16, 33, 3, stride=2)
# input = torch.randn(20, 16, 50)
# output = m(input)
# print(output.size())

t = OrderedDict([('l0.weight', tensor([[ 0.1400, 0.4563, -0.0271, -0.4406],
                                   [-0.3289, 0.2827, 0.4588, 0.2031]])),
             ('l0.bias', tensor([ 0.0300, -0.1316])),
             ('l1.weight', tensor([[0.6533, 0.3413]])),
             ('l1.bias', tensor([-0.1112]))])