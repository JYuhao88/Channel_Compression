import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import OrderedDict

# m1 = nn.Conv1d(1, 1, 4, stride=4)
# m2 = nn.ConvTranspose1d(1, 1, 4, 4)

# input = torch.randn(512, 512)
# output = input.unsqueeze(1)
# output = m1(output)
# output = output.squeeze(1)
# print(output.size())

m = nn.Sigmoid()
input = torch.randn(2)
output = m(input)
print(output)