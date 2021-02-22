'''
final
'''

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import OrderedDict
import torchsnooper


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

def conv(in_planes, out_planes, kernel_size=3, stride=1, groups=1):
    """convolution with padding"""
    if not isinstance(kernel_size, int):
        padding = [(i - 1) // 2 for i in kernel_size]
    else:
        padding = (kernel_size - 1) // 2
    return  nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                    padding=padding, groups=groups, bias=False)

class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))

class Att24x16(nn.Module):
    def __init__(self, in_ch, ch):
        super(Att24x16, self).__init__()
        self.ch = ch
        self.path1 = nn.Sequential(
            conv(ch, ch, 3),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            conv(ch, ch, [1, 7]),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            conv(ch, ch, [7, 1]),
        )
        self.path2 = nn.Sequential(
            conv(ch, ch, [1, 5]),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            conv(ch, ch, [5, 1]),
        )
        self.in_conv =  nn.Sequential(
            ConvBN(in_ch, ch, 1),
            nn.LeakyReLU(negative_slope=0.3, inplace=True))
        self.out_conv = conv(ch * 2, in_ch, 1)
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.identity = nn.Identity()
        self.bn = nn.BatchNorm2d(in_ch)

    def forward(self, x):
        identity = self.identity(x)
        x = self.in_conv(x)
        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.out_conv(out)
        out = self.relu(out) + identity
        return self.bn(out)

class Att4x24(nn.Module):
    def __init__(self, in_ch, ch):
        super(Att4x24, self).__init__()
        self.ch = ch
        self.attention1 = nn.Sequential(
            conv(in_ch, ch, 1),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            conv(ch, in_ch, [1, 3]),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
        )
        self.attention2 = nn.Sequential(
            conv(in_ch, ch, [1, 5]),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            conv(ch, in_ch, [1, 7]),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
        )
        self.attention3 = nn.Sequential(
            conv(in_ch, ch, [1, 3]),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            conv(ch, in_ch, 1),
            nn.LeakyReLU(negative_slope=0.3, inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        x = x.permute(0,3,1,2)
        out1 = self.pool(self.attention1(x))
        out2 = self.pool(self.attention2(out1))
        out3 = self.pool(self.attention3(out2))
        out = torch.cat((out1*x, out2*x, out3*x), 2)
        return out.permute(0, 2, 3, 1)

class Att4x16(nn.Module):
    def __init__(self, in_ch, ch):
        super(Att4x16, self).__init__()
        self.ch = ch
        self.attention1 = nn.Sequential(
            conv(in_ch, ch, 1),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            conv(ch, in_ch, [1, 3]),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
        )
        self.attention2 = nn.Sequential(
            conv(in_ch, ch, [1, 5]),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            conv(ch, in_ch, [1, 7]),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
        )
        self.attention3 = nn.Sequential(
            conv(in_ch, ch, [1, 3]),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            conv(ch, in_ch, 1),
            nn.LeakyReLU(negative_slope=0.3, inplace=True)
        )
    
    def forward(self, x):
        x = x.permute(0,2,1,3)
        out1 = self.attention1(x)
        out2 = self.attention2(out1)
        out3 = self.attention3(out2)
        out = torch.cat((out1*x, out2*x, out3*x), 2)
        return out.permute(0, 2, 1, 3)

class multiAttBlock(nn.Module):
    def __init__(self,in_ch, out_ch, stride=[1,1], padding=0):
        super(multiAttBlock, self).__init__()
        self.Att24x16 = Att24x16(in_ch[0], in_ch[0]*8) # 4
        self.Att4x16 = Att4x16(in_ch[1], in_ch[1]*8)     #24
        self.Att4x24 = Att4x24(in_ch[2], in_ch[2]*8)     #16
        self.conv = ConvBN(in_ch[0]*8, out_ch, 1)
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)
    
    # @torchsnooper.snoop()
    def forward(self, x):
        identity = self.identity(x)
        out24x16 = self.Att24x16(x)
        out4x16 = self.Att4x16(x)
        out4x24 = self.Att4x24(x)
        out = torch.cat((identity, out24x16, out4x16, out4x24),1)
        out = self.conv(out)
        out = self.relu(out)
        return out

# net = multiAttBlock([4, 24, 16], 16, [1,1], 1)
# encoder = nn.Sequential(
#             multiAttBlock([4, 24, 16], 16, [1,1], 1),
#             multiAttBlock([16, 24, 16], 64, [1,1], 1),
#             multiAttBlock([64, 24, 16], 128, [2,2], 0),
#             multiAttBlock([128, 11, 7], 256, [2,2], 0),
#         )
encoder = nn.Sequential(
            multiAttBlock([4, 24, 16], 16),
            nn.Conv2d(16, 16, 3, [1,1], 1, padding_mode='replicate'),
            multiAttBlock([16, 24, 16], 64),
            nn.Conv2d(64, 64, 3, [1,1], 1, padding_mode='replicate'),
            multiAttBlock([64, 24, 16], 128),
            nn.Conv2d(128, 128, 3, [2,2], 0, padding_mode='replicate'),
            multiAttBlock([128, 11, 7], 256),
            nn.Conv2d(256, 256, 3, [2,2], 0, padding_mode='replicate'),
        )
x = torch.rand(2, 4, 24, 16)
# net = multiAttBlock([64, 11, 7], 128, [2,2], 0)
# x1 = torch.rand(2, 64, 11, 7)
y = encoder(x)
pool = nn.AdaptiveAvgPool2d((1, 1))
y = pool(x)
print((x*y).size())
