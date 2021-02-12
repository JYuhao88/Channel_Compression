import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import module

class SE_module(nn.Module):

    def __init__(self, channel, r):
        super(SE_module, self).__init__()

        self.__avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.__fc = nn.Sequential(
            nn.Conv2d(channel, channel//r, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(channel//r, channel, 1, bias=False),
            nn.Sigmoid(),
        )


    def forward(self, x):
        y = self.__avg_pool(x)
        y = self.__fc(y)
        return y

class Channel_Attention(nn.Module):

    def __init__(self, channel, r):
        super(Channel_Attention, self).__init__()

        self.__avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.__max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.__fc = nn.Sequential(
            nn.Conv2d(channel, channel//r, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(channel//r, channel, 1, bias=False),
        )
        self.__sigmoid = nn.Sigmoid()


    def forward(self, x):
        y1 = self.__avg_pool(x)
        y1 = self.__fc(y1)

        y2 = self.__max_pool(x)
        y2 = self.__fc(y2)

        y = self.__sigmoid(y1+y2)
        return y

class Spartial_Attention(nn.Module):

    def __init__(self, kernel_size):
        super(Spartial_Attention, self).__init__()

        assert kernel_size % 2 == 1, "kernel_size = {}".format(kernel_size)
        padding = (kernel_size - 1) // 2

        self.__layer = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid(),
        )

    
    def forward(self, x):
        avg_mask = torch.mean(x, dim=1, keepdim=True)
        max_mask, _ = torch.max(x, dim=1, keepdim=True)
        mask = torch.cat([avg_mask, max_mask], dim=1)

        mask = self.__layer(mask)
        return x * mask

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)

class AttentionRefineNet(nn.Module):
    def __init__(self, channel):
        super(AttentionRefineNet, self).__init__()
        self.Attention1 = Channel_Attention(channel*8, 1)
        self.Attention2 = Channel_Attention(channel*4, 1)
        self.conv1 = nn.Sequential(conv3x3(channel*8, channel*4))
        self.conv2 = nn.Sequential(conv3x3(channel*4, channel))
        self.layer1 = nn.Sequential(
            conv3x3(channel, channel*8),
            nn.BatchNorm2d(channel*8),
            nn.LeakyReLU(0.3),
        )
        self.layer2 = nn.Sequential(
            conv3x3(channel*8, channel*8),
            nn.BatchNorm2d(channel*8),
            nn.LeakyReLU(0.3),
        )
        self.layer3 = nn.Sequential(
            conv3x3(channel*4, channel*4),
            nn.BatchNorm2d(channel*4)
        )

    def forward(self, x):
        residual = self.layer1(x)

        att1 = self.Attention1(residual)
        residual = self.layer2(residual)
        residual = residual * att1
        residual = self.conv1(residual)

        att2 = self.Attention2(residual)
        residual = self.layer3(residual)
        residual = residual * att2
        residual = self.conv2(residual)

        return x+residual

