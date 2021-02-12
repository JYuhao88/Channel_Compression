import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import module

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)

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
        return x * y


class AttentionRefineNet(nn.Module):
    def __init__(self, channel, exapand = 4):
        super(AttentionRefineNet, self).__init__()
        self.multiAtt = nn.ModuleList()

        for _ in range(3):
            self.multiAtt.append(nn.Sequential(
                conv3x3(channel, channel*exapand),
                nn.BatchNorm2d(channel*exapand),
                nn.LeakyReLU(0.3),
                Channel_Attention(channel*exapand, 2),
                conv3x3(channel*exapand, channel),
                nn.BatchNorm2d(channel)
                ))

    def forward(self, x):
        for i in range(3):
            residual = self.multiAtt[i](x)
            x = residual + x
        return x