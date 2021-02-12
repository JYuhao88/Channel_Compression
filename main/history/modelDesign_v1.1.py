#=======================================================================================================================
#=======================================================================================================================
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
# from CBAM import *

NUM_FEEDBACK_BITS = 512 #pytorch版本一定要有这个参数

#=======================================================================================================================
#=======================================================================================================================
# Number to Bit Defining Function Defining
def Num2Bit(Num, B):
    Num_ = Num.type(torch.uint8)
    def integer2bit(integer, num_bits=B * 2):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) // 2 ** exponent_bits
        return (out - (out % 1)) % 2
    bit = integer2bit(Num_)
    bit = (bit[:, :, B:]).reshape(-1, Num_.shape[1] * B)
    return bit.type(torch.float32)
def Bit2Num(Bit, B):
    Bit_ = Bit.type(torch.float32)
    Bit_ = torch.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = torch.zeros(Bit_[:, :, 1].shape).cuda()
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return num
#=======================================================================================================================
#=======================================================================================================================
# Quantization and Dequantization Layers Defining
class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = torch.round(x * step - 0.5)
        out = Num2Bit(out, B)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of constant arguments to forward must be None.
        # Gradient of a number is the sum of its B bits.
        b, _ = grad_output.shape
        grad_num = torch.sum(grad_output.reshape(b, -1, ctx.constant), dim=2) / ctx.constant
        return grad_num, None
class Dequantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = Bit2Num(x, B)
        out = (out + 0.5) / step
        return out
    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # repeat the gradient of a Num for B time.
        b, c = grad_output.shape
        grad_output = grad_output.unsqueeze(2) / ctx.constant
        grad_bit = grad_output.expand(b, c, ctx.constant)
        return torch.reshape(grad_bit, (-1, c * ctx.constant)), None
class QuantizationLayer(nn.Module):
    def __init__(self, B):
        super(QuantizationLayer, self).__init__()
        self.B = B
    def forward(self, x):
        out = Quantization.apply(x, self.B)
        return out
class DequantizationLayer(nn.Module):
    def __init__(self, B):
        super(DequantizationLayer, self).__init__()
        self.B = B
    def forward(self, x):
        out = Dequantization.apply(x, self.B)
        return out
#=======================================================================================================================
#=======================================================================================================================
# Encoder and Decoder Class Defining
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
class Encoder(nn.Module):
    num_quan_bits = 4
    def __init__(self, feedback_bits):
        super(Encoder, self).__init__()
        self.conv1 = conv3x3(2, 2)
        self.conv2 = conv3x3(2, 2)
        self.fc = nn.Linear(768, int(feedback_bits / self.num_quan_bits))
        self.sig = nn.Sigmoid()
        self.quantize = QuantizationLayer(self.num_quan_bits)
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = out.contiguous().view(-1, 768) #需使用contiguous().view(),或者可修改为reshape
        out = self.fc(out)
        out = self.sig(out)
        out = self.quantize(out)
        return out

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
        return x*y

class AttentionRefineNet(nn.Module):
    def __init__(self, channel):
        super(AttentionRefineNet, self).__init__()
        self.multiAtt = nn.Sequential(
            conv3x3(channel, channel*8),
            nn.BatchNorm2d(channel*8),
            nn.LeakyReLU(0.3),
            Channel_Attention(channel*8, 1),
            conv3x3(channel*8, channel*4),
            nn.BatchNorm2d(channel*4),
            nn.LeakyReLU(0.3),
            Channel_Attention(channel*4, 1),
            conv3x3(channel*4, channel),
            nn.BatchNorm2d(channel)
        )
    def forward(self, x):
        residual = self.multiAtt(x)
        return x+residual

class Decoder(nn.Module):
    num_quan_bits = 4
    def __init__(self, feedback_bits):
        super(Decoder, self).__init__()
        self.feedback_bits = feedback_bits
        self.dequantize = DequantizationLayer(self.num_quan_bits)
        self.fc = nn.Linear(int(feedback_bits / self.num_quan_bits), 768)
        self.sig = nn.Sigmoid()
        # self.AttentionNet = nn.Sequential(
        #         AttentionRefineNet(2),
        #         nn.LeakyReLU(0.3),
        #         AttentionRefineNet(2),
        #         nn.LeakyReLU(0.3),
        #         AttentionRefineNet(2),
        #         conv3x3(2, 2),
        #         nn.BatchNorm2d(2),
        #         nn.Sigmoid())
        self.multiConvs = nn.ModuleList()
        for _ in range(3):
            self.multiConvs.append(nn.Sequential(
                AttentionRefineNet(2),
                conv3x3(2, 8),
                nn.ReLU(),
                conv3x3(8, 16),
                nn.ReLU(),
                conv3x3(16, 2),
                nn.ReLU()))
    def forward(self, x):
        out = self.dequantize(x)
        out = out.contiguous().view(-1, int(self.feedback_bits / self.num_quan_bits)) #需使用contiguous().view(),或者可修改为reshape
        out = self.sig(self.fc(out))
        out = out.contiguous().view(-1, 2, 24, 16) #需使用contiguous().view(),或者可修改为reshape
        for i in range(3):
            residual = out
            out = self.multiConvs[i](out)
            out = residual + out
        out = out.permute(0, 2, 3, 1)
        return out
class AutoEncoder(nn.Module):
    def __init__(self, feedback_bits):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(feedback_bits)
        self.decoder = Decoder(feedback_bits)
    def forward(self, x):
        feature = self.encoder(x)
        out = self.decoder(feature)
        return out
#=======================================================================================================================
#=======================================================================================================================
# NMSE Function Defining

def NMSE(x, x_hat):
    x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
    x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
    x_C = x_real - 0.5 + 1j * (x_imag - 0.5)
    x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
    power = np.sum(abs(x_C) ** 2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse

def NMSE_cuda(x, x_hat):
    x_real = x[:, :, :, 0].view(len(x),-1) - 0.5
    x_imag = x[:, :, :, 1].view(len(x),-1) - 0.5
    x_hat_real = x_hat[:, :, :, 0].view(len(x_hat), -1) - 0.5
    x_hat_imag = x_hat[:, :, :, 1].view(len(x_hat), -1) - 0.5
    power = torch.sum(x_real**2 + x_imag**2, axis=1)
    mse = torch.sum((x_real-x_hat_real)**2 + (x_imag-x_hat_imag)**2, axis=1)
    nmse = mse/power
    return nmse
    
class NMSELoss(nn.Module):
    def __init__(self, reduction='sum'):
        super(NMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, x_hat, x):
        nmse = NMSE_cuda(x, x_hat)
        if self.reduction == 'mean':
            nmse = torch.mean(nmse) 
        else:
            nmse = torch.sum(nmse)
        return nmse

def Score(NMSE):
    score = 1 - NMSE
    return score


# dataLoader
class DatasetFolder(Dataset):

    def __init__(self, matData):
        self.matdata = matData

    def __len__(self):
        return self.matdata.shape[0]
    
    def __getitem__(self, index):
        return self.matdata[index] #, self.matdata[index]



