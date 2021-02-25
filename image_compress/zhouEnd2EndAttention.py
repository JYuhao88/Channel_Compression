'''
final
'''

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import OrderedDict
from layers import *
import torchsnooper


NUM_FEEDBACK_BITS = 512
# This part implement the quantization and dequantization operations.
# The output of the encoder must be the bitstream.
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
        # Gradient of a number is the sum of its four bits.
        b, _ = grad_output.shape
        grad_num = torch.sum(grad_output.reshape(b, -1, ctx.constant), dim=2)
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
        # repeat the gradient of a Num for four time.
        #b, c = grad_output.shape
        #grad_bit = grad_output.repeat(1, 1, ctx.constant) 
        #return torch.reshape(grad_bit, (-1, c * ctx.constant)), None
        grad_bit = grad_output.repeat_interleave(ctx.constant, dim=1)
        return grad_bit, None


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


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes)),
            # ('gdn', GDN(out_planes, inverse=gdnInverse))
        ]))

class RB(nn.Module):
    """Simple residual unit."""
    def __init__(self, in_plane, conv_plane):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBN(in_plane, conv_plane, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            ConvBN(conv_plane, in_plane, kernel_size=3),
        )
        self.bn = nn.BatchNorm2d(in_plane)
    def forward(self, x):
        identity = x
        out = self.conv(x)
        out += identity
        return self.bn(out)

class RNAB(nn.Module):
    def __init__(self, in_plane, conv_plane):
        super(RNAB, self).__init__()
        self.conv_a = nn.Sequential(
            RB(in_plane, conv_plane),
            # RB(in_plane, conv_plane),
            # RB(in_plane, conv_plane)
            )

        self.conv_b = nn.Sequential(
            # RB(in_plane, conv_plane),
            RB(in_plane, conv_plane),
            ConvBN(in_plane, conv_plane, kernel_size=[1, 5]),
            # RB(conv_plane, conv_plane),
            # RB(conv_plane, conv_plane),
            # conv3x3(conv_plane, conv_plane),
            # RB(conv_plane, conv_plane),
            RB(conv_plane, conv_plane),
            ConvBN(conv_plane, in_plane, kernel_size=[5, 1])
        )
        self.bn = nn.BatchNorm2d(in_plane)
    # @torchsnooper.snoop()
    def forward(self, x):
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return self.bn(out)


class Encoder(nn.Module):
    B = 4
    N = 128
    def __init__(self, feedback_bits, quantization=True):
        super(Encoder, self).__init__()
        N = self.N
        self.layer1 = nn.Sequential(
            ConvBN(2, N, kernel_size=3),
            ConvBN(N, N//2, 3)
            # GDN(N//2)
            )

        self.conv1 = ConvBN(N//2, N//8, kernel_size=[1, 7])
        self.layer2 = nn.Sequential(
            ConvBN(N//2, N//2, kernel_size=3),
            # GDN(N//2)
            )
        self.conv2 = ConvBN(N//2, N//4, kernel_size=[7, 1])
        self.layer3 = nn.Sequential(
            RNAB(N//2, N),
            ConvBN(N//2, N//2, kernel_size=3),
            # GDN(N//2)
            )
        self.conv3 = ConvBN(N//2, N//2, kernel_size=[5, 5])
        self.layer4 = nn.Sequential(
            RNAB(N//2, N),
            ConvBN(N//2, N//2, kernel_size=3),
            # GDN(N//2)
            )
        self.conv4 = ConvBN(N*11//8, 2, kernel_size=3)
        # self.__avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(768, int(feedback_bits / self.B))
        self.sig = nn.Sigmoid()
        self.quantize = QuantizationLayer(self.B)
        self.quantization = quantization 

    # @torchsnooper.snoop()
    def forward(self, x):
        x = x.permute(0,3,1,2)

        out1 = self.layer1(x)
        
        out2 = self.layer2(out1)
        out1 = self.conv1(out1)
        out3 = self.layer3(out2)
        out2 = self.conv2(out2)
        out4 = self.layer4(out3)
        out3 = self.conv3(out3)
        out = torch.cat((out1, out2, out3, out4), 1)
        out = self.conv4(out)
        # out = self.__avg_pool(out)

        out = out.reshape(-1, 768)
        out = self.fc(out)
        out = self.sig(out)
        if self.quantization:
            out = self.quantize(out)
        else:
            out = out
        return out

class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1,groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))


class CRBlock(nn.Module):
    def __init__(self,ch):
        super(CRBlock, self).__init__()
        self.ch = ch
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3', ConvBN(ch, ch, 3)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=False)),
            ('conv1x7', ConvBN(ch, ch, [1, 7])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=False)),
            ('conv7x1', ConvBN(ch, ch, [7, 1])),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5', ConvBN(ch, ch, [1, 5])),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=False)),
            ('conv5x1', ConvBN(ch, ch, [5, 1])),
        ]))
        self.conv1x1 = ConvBN(ch * 2, ch, 1)
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=False)

    def forward(self, x):
        identity = self.identity(x)
        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.conv1x1(out)
        out = self.relu(out)
        out = out+identity
        return out

class Decoder(nn.Module):
    B = 4
    N = 128
    def __init__(self, feedback_bits, quantization=True):
        super(Decoder, self).__init__()
        N = self.N
        self.feedback_bits = feedback_bits
        self.dequantize = DequantizationLayer(self.B)
        self.fc = nn.Linear(int(feedback_bits / self.B), 768)
        # self.decoder_feature = nn.Sequential(
        #     RB(2, N),
        #     RB(2, N),
        #     ConvBN(2, N//2, gdnInverse=True),
        #     # GDN(N//2, inverse=True),
        #     ConvBN(N//2, N//2, gdnInverse=True),
        #     RNAB(N//2, N),
        #     # GDN(N//2, inverse=True),
        #     ConvBN(N//2, N//2, gdnInverse=True),
        #     # GDN(N//2, inverse=True),
        #     ConvBN(N//2, 2, gdnInverse=True),
        # )
        self.decoder_feature = nn.Sequential(
            # RB(2, N),
            ConvBN(2, N//2, kernel_size=3), # 2 ->N
            CRBlock(N//2),
            RNAB(N//2, N),
            CRBlock(N//2),
            ConvBN(N//2, N, kernel_size=3), # N/2 ->N
            CRBlock(N),
            RNAB(N, N),
            CRBlock(N),
            ConvBN(N, N//2, kernel_size=3), # N ->N/2
            CRBlock(N//2),
            RNAB(N//2, N),
            CRBlock(N//2),
            ConvBN(N//2, 2, kernel_size=3) # N/2 ->2
        )
        self.sig = nn.Sigmoid()
        self.quantization = quantization        

    # @torchsnooper.snoop()
    def forward(self, x):
        if self.quantization:
            out = self.dequantize(x)
        else:
            out = x
        out = out.contiguous().view(-1, int(self.feedback_bits / self.B)) #需使用contiguous().view(),或者可修改为reshape
        out = self.sig(self.fc(out))
        out = out.reshape(-1, 2, 24, 16)
        out = self.decoder_feature(out)
        out = self.sig(out)
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

    def forward(self, x, x_hat):
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
