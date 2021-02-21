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


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


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

class CR24x16(nn.Module):
    def __init__(self, in_ch, ch):
        super(CR24x16, self).__init__()
        self.ch = ch
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3', ConvBN(ch, ch, 3)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv1x7', ConvBN(ch, ch, [1, 7])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv7x1', ConvBN(ch, ch, [7, 1])),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5', ConvBN(ch, ch, [1, 5])),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x1', ConvBN(ch, ch, [5, 1])),
        ]))
        self.in_conv =  ConvBN(in_ch, ch, 1)
        self.out_conv = ConvBN(ch * 2, in_ch, 1)
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    # @torchsnooper.snoop()
    def forward(self, x):
        identity = self.identity(x)
        x = self.in_conv(x)
        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.out_conv(out)
        out = self.relu(out)
        out = out+identity
        return out

class CR4x24(nn.Module):
    def __init__(self, in_ch, ch):
        super(CR4x24, self).__init__()
        self.ch = ch
        self.path1 = nn.Sequential(OrderedDict([
            ('conv1x7', ConvBN(ch, ch, [1, 7])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv7x1', ConvBN(ch, ch, [3, 1])),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5', ConvBN(ch, ch, [3, 3])),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x1', ConvBN(ch, ch, [1, 5])),
        ]))
        self.in_conv = ConvBN(in_ch, ch, 1)
        self.out_conv = ConvBN(ch*2, in_ch, 1)
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        x = x.permute(0,3,1,2)
        identity = self.identity(x)
        x = self.in_conv(x)
        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.out_conv(out)
        out = self.relu(out)
        out = out+identity
        return out.permute(0, 2, 3, 1)

class CR4x16(nn.Module):
    def __init__(self, in_ch, ch):
        super(CR4x16, self).__init__()
        self.ch = ch
        self.path1 = nn.Sequential(OrderedDict([
            ('conv1x7', ConvBN(ch, ch, [1, 7])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv7x1', ConvBN(ch, ch, [3, 1])),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5', ConvBN(ch, ch, [3, 3])),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x1', ConvBN(ch, ch, [1, 5])),
        ]))
        self.conv_in = ConvBN(in_ch, ch, 1)
        self.conv_out = ConvBN(ch*2, in_ch, 1)
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)
    
    def forward(self, x):
        x = x.permute(0,2,1,3)
        identity = self.identity(x)
        x = self.conv_in(x)
        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.conv_out(out)
        out = self.relu(out)
        out = out+identity
        return out.permute(0, 2, 1, 3)

class CRBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CRBlock, self).__init__()
        self.cr24x16 = CR24x16(in_ch, 32)
        self.cr4x16 = CR4x16(24, 96)
        self.cr4x24 = CR4x24(16, 128)
        self.conv = ConvBN(in_ch*3, out_ch, 1)

    
    def forward(self, x):
        # out = self.cr24x16(x) + self.cr4x16(x) + self.cr4x24(x)
        out24x16 = self.cr24x16(x)
        out4x16 = self.cr4x16(x)
        out4x24 = self.cr4x24(x)
        out = torch.cat((out24x16, out4x16, out4x24),1)
        out = self.conv(out)
        return out


class Encoder(nn.Module):
    B = 4
    def __init__(self, feedback_bits, N = 64, quantization=True):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            CRBlock(4, 12),
            CRBlock(12, 12),
            CRBlock(12, 4),
        )
        self.encoder_out = nn.Sequential(
            nn.Linear(1536, 768),
            nn.Dropout(0.2),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            nn.Linear(768, int(feedback_bits / self.B)),
            nn.Dropout(0.2),
            nn.Sigmoid()
        )

        self.quantize = QuantizationLayer(self.B)
        self.quantization = quantization 

    # @torchsnooper.snoop()
    def forward(self, x):
        x = x.permute(0,3,1,2)
        r = torch.sqrt(x[:,0,:,:]**2+x[:,1,:,:]**2).unsqueeze(1)
        theta = torch.arctan(x[:,0,:,:]/x[:,1,:,:]).unsqueeze(1)
        x = torch.cat((x, r, theta), 1)
        out = self.encoder(x)
        out = out.reshape(-1, 1536)
        out = self.encoder_out(out)
        if self.quantization:
            out = self.quantize(out)
        else:
            out = out
        return out


class Decoder(nn.Module):
    B = 4

    def __init__(self, feedback_bits, N = 64, quantization=True):
        super(Decoder, self).__init__()
        self.feedback_bits = feedback_bits
        self.dequantize = DequantizationLayer(self.B)
        self.fc = nn.Linear(int(feedback_bits / self.B), 768)
        self.decoder_feature = nn.Sequential(
            CRBlock(2, 12),
            CRBlock(12, 12),
            CRBlock(12, 2),
        )
        self.decoder_in = nn.Sequential(
            nn.Linear(int(feedback_bits / self.B), 768),
            # nn.LeakyReLU(negative_slope=0.3, inplace=True),
            # nn.Linear(768, 1536)
        )
        self.conv_out = ConvBN(4, 2, 3)
        self.sig = nn.Sigmoid()
        self.quantization = quantization        

    # @torchsnooper.snoop()
    def forward(self, x):
        if self.quantization:
            out = self.dequantize(x)
        else:
            out = x
        out = out.contiguous().view(-1, int(self.feedback_bits / self.B))
        out = self.decoder_in(out)
        out = out.reshape(-1, 2, 24, 16)

        out = self.decoder_feature(out)
        # out = self.conv_out(out)
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
