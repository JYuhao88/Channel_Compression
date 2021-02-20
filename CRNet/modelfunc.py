
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import OrderedDict

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


class CRBlock_Dense(nn.Module):
    def __init__(self,ch):
        super(CRBlock_Dense, self).__init__()
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
    
    
class CR_encoder(nn.Module):
    def __init__(self,ch):
        super().__init__()
        self.ch = ch
        self.encoder1 = nn.Sequential(OrderedDict([
            ("conv1x7_bn", ConvBN(ch, ch, [1, 7])),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=False)),
            ("conv7x1_bn", ConvBN(ch, ch, [7, 1])),
        ]))
        self.encoder2 = ConvBN(ch,ch, 3)
        self.encoder3 = nn.Sequential(OrderedDict([
            ("conv1x5_bn", ConvBN(ch, ch, [1, 5])),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=False)),
            ("conv5x1_bn", ConvBN(ch, ch, [5, 1])),
        ]))
        self.encoder_conv = nn.Sequential(OrderedDict([
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=False)),
            ("conv1x1_bn", ConvBN(ch*3, ch, 1)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=False)),
        ]))
    def forward(self, x):
        encode1 = self.encoder1(x)
        encode2 = self.encoder2(x)
        encode3 = self.encoder3(x)
        out = torch.cat((encode1, encode2,encode3), dim=1)
        #out = self.encoder_conv(out)
        return out
    
class ResBlock(nn.Module):
    def __init__(self,ch,nblocks=1,shortcut=True):
        super().__init__()
        self.shortcut= shortcut
        self.nblocks = nblocks
        self.ch = ch
        self.module_list = nn.ModuleList()
        self.identity = nn.Identity()
        for i in range(nblocks):
            resblock = nn.ModuleList()
            resblock.append(ConvBN(ch,ch,1))
            resblock.append(nn.LeakyReLU(negative_slope=0.3, inplace=False))
            resblock.append(ConvBN(ch,ch,3))
            resblock.append(nn.LeakyReLU(negative_slope=0.3, inplace=False))
            resblock.append(ConvBN(ch,ch,1))
            resblock.append(nn.LeakyReLU(negative_slope=0.3, inplace=False))
            self.module_list.append(resblock)
    def forward(self,x):
        for module in self.module_list:
            h = x
            y =  self.identity(x)
            for res in module:
                h = res(h)
            x = y+h if self.shortcut else h 
        return x     
class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        x = x*(torch.tanh(torch.nn.functional.softplus(x)))
        return x 

    
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
    
class ResBlock_Attention(nn.Module):
    def __init__(self,ch,nblocks=1,shortcut=True):
        super().__init__()
        self.shortcut= shortcut
        self.nblocks = nblocks
        self.ch = ch
        self.module_list = nn.ModuleList()
        self.identity = nn.Identity()
        for i in range(nblocks):
            resblock = nn.ModuleList()
            resblock.append(ConvBN(ch,ch,1))
            resblock.append(nn.LeakyReLU(negative_slope=0.3, inplace=False))
            resblock.append(ConvBN(ch,ch,3))
            resblock.append(nn.LeakyReLU(negative_slope=0.3, inplace=False))
            resblock.append(Channel_Attention(ch,6)) ## 6 为中间压缩倍数
            resblock.append(Spartial_Attention(7))
            resblock.append(ConvBN(ch,ch,1))
            resblock.append(nn.LeakyReLU(negative_slope=0.3, inplace=False))
            self.module_list.append(resblock)
    def forward(self,x):
        for module in self.module_list:
            h = x
            y = self.identity(x)
            for res in module:
                h = res(h)
            x = y+h if self.shortcut else h 
        return x  
    
    
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
        return x * y    
    

    
class Self_Attention(nn.Module):
    def __init__(self,n_dims, width=24, height=16):
        super(Self_attention, self).__init__()

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.rel_h = nn.Parameter(torch.randn([1, n_dims, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, n_dims, width, 1]), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, C, -1)
        k = self.key(x).view(n_batch, C, -1)
        v = self.value(x).view(n_batch, C, -1)

        content_content = torch.bmm(q.permute(0, 2, 1), k)

        content_position = (self.rel_h + self.rel_w).view(1, C, -1).permute(0, 2, 1)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(n_batch, C, width, height)

        return out
    
  
    
class ResBlock_CRNET(nn.Module):
    def __init__(self,ch,nblocks=1,shortcut=True):
        super().__init__()
        self.shortcut= shortcut
        self.nblocks = nblocks
        self.ch = ch
        self.module_list = nn.ModuleList()
        self.identity = nn.Identity()
        for i in range(nblocks):
            resblock = nn.ModuleList()
            resblock.append(ConvBN(ch,ch,1))
            resblock.append(nn.LeakyReLU(negative_slope=0.3, inplace=False))
            resblock.append(CR_encoder(ch))
            resblock.append(nn.LeakyReLU(negative_slope=0.3, inplace=False))
            resblock.append(ConvBN(ch*3,ch,1))
            resblock.append(nn.LeakyReLU(negative_slope=0.3, inplace=False))
            self.module_list.append(resblock)
    def forward(self,x):
        for module in self.module_list:
            h = x
            y = self.identity(x)
            for res in module:
                h = res(h)
            x = y+h if self.shortcut else h 
        return x 

class Encoder(nn.Module):
    B = 4
    def __init__(self, feedback_bits, quantization=True):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(2, 32, 3)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("ResBlock_1", ResBlock(32,3)),
            ('bn1', nn.BatchNorm2d(32)),
            ("ResBlock_CRNET_2", ResBlock_CRNET(32,3)),
            ('bn2', nn.BatchNorm2d(32)),
            ("ResBlock_Attention_3", ResBlock_Attention(32,3)),
            ('bn3', nn.BatchNorm2d(32)),
            ("ResBlock_Attention_4", ResBlock_Attention(32,2)),
            ('bn4', nn.BatchNorm2d(32)),
        ]))
        self.encoder_conv = nn.Sequential(OrderedDict([
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x1_bn", ConvBN(32, 2, 1)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        
        self.fc = nn.Linear(768, int(feedback_bits / self.B))
        self.sig = nn.Sigmoid()
        self.quantize = QuantizationLayer(self.B)
        self.quantization = quantization 

    def forward(self, x):
        x = x.permute(0,3,1,2)
        encode = self.encoder(x)
        out = self.encoder_conv(encode)
        out = out.reshape(-1, 768)
        out = self.fc(out)
        out = self.sig(out)
        if self.quantization:
            out = self.quantize(out)
        else:
            out = out
        return out


class Decoder(nn.Module):
    B = 4

    def __init__(self, feedback_bits, quantization=True):
        super(Decoder, self).__init__()
        self.feedback_bits = feedback_bits
        self.dequantize = DequantizationLayer(self.B)
        self.fc = nn.Linear(int(feedback_bits / self.B), 768)
        
        self.bottleneck_1 = nn.Sequential(OrderedDict([
            ("relu1_1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x1_bn_1", ConvBN(64, 32, 1)),
            ("relu1_2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.bottleneck_2 = nn.Sequential(OrderedDict([
            ("relu2_1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x1_bn_2", ConvBN(96, 32, 1)),
            ("relu2_2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.bottleneck_3 = nn.Sequential(OrderedDict([
            ("relu3_1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x1_bn_3", ConvBN(128, 32, 1)),
            ("relu3_2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        
        self.CRBlock1 = CRBlock_Dense(32)
        self.CRBlock2 = CRBlock_Dense(64)
        self.CRBlock3 = CRBlock_Dense(96)
        self.CRBlock4 = CRBlock_Dense(128)
        
        decoder = OrderedDict([
            ("conv5x5_bn", ConvBN(2, 32, 1)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)), 
        ])
        self.decoder_feature = nn.Sequential(decoder)
        self.out_cov = ConvBN(32, 2, 1)
        self.sig = nn.Sigmoid()
        self.quantization = quantization        

    def forward(self, x):
        if self.quantization:
            out = self.dequantize(x)
        else:
            out = x
        out = out.contiguous().view(-1, int(self.feedback_bits / self.B))
        out = self.fc(out)
        out = out.reshape(-1, 2, 24, 16)
        
        out1 = self.decoder_feature(out)   ##2 ->32
        out2 = self.CRBlock1(out1)       #32 ->32 
        
        x2 = torch.cat((out1, out2), dim=1)   #32+32 ->64 
        out3 = self.CRBlock2(x2)           #64->64
        out3 = self.bottleneck_1(out3)            #64->32
        
        x3 = torch.cat((out1, out2,out3), dim=1)      #32+32+32->96
        out4 = self.CRBlock3(x3)                #96->96 
        out4 = self.bottleneck_2(out4)            #96->32
        
        x4 = torch.cat((out1, out2,out3,out4), dim=1) #32+32+32+32->128
        out5 = self.CRBlock4(x4)               #128->128 
        out5 = self.bottleneck_3(out5)           #128->32
        
        y = self.out_cov(out5)
        y = self.sig(y)
        y = y.permute(0, 2, 3, 1)
        return y



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
