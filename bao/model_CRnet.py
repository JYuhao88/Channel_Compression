import os 
import scipy.io as sio 
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import OrderedDict
feedback_bits = 768
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
        identity = self.identity(x)
        y1 = self.__avg_pool(x)
        y1 = self.__fc(y1)
        y2 = self.__max_pool(x)
        y2 = self.__fc(y2)
        y = self.__sigmoid(y1+y2)
        return x * y+identity
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
        identity = self.identity(x)
        avg_mask = torch.mean(x, dim=1, keepdim=True)
        max_mask, _ = torch.max(x, dim=1, keepdim=True)
        mask = torch.cat([avg_mask, max_mask], dim=1)

        mask = self.__layer(mask)
        return x * mask+identity
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
        identity = self.identity(x)   ## 残差结构
        y = self.__avg_pool(x)
        y = self.__fc(y)
        return x * y+identity 
    
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


class CRBlock(nn.Module):
    def __init__(self):
        super(CRBlock, self).__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3', ConvBN(24, 24, 3)),
            ('Channel_Attention_1',Channel_Attention(24,24)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv1x9', ConvBN(24, 24, [1, 9])),
            ('Channel_Attention_2',Channel_Attention(24,24)),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv9x1', ConvBN(24, 24, [9, 1])),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5', ConvBN(24, 24, [1, 5])),
            ('Channel_Attention_',Channel_Attention(24,24)),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x1', ConvBN(24, 24, [5, 1])),
        ]))
        self.conv1x1 = ConvBN(24 * 2, 24, 1)
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        identity = self.identity(x)

        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.conv1x1(out)

        out = self.relu(out + identity)
        return out


class Encoder(nn.Module):
    B = 4

    def __init__(self, feedback_bits, quantization=True):
        super(Encoder, self).__init__()
        self.encoder1 = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(2, 24, 3)),
            ('Channel_Attention_1',Channel_Attention(24,24)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x9_bn", ConvBN(24, 24, [1, 9])),
            ('Channel_Attention_2',Channel_Attention(24,24)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv9x1_bn", ConvBN(24, 24, [9, 1])),
        ]))
        self.encoder2 = ConvBN(2, 24, 3)
        self.encoder_conv = nn.Sequential(OrderedDict([
            ('Channel_Attention_1',Channel_Attention(24*2,24*2)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x1_bn_1", ConvBN(24*2, 24, 1)),
            ('Channel_Attention_2',Channel_Attention(24,24)),
            ("conv1x1_bn_2", ConvBN(24, 2, 1)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))

        self.fc = nn.Linear(768, int(feedback_bits / self.B))
        self.sig = nn.Sigmoid()
        self.quantize = QuantizationLayer(self.B)
        self.quantization = quantization 

    def forward(self, x):
        x = x.permute(0,3,1,2)
        encode1 = self.encoder1(x)
        encode2 = self.encoder2(x)
        out = torch.cat((encode1, encode2), dim=1)
        out = self.encoder_conv(out)
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
        decoder = OrderedDict([
            ("conv5x5_bn", ConvBN(2, 24, 5)),
            ('Channel_Attention',Channel_Attention(24,24)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),           
            ("CRBlock1", CRBlock()),
            ("CRBlock2", CRBlock()),
            #("CRBlock3", CRBlock()),
        ])
        self.decoder_feature = nn.Sequential(decoder)
        self.out_cov = conv3x3(24, 2)
        self.sig = nn.Sigmoid()
        self.quantization = quantization        

    def forward(self, x):
        if self.quantization:
            out = self.dequantize(x)
        else:
            out = x
        out = out.view(-1, int(self.feedback_bits / self.B))
        out = self.fc(out)
        out = out.view(-1, 2, 24, 16)
        out = self.decoder_feature(out)
        out = self.out_cov(out)
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


# In[ ]:

import numpy as np
import h5py
import torch
import os
import torch.nn as nn
import random

gpu_list = '0,1'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
if __name__ == '__main__':    
    SEED = 89
    seed_everything(SEED) 

    batch_size = 1024
    epochs = 50
    learning_rate = 2e-3 # bigger to train faster
    num_workers = 4
    print_freq = 50  
    train_test_ratio = 0.8
    # parameters for data
    feedback_bits = 768
    img_height = 16
    img_width = 24
    img_channels = 2


    # Model construction
    model = AutoEncoder(feedback_bits)

    model.encoder.quantization = False
    model.decoder.quantization = False

    if len(gpu_list.split(',')) > 1:
        model = torch.nn.DataParallel(model).cuda()  # model.module
    else:
        model = model.cuda()

    criterion = NMSELoss(reduction='mean') #nn.MSELoss()
    criterion_test = NMSELoss(reduction='sum')

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=5,eta_min=1e-4,last_epoch=-1)
    # Data loading
    mat = sio.loadmat('H_4T4R.mat')
    data = mat['H_4T4R']
    data = data.astype('float32')
    data = np.reshape(data,(len(data),img_width,img_height,img_channels))
    np.random.shuffle(data)

    split = int(data.shape[0] * 0.8)
    data_train, data_test = data[:split], data[split:]
    train_dataset = DatasetFolder(data_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dataset = DatasetFolder(data_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    best_loss = 0.2
    for epoch in range(epochs):
        print('========================')
        print('lr:%.4e'%optimizer.param_groups[0]['lr']) 
        # model training
        model.train()
        if epoch < epochs//10:
            try:
                model.encoder.quantization = False
                model.decoder.quantization = False
            except:
                model.module.encoder.quantization = False
                model.module.decoder.quantization = False
        else:
            try:
                model.encoder.quantization = True
                model.decoder.quantization = True
            except:
                model.module.encoder.quantization = True
                model.module.decoder.quantization = True

        #if epoch == epochs//4 * 2:
        #    optimizer.param_groups[0]['lr'] =  optimizer.param_groups[0]['lr'] * 0.25
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10,eta_min=2e-4,last_epoch=-1)
        for i, input in enumerate(train_loader):

            input = input.cuda()
            output = model(input)
            
            loss = criterion(input,output)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss:.4f}\t'.format(
                    epoch, i, len(train_loader), loss=loss.item()))
        model.eval()
        try:
            model.encoder.quantization = True
            model.decoder.quantization = True
        except:
            model.module.encoder.quantization = True
            model.module.decoder.quantization = True
        total_loss = 0
        with torch.no_grad():
            for i, input in enumerate(test_loader):
                # convert numpy to Tensor
                input = input.cuda()
                output = model(input)
                total_loss += criterion_test(input,output).item()
            average_loss = total_loss / len(test_dataset)
            print('NMSE %.4f'%average_loss)
            if average_loss < best_loss:
                # model save
                # save encoder
                modelSave1 = 'Modelsave/encoder.pth.tar'
                try:
                    torch.save({'state_dict': model.encoder.state_dict(), }, modelSave1)
                except:
                    torch.save({'state_dict': model.module.encoder.state_dict(), }, modelSave1)
                # save decoder
                modelSave2 = 'Modelsave/decoder.pth.tar'
                try:
                    torch.save({'state_dict': model.decoder.state_dict(), }, modelSave2)
                except:
                    torch.save({'state_dict': model.module.decoder.state_dict(), }, modelSave2)
                print('Model saved!')
                best_loss = average_loss
                
    del model, optimizer, train_loader,test_loader
    torch.cuda.empty_cache()





