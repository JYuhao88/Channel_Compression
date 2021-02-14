import scipy.io as sio
import numpy as np
import h5py
import os
import random
import argparse

import torch
from torch import double, optim, var
import torch.nn as nn

from modelDesign import *

def set_quantization(model, is_quantization = True):
    try:
        model.encoder.quantization = is_quantization
        model.decoder.quantization = is_quantization
    except:
        model.module.encoder.quantization = is_quantization
        model.module.decoder.quantization = is_quantization
    return model

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def save_model(model, model_save_address='./modelSubmit'):
    # model save
    # save encoder
    modelSave1 = model_save_address + '/encoder.pth.tar'
    try:
        torch.save({'state_dict': model.encoder.state_dict(), }, modelSave1)
    except:
        torch.save({'state_dict': model.module.encoder.state_dict(), }, modelSave1)
    # save decoder
    modelSave2 = model_save_address + '/decoder.pth.tar'
    try:
        torch.save({'state_dict': model.decoder.state_dict(), }, modelSave2)
    except:
        torch.save({'state_dict': model.module.decoder.state_dict(), }, modelSave2)
    print('Model saved!')

if __name__ == "__main__":
    # Parameters for training
    parser = argparse.ArgumentParser()
    parser.add_argument("--continue_training", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--print_freq", type=int, default=500)
    parser.add_argument("--train_test_ratio", type=float, default=0.8)
    parser.add_argument("--feedback_bits", type=int, default=512)
    parser.add_argument("--is_quantization", type=bool, default=True)
    parser.add_argument("--data_load_address", type=str, default='./channelData')
    parser.add_argument("--model_save_address", type=str, default='./modelSubmit')
    parser.add_argument("--gpu_list", type=str, default='0')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list

    SEED = 42
    seed_everything(SEED)

    learning_rate = args.learning_rate
    num_workers = 4

    # parameters for data
    feedback_bits = args.feedback_bits
    img_height = 24
    img_width = 16
    img_channels = 2

    # Model construction
    model = AutoEncoder(feedback_bits)
    model = set_quantization(model, False)

    if args.continue_training:
        model = set_quantization(model, True)
        learning_rate = 5e-4
    
    if len(args.gpu_list.split(',')) > 1:
        model = torch.nn.DataParallel(model).cuda()  # model.module
    else:
        model = model.cuda()

    criterion = NMSELoss(reduction='mean') #nn.MSELoss()
    criterion_test = NMSELoss(reduction='sum')

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5, last_epoch=-1)

    # Data loading
    mat = sio.loadmat(args.data_load_address + '/H_4T4R.mat')
    data = mat['H_4T4R']  # shape=(320000, 1024)
    data = data.astype('float32')
    data = np.reshape(data, [len(data), img_height, img_width, img_channels])
    # split data for training(80%) and validation(20%)
    np.random.shuffle(data)
    start = int(data.shape[0] * args.train_test_ratio)
    x_train, x_test = data[:start], data[start:]

    # dataLoader for training
    train_dataset = DatasetFolder(x_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

    # dataLoader for training
    test_dataset = DatasetFolder(x_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    best_loss = 100
    for epoch in range(args.epochs):
        print('========================')
        print('lr:%.4e'%optimizer.param_groups[0]['lr']) 
        # model training
        model.train()
        if epoch < args.epochs/10 or best_loss>0.095:
            model = set_quantization(model, False)
        else:
            model = set_quantization(model, True)
            
        if epoch == args.epochs//4 * 3:
            optimizer.param_groups[0]['lr'] =  optimizer.param_groups[0]['lr'] * 0.25
        
        for i, input in enumerate(train_loader):
            input = input.cuda()
            output = model(input)
            
            loss = criterion(input, output)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss:.4f}\t'.format(
                    epoch, i, len(train_loader), loss=loss.item()))
        model.eval()
        model = set_quantization(model, True)

        total_loss = 0
        with torch.no_grad():
            for i, input in enumerate(test_loader):
                # convert numpy to Tensor
                input = input.cuda()
                output = model(input)
                total_loss += criterion_test(input, output).item()
            average_loss = total_loss / len(test_dataset)
            print('NMSE %.4f'%average_loss)
            if average_loss < best_loss:
                save_model(model, args.model_save_address)
                best_loss = average_loss

    del model, optimizer, train_loader,test_loader
    torch.cuda.empty_cache()