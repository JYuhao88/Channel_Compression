#=======================================================================================================================
#=======================================================================================================================
import numpy as np
from modelDesign import *
import torch
import scipy.io as sio
#=======================================================================================================================
#=======================================================================================================================
# Parameters Setting
NUM_FEEDBACK_BITS = NUM_FEEDBACK_BITS #128
CHANNEL_SHAPE_DIM1 = 24
CHANNEL_SHAPE_DIM2 = 16
CHANNEL_SHAPE_DIM3 = 2
#=======================================================================================================================
#=======================================================================================================================
# Data Loading
mat = sio.loadmat('channelData/H_4T4R.mat')
data = mat['H_4T4R']
data = data.astype('float32')
data = np.reshape(data, (len(data), CHANNEL_SHAPE_DIM1, CHANNEL_SHAPE_DIM2, CHANNEL_SHAPE_DIM3))
H_test = data
# encOutput Loading
encode_feature = np.load('./encOutput.npy')
test_dataset = DatasetFolder(encode_feature)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

#=======================================================================================================================
#=======================================================================================================================
# Model Loading and Decoding
autoencoderModel = AutoEncoder(NUM_FEEDBACK_BITS).cuda()
model_decoder = autoencoderModel.decoder
model_decoder.load_state_dict(torch.load('./modelSubmit/decoder.pth.tar')['state_dict'])
print("weight loaded")
model_decoder.eval()
H_pre = []
with torch.no_grad():
    for i, decoderOutput in enumerate(test_loader):
        # convert numpy to Tensor
        decoderOutput = decoderOutput.cuda()
        output = model_decoder(decoderOutput)
        output = output.cpu().numpy()
        if i == 0:
            H_pre = output
        else:
            H_pre = np.concatenate((H_pre, output), axis=0)
if (NMSE(H_test, H_pre) < 0.1):
    print('Valid Submission')
    print('The Score is ' + np.str(1.0 - NMSE(H_test, H_pre)))
print('Finished!')
#=======================================================================================================================
#=======================================================================================================================










