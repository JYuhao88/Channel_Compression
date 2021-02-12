#=======================================================================================================================
#=======================================================================================================================
import numpy as np
from modelDesign import *
import torch
import scipy.io as sio
#=======================================================================================================================
#=======================================================================================================================
# Parameters Setting
NUM_FEEDBACK_BITS = 512 #128
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
test_dataset = DatasetFolder(H_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
#=======================================================================================================================
#=======================================================================================================================
# Model Loading
autoencoderModel = AutoEncoder(NUM_FEEDBACK_BITS).cuda()
model_encoder = autoencoderModel.encoder
model_encoder.load_state_dict(torch.load('./modelSubmit/encoder.pth.tar')['state_dict'])
print("weight loaded")
#=======================================================================================================================
#=======================================================================================================================
# Encoding
model_encoder.eval()
encode_feature = []
with torch.no_grad():
    for i, autoencoderInput in enumerate(test_loader):
        autoencoderInput = autoencoderInput.cuda()
        autoencoderOutput = model_encoder(autoencoderInput)
        autoencoderOutput = autoencoderOutput.cpu().numpy()
        if i == 0:
            encode_feature = autoencoderOutput
        else:
            encode_feature = np.concatenate((encode_feature, autoencoderOutput), axis=0)
            
print("feedbackbits length is ", np.shape(encode_feature)[-1])
np.save('./encOutput.npy', encode_feature)
print('Finished!')
#=======================================================================================================================
#=======================================================================================================================
