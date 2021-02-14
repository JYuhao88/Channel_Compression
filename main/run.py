import subprocess
import os
args = ['python', 'modelTrain.py']
args.append('--continue_training=False')
args.append('--feedback_bits=512')
args.append('--batch_size=256')
args.append('--epochs=100')
args.append('--learning_rate=2e-3')
args.append('--print_freq=500')
args.append('--train_test_ratio=0.8')
args.append('--is_quantization=True')
args.append('--data_load_address=\'./channelData\'')
args.append('--model_save_address=\'./modelSubmit\'')
args.append('--gpu_list=\'0\'')

os.system(" ".join(args))

