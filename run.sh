#!/bin/bash

python modelTrain.py \
--continue_training=True \
--batch_size=256 \
--epochs=100 \
--learning_rate=2e-3 \
--print_freq=500 \
--train_test_ratio=0.8  \
--feedback_bits=512 \
--is_quantization=True \
--data_load_address='./channelData' \
--model_save_address='./modelSubmit' \
--gpu_list='0'