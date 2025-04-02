import numpy as np
import cv2
import argparse
import numpy as np
import re
import os
import tensorflow as tf
import C3D_model
import torch
import torch.nn as nn


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

model = C3D_model.C3D(num_classes=101)
checkpoint = torch.load('C3D_config_0.5_irregular.pt', map_location=lambda storage, loc: storage)

#model.load_state_dict(checkpoint['state_dict'])
model.to(device)
model.eval()


network_weights = np.array([])
weights = []
bias = []
layer_weights = []
network_weights = []

for name, param in model.named_parameters():
    if "conv" in name and 'bias' in name:
        bias.append(param.detach().numpy())
        for i, b in enumerate(bias):
            #print(f'bias of layer {i+1}: {b}')
            continue
        print ('shape of b:', b.shape)
    if "conv" in name and 'weight' in name:
        #print (name)
        weights.append(param.detach().numpy())
        for i, w in enumerate(weights):
            #print(f'Weight of layer {i+1}: {w}')
            continue
        #print ('shape of w:', w.shape)
        inp_c, out_c, _, _, _ = w.shape
        py_tensor = [[w[i, o, :, :, :] for i in range(inp_c)] for o in range(out_c)]
        # print (py_tensor)
        A = np.array(py_tensor)
        #print ('shape of A:', A.shape)
    elif "fc" in name and 'weight' in name:
        print (name)
        weights.append(param.detach().numpy())
        for i, w in enumerate(weights):
            #print(f'Weight of layer {i+1}: {w}')
            continue
        print ('shape of w:', w.shape)
        w = w.T
        print ('shape of w after:', w.shape)
        inp_chans = 1
        for prev_i in range(i, 0, -1):
            # print ('prev_i',prev_i )
            layer_name = model.layers[prev_i].name
            # print ('i',i)
            # print ('layer_name', layer_name)
            if ("global" in name):
                inp_chans = model.layers[prev_i].output_shape[2]
                #print('inp_chans', inp_chans)
                break
            if ("conv" in name) or ("max_pooling3d" in name) or prev_i == 0:
                #print(name)
                print('model.layers[prev_i].output_shape',model.layers[prev_i].output_shape)
                #keras input channel [4]; pytroch input channel [2]
                inp_chans = model.layers[prev_i].output_shape[2]
                break
          Remap to PyTorch shape
          fc_h, fc_w = w.shape
          channel_cols = [np.hstack([w[:, [i]] for i in range(chan, fc_w, inp_chans)])
                          for chan in range(inp_chans)]
          w = np.hstack(channel_cols)
    
    else:
        continue
    layer_weights = np.concatenate((w.flatten(), b.flatten()))
    network_weights = np.concatenate((network_weights, layer_weights))
    
np.save(os.path.join(f"model_c3d_pytorch_0.5.npy"), network_weights.astype(np.float64))



    
    
    
