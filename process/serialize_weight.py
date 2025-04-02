import argparse
import numpy as np
import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from model import create_model_functional
import numpy as np
import cv2


"""
Serialize a Keras model into a flattened NumPy array with the correct shape for use with PyTorch in Rust.

All the weights need to be flattened into a single array for rust interopt

"""

def serialize_weights(model):
    network_weights = np.array([])
    for i, layer in enumerate(model.layers):
        if "conv" in layer.name:
            A, b = layer.get_weights()
            # Keras stores the filter as the first two dimensions and the
            # channels as the 3rd and 4th. PyTorch does the opposite so flip
            # everything around
            _, _, _, inp_c, out_c = A.shape
            py_tensor = [[A[:, :, :, i, o] for i in range(inp_c)] for o in range(out_c)]
            A = np.array(py_tensor)
        elif "dense" in layer.name:
            A, b = layer.get_weights()
            A = A.T
            # Get the shape of last layer output to transform the FC
            inp_chans = 1
            for prev_i in range(i, 0, -1):
                layer_name = model.layers[prev_i].name
                if ("global" in layer_name):
                    inp_chans = model.layers[prev_i].output_shape[1]
                    break
                if ("conv" in layer_name) or ("max_pooling3d" in layer_name) or prev_i == 0:
                    inp_chans = model.layers[prev_i].output_shape[3]
                    break
            # Remap to PyTorch shape
            fc_h, fc_w = A.shape
            channel_cols = [np.hstack([A[:, [i]] for i in range(chan, fc_w, inp_chans)])
                            for chan in range(inp_chans)]
            A = np.hstack(channel_cols)
        else:
            continue
        layer_weights = np.concatenate((A.flatten(), b.flatten()))
        network_weights = np.concatenate((network_weights, layer_weights))

    np.save(os.path.join(f"model_new.npy"), network_weights.astype(np.float64))
    
    
if __name__ == "__main__":

     model = create_model_functional()
     serialize_weights(model)



   

    
   