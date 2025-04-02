import os
import zipfile
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D, ZeroPadding3D


def create_model_sequential():

    model = Sequential()
    input_shape = (16, 112, 112, 3)

    model.add(Conv3D(64, (3, 3, 3), activation='relu',
                     padding='same', name='conv1',
                     input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           padding='valid', name='pool1'))
    # 2nd layer group
    model.add(Conv3D(128, (3, 3, 3), activation='relu',
                     padding='same', name='conv2'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool2'))
    # 3rd layer group
    model.add(Conv3D(256, (3, 3, 3), activation='relu',
                     padding='same', name='conv3a'))
    model.add(Conv3D(256, (3, 3, 3), activation='relu',
                     padding='same', name='conv3b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool3'))
    # 4th layer group
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                     padding='same', name='conv4a'))
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                     padding='same', name='conv4b'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool4'))
    # 5th layer group
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                     padding='same', name='conv5a'))
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                     padding='same', name='conv5b'))
    model.add(ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)), name='zeropad5'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool5'))
    model.add(Flatten())
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5))
    model.add(Dense(487, activation='softmax', name='fc8'))

    return model


def create_model_functional():
  
    inputs = Input(shape=(112,112,16,3))

    conv1 = Conv3D(64, (3, 3, 3), activation='relu',
                   padding='same', name='conv1')(inputs)
    pool1 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                         padding='valid', name='pool1')(conv1)

    conv2 = Conv3D(128, (3, 3, 3), activation='relu',
                   padding='same', name='conv2')(pool1)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding='valid', name='pool2')(conv2)

    conv3a = Conv3D(256, (3, 3, 3), activation='relu',
                    padding='same', name='conv3a')(pool2)
    conv3b = Conv3D(256, (3, 3, 3), activation='relu',
                    padding='same', name='conv3b')(conv3a)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding='valid', name='pool3')(conv3b)

    conv4a = Conv3D(512, (3, 3, 3), activation='relu',
                    padding='same', name='conv4a')(pool3)
    conv4b = Conv3D(512, (3, 3, 3), activation='relu',
                    padding='same', name='conv4b')(conv4a)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding='valid', name='pool4')(conv4b)

    conv5a = Conv3D(512, (3, 3, 3), activation='relu',
                    padding='same', name='conv5a')(pool4)
    conv5b = Conv3D(512, (3, 3, 3), activation='relu',
                    padding='same', name='conv5b')(conv5a)
    zeropad5 = ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)),
                             name='zeropad5')(conv5b)
    pool5 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding='valid', name='pool5')(zeropad5)

    flattened = Flatten()(pool5)
    fc6 = Dense(4096, activation='relu', name='fc6')(flattened)
    dropout1 = Dropout(rate=0.5)(fc6)

    fc7 = Dense(4096, activation='relu', name='fc7')(dropout1)
    dropout2 = Dropout(rate=0.5)(fc7)

    predictions = Dense(487, activation='softmax', name='fc8')(dropout2)

    return Model(inputs=inputs, outputs=predictions)


def get_model(width=128, height=128, depth=64):
    """Build a C3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu",padding='same', name='conv1')(inputs)
    x = layers.MaxPool3D(pool_size=(1,2,2), strides=(1,2,2), padding='same', name='pool1')(x)
    #x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu",padding='same', name='conv2')(x)
    x = layers.MaxPool3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool2')(x)
    #x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu",padding='same', name='conv3a')(x)
    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu",padding='same', name='conv3b')(x)
    x = layers.MaxPool3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool3')(x)
    

    x = layers.Conv3D(filters=512, kernel_size=3, activation="relu", padding='same', name='conv4a')(x)
    x = layers.Conv3D(filters=512, kernel_size=3, activation="relu", padding='same', name='conv4b')(x)
    x = layers.MaxPool3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool4')(x)
   
    
    x = layers.Conv3D(filters=512, kernel_size=3, activation="relu", padding='same', name='conv5a')(x)
    x = layers.Conv3D(filters=512, kernel_size=3, activation="relu", padding='same', name='conv5b')(x)
    x = layers.ZeroPadding3D(padding=(0,1,1))(x)
    x = layers.MaxPool3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool5')(x)
    
    
    outputs = layers.Dense(4096, activation='relu', name='fc6')(x)
    outputs = layers.Dropout(0.5)(outputs)
    outputs = layers.Dense(4096, activation='relu', name='fc7')(outputs)
    outputs = layers.Dropout(0.5)(outputs)
    outputs = layers.Dense(487, activation='softmax', name='fc8')(outputs)

    # Define the model.
    model = keras.Model(inputs, outputs, name="c3d")
    return model


def create_features_exctractor(C3D_model, layer_name='fc6'):
    extractor = Model(inputs=C3D_model.input,
                      outputs=C3D_model.get_layer(layer_name).output)
    return extractor
