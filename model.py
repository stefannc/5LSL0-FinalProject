# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:53:06 2019

@author: s143239
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.losses import MSE, categorical_crossentropy

def deep_cnn2d_mfcc(shape, num_classes, act = 'relu'):
    model = Sequential()
    # apply 32 convolution filters of size 3x3 each
    model.add(Conv2D(32, (3,3), padding="same", input_shape=shape, activation = act))
    model.add(Conv2D(32, (3,3), padding="same", activation = act))
    model.add(MaxPooling2D((2,2), strides = (2,2), padding = 'same'))
    
    model.add(Conv2D(32, (3,3), padding="same", activation = act))
    model.add(Conv2D(32, (3,3), padding="same", activation = act))
    model.add(MaxPooling2D((2,2), strides = (2,2), padding = 'same'))
    
    model.add(Conv2D(32, (3,3), padding="same", activation = act))
    model.add(Conv2D(32, (3,3), padding="same", activation = act))
    model.add(MaxPooling2D((2,2), strides = (2,2), padding = 'same'))
    
    model.add(Flatten())
    
    model.add(Dense(64, activation = act))
    
    model.summary()
    return model

def deep_cnn2d_spectrum(shape, num_classes, act = 'relu'):
    model = Sequential()
    # apply 32 convolution filters of size 3x3 each
    model.add(Conv2D(32, (3,3), padding="same", input_shape=shape, activation = act))
    model.add(Conv2D(32, (3,3), padding="same", activation = act))
    model.add(MaxPooling2D((2,2), strides = (2,2), padding = 'same'))
    
    model.add(Conv2D(32, (3,3), padding="same", activation = act))
    model.add(Conv2D(32, (3,3), padding="same", activation = act))
    model.add(MaxPooling2D((2,2), strides = (2,2), padding = 'same'))
    
    model.add(Conv2D(32, (3,3), padding="same", activation = act))
    model.add(Conv2D(32, (3,3), padding="same", activation = act))
    model.add(MaxPooling2D((2,2), strides = (2,2), padding = 'same'))
    
    model.add(Conv2D(32, (3,3), padding="same", activation = act))
    model.add(Conv2D(32, (3,3), padding="same", activation = act))
    model.add(MaxPooling2D((2,2), strides = (2,2), padding = 'same'))
    
    model.add(Conv2D(32, (3,3), padding="same", activation = act))
    model.add(Conv2D(32, (3,3), padding="same", activation = act))
    model.add(MaxPooling2D((2,2), strides = (2,2), padding = 'same'))
    
    model.add(Conv2D(32, (3,3), padding="same", activation = act))
    model.add(Conv2D(32, (3,3), padding="same", activation = act))
    model.add(MaxPooling2D((2,2), strides = (2,2), padding = 'same'))
    
    model.add(Conv2D(32, (3,3), padding="same", activation = act))
    model.add(Conv2D(32, (3,3), padding="same", activation = act))
    model.add(MaxPooling2D((2,2), strides = (2,2), padding = 'same'))
    
    
    model.add(Flatten())
    
    model.add(Dense(64, activation = act))
    
    model.summary()
    return model

def deep_cnn1d_speech(shape, num_classes, act = 'relu'):
    model = Sequential()
    # apply 32 convolution filters of size 3x3 each
    model.add(Conv1D(32, 3, padding="same", input_shape=shape, activation = act))
    model.add(Conv1D(32, 3, padding="same", activation = act))
    model.add(MaxPooling1D(2, strides = 2, padding = 'same'))
    
    model.add(Conv1D(32, 3, padding="same", activation = act))
    model.add(Conv1D(32, 3, padding="same", activation = act))
    model.add(MaxPooling1D(2, strides = 2, padding = 'same'))
    
    model.add(Conv1D(32, 3, padding="same", activation = act))
    model.add(Conv1D(32, 3, padding="same", activation = act))
    model.add(MaxPooling1D(2, strides = 2, padding = 'same'))
    
    model.add(Conv1D(32, 3, padding="same", activation = act))
    model.add(Conv1D(32, 3, padding="same", activation = act))
    model.add(MaxPooling1D(2, strides = 2, padding = 'same'))
    
    model.add(Conv1D(32, 3, padding="same", activation = act))
    model.add(Conv1D(32, 3, padding="same", activation = act))
    model.add(MaxPooling1D(2, strides = 2, padding = 'same'))

    model.add(Conv1D(32, 3, padding="same", activation = act))    
    model.add(Conv1D(32, 3, padding="same", activation = act))
    model.add(MaxPooling1D(2, strides = 2, padding = 'same'))
    
    model.add(Conv1D(32, 3, padding="same", activation = act))
    model.add(Conv1D(32, 3, padding="same", activation = act))
    model.add(MaxPooling1D(2, strides = 2, padding = 'same'))
    
    model.add(Conv1D(32, 3, padding="same", activation = act))    
    model.add(Conv1D(32, 3, padding="same", activation = act))
    model.add(MaxPooling1D(2, strides = 2, padding = 'same'))
    
    model.add(Conv1D(32, 3, padding="same", activation = act))
    model.add(Conv1D(32, 3, padding="same", activation = act))
    model.add(MaxPooling1D(2, strides = 2, padding = 'same'))
    
    model.add(Conv1D(32, 3, padding="same", activation = act))    
    model.add(Conv1D(32, 3, padding="same", activation = act))
    model.add(MaxPooling1D(2, strides = 2, padding = 'same'))
    
    model.add(Conv1D(32, 3, padding="same", activation = act))
    model.add(Conv1D(32, 3, padding="same", activation = act))
    model.add(MaxPooling1D(2, strides = 2, padding = 'same'))
    model.add(Flatten())
    
    model.add(Dense(64, activation = act))
    #model.add(Dropout())
    #model.add(Dense(num_classes))
    
    model.summary()
    return model


