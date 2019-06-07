"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : generator.py
                    Creates a generator for the Kaggle dataset
    Author        : Bart van Erp
    Date          : 06/06/2019

==============================================================================
"""

## import libraries
from scipy.io import wavfile
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
#fs, data = wavfile.read('./output/audio.wav')

class DataGenerator(keras.utils.Sequence):
    'Create data generator using the keras class'    
    
    def __init__(self, data_path, data_listing, batch_size, dims, labels):
        'Initializes generator'
        # Initialize generator
        self.batch_size = batch_size
        self.data_path = data_path
        self.dims = dims
        self.labels = labels
        self.n_classes = len(self.labels)
        self.trainset = [x[0:-1] for x in open(data_listing, "r")]
        self.num_lines = len(self.trainset)
        self.batches = int(np.floor(self.num_lines/self.batch_size))
        self.shuffle = np.random.permutation(self.num_lines)

        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_lines/self.batch_size))
        
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.shuffle[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y
    
    def __iter__(self):
        'Create a generator that iterate over the Sequence.'
        for item in (self[i] for i in range(len(self))):
            yield item
        
    def on_epoch_end(self):
        'Shuffle order of data per epoch'
        self.shuffle = np.random.permutation(self.num_lines)
 
        
    def __data_generation(self, items):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, self.dims[0]))
        y = np.empty((self.batch_size), dtype=int)
    
        # Generate data
        for i, ID in enumerate(items):
            # Store sample
            _, x_temp = wavfile.read(self.data_path+self.trainset[ID])
            X[i,] = np.pad(x_temp, (0, 16000-len(x_temp)), 'constant')
    
            # Store class
            y_temp = self.trainset[ID].split('/')[0]
            if y_temp in self.labels:
                y[i] = self.labels.index(y_temp)
            else:
                y[i] = len(self.labels)-1
    
        return  np.expand_dims(X,-1), keras.utils.to_categorical(y, num_classes=self.n_classes)
    
    
def generator2(data_path, data_listing, labels, batch_size):
    'Create data generator without keras class'
    
    trainset = [x[0:-1] for x in open(data_listing, "r")]

    X = np.zeros((batch_size, 16000))
    y = np.zeros((batch_size,1))
    while True:
        items = np.random.permutation(len(trainset))[0:batch_size]
        for i in range(batch_size):
            # choose random index in features
            _, x_temp = wavfile.read(data_path+trainset[items[i]])
            X[i,] = np.pad(x_temp, (0, 16000-len(x_temp)), 'constant')
            # Store class
            y_temp = trainset[items[i]].split('/')[0]
            if y_temp in labels:
                y[i] = labels.index(y_temp)
            else:
                y[i] = len(labels)-1
        yield np.expand_dims(X,-1), keras.utils.to_categorical(y, num_classes=len(labels))