# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:14:26 2019

@author: s143239
"""
## import libraries
from scipy.io import wavfile
import os
import numpy as np
import tensorflow as tf
import keras
from scipy import signal
import random

class DataGenerator(keras.utils.Sequence):
    'Create data generator using the keras class'    
    
    def __init__(self, data_path, data_listing, batch_size, dims_in, dims_out, labels):
        'Initializes generator'
        self.batch_size = batch_size
        self.data_path = data_path
        self.dims_in = dims_in
        self.dims_out = dims_out
        self.labels = labels
        self.n_classes = len(self.labels)
        self.trainset = [x[0:-1] for x in open(data_listing, "r")]
        self.num_lines = len(self.trainset)
        self.batches = int(np.floor(self.num_lines/self.batch_size))
        self.indexes = list(range(self.num_lines))
        self.normalize_eps = 1e-8
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_lines/self.batch_size))
        
    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size : (index+1)*self.batch_size]
        
        # Generate data
        X = self.__data_generation(indexes)

        return X
    
    def __iter__(self):
        'Create a generator that iterate over the Sequence.'
        for item in (self[i] for i in range(len(self))):
            yield item
 
        
    def __data_generation(self, items):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, self.dims_in[0]))
        spectrum = np.empty([self.batch_size,self.dims_out[0], self.dims_out[1]])
    
        # Generate data
        for i, ID in enumerate(items):
            # Store sample
            sample_rate, samples = wavfile.read(self.data_path+self.trainset[ID])

            # Downsampling (optional)
            # new_sample_rate = int(8e3)
            # samples = signal.resample(samples, int(new_sample_rate/sample_rate * samples.shape[0]))
            # sample_rate = new_sample_rate

            # VAD and move to 0
            mean = np.mean(abs(samples))
            first = np.argmax(abs(samples)>0.3*mean)
            last = len(samples) - int(np.argmax(abs(samples[::-1])>0.3*mean))
            samples = samples[first:last]

            # normalisation (per audio file)
            samples = samples - np.mean(samples)
            samples = samples/(self.normalize_eps+np.std(samples))

            # Zero padding
            X[i,] = np.pad(samples, (0, sample_rate-len(samples)), 'constant')
            
            # Get spectrum, returns (99,161) matrix (dependent on sample rate)
            window_size=20
            step_size=10
            eps=1e-10
            nperseg = int(round(window_size * sample_rate / 1e3))
            noverlap = int(round(step_size * sample_rate / 1e3))
            freqs, times, spec = signal.spectrogram(X[i,],
                                            fs=sample_rate,
                                            window='hann',
                                            nperseg=nperseg,
                                            noverlap=noverlap,
                                            detrend=False)
            spectrum[i,] = np.log(spec.T.astype(np.float32) + eps) # output (99,161)

    
        return  np.expand_dims(spectrum,-1)