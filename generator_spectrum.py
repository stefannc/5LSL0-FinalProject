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
from scipy import signal
import random
import librosa
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

            # Zero padding
            X[i,] = np.pad(samples, (0, sample_rate-len(samples)), 'constant')
            
            # Get spectrum, returns (99,81) matrix (afhankelijk van sample rate)
            # ? only need the log of spec
            # ? freqs and times are/should be equal for all audio files
            # Nog even kijken hoe het spectrum uiteindelijk gereturned kan worden
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
            spectrum = np.log(spec.T.astype(np.float32) + eps) #Dit is dus de uiteindelijke output (99,81)

            # Store class
            y_temp = self.trainset[ID].split('/')[0]
            if y_temp in self.labels:
                y[i] = self.labels.index(y_temp)
            else:
                y[i] = len(self.labels)-1
    
        return  np.expand_dims(spectrum,-1), keras.utils.to_categorical(y, num_classes=self.n_classes)