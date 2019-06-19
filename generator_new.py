"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : generator_new.py
                    Updated generator class
    Author        : Bart van Erp
    Date          : 17/06/2019

==============================================================================
"""

## import libraries
from scipy.io import wavfile
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
from scipy.signal import decimate
from scipy import signal
import librosa

## create class
class DataGenerator(keras.utils.Sequence):
    """
    Create data generator using the keras class, with all functions specified 
    by the options argument:
    options = {
    "return_label": True,            # return label: True, False
    "data_type": "mfcc",             # type of data output: "speech", "spectrum", "mfcc"
    "data_path": "./data/",          # path to data folder
    "lists_path": "./lists/",        # path to lists
    "batch_size": 64,                # number of data segments per batch
    "labels": (),                    # list of labels
    "dims_data": (),                 # dimensions of original data (+zero append)
    "dims_out": (),                  # output dimensions as tuple
    "shuffle_on_end": True,          # shuffle data order per epoch: True, False
    "VAD": True,                     # apply voice activity detection
    "downsample": 2,                 # downsample factor (disable: 1)
    "normalize_class": "class",      # normalize sample with respect to "class", "sample"
    "subtract_mean": True,           # subtract the mean of the signal
    "normalize_type": "std",         # normalize the signal by dividing by None, "std" (class) or None, "std", "var", "95", "98" (sample)
    "normalize_eps": 1e-8,           # constant to prevent division by zero
    "noise_var": 1                   # variance of additivie Gaussian noise
    }
    """    
    
    def __init__(self, options):
        """
        This function initializes the data generator by using the above-defined 
        options dictionary. Besides this it adds some frequently used variables 
        to the generator for more efficient processing.
        """
        
        # copy data
        self.return_label = options["return_label"]
        self.data_type = options["data_type"]
        self.data_path = options["data_path"]
        self.list_path = options["list_path"]
        self.batch_size = options["batch_size"]
        self.labels = options["labels"]
        self.dims_data= options["dims_data"]
        self.dims_output = options["dims_output"]
        self.shuffle_on_end = options["shuffle_on_end"]
        self.VAD = options["VAD"]
        self.downsample = options["downsample"]
        self.normalize_class = options["normalize_class"]
        self.subtract_mean = options["subtract_mean"]
        self.normalize_type = options["normalize_type"]
        self.normalize_eps = options["normalize_eps"]
        self.noise_var = options["noise_var"]
        
        # check whether dimensions are satisfactory to data type chosen
        if type(self.data_type) == tuple:
            if self.data_type == "speech":
                if np.size(options["dims_output"]) != 1:
                    print("ERROR: output dimensions inconsistent with data type")
            elif self.data_type == "spectrum":
                if np.size(options["dims_output"]) != 2:
                    print("ERROR: output dimensions inconsistent with data type")
            elif self.data_type == "mfcc":
                if np.size(options["dims_output"]) != 2:
                    print("ERROR: output dimensions inconsistent with data type")
            else:
                print("ERROR: unknown data type")
            
        # check if downsample factor is int
        if not isinstance(self.downsample, int):
            print("ERROR: downsample factor is not an integer")
        
        # get all file names of listing
        self.data_files = [x[0:-1] for x in open(self.list_path, "r")]
        self.sample_rate = 16000
        
        # determine order for first epoch
        self.order = self.on_epoch_end()
        
  
    def __len__(self):
        """
        This function calculates the amount of iterations required to loop once 
        through the entire data set with the specified batch size.
        """
        return len(self.data_files)//self.batch_size
        
    
    def __getitem__(self, index):
        """
        Gets a batch of data at location 'index', called by 
        DataGenerator[index]. Is used in __iter__() to create a generator 
        object.
        """
        data_indices = self.order[self.batch_size*index:self.batch_size*(index+1)]
        return self.generate_batch(data_indices)

        
    def __iter__(self):
        """
        Create a generator from the class that iterates over the different 
        batches.
        """
        for item in (self[i] for i in range(len(self))):
            yield item
            
            
    def on_epoch_end(self):
        """
        This function determines the order of the data processing over the 
        course of the epoch. If enabled, the function will shuffle the order
        randomly.
        """
        if self.shuffle_on_end:
            return np.random.permutation(len(self.data_files))
        else:
            return [i for i in range(len(self.data_files)-1)]
                    
        
    def generate_batch(self, indices):
        """
        This function generates a batch of samples at the indices in the data 
        set specified by 'indices'.
        """
       
        # create empty structs for data
        data = np.empty(shape=(self.batch_size, int(self.sample_rate/self.downsample)))
        if type(self.dims_output) == tuple:
            # single output
            X = np.empty(shape=((self.batch_size,)+self.dims_output))
        else:
            # multiple outputs
            X1 = np.empty(shape=((self.batch_size,)+self.dims_output[0]))
            X2 = np.empty(shape=((self.batch_size,)+self.dims_output[1]))
            X3 = np.empty(shape=((self.batch_size,)+self.dims_output[2]))
        y= np.empty(shape=(self.batch_size,), dtype=np.int8)
        
        # loop through data set and process data segments
        for i, index in enumerate(indices):
            
            # load data
            _, samples = wavfile.read(self.data_path+self.data_files[index])
            
            # get label (always needed for class normalization)
            
            y_temp = self.data_files[index].split('/')[0]
            if y_temp in self.labels:
                y[i] = int(self.labels.index(y_temp))
            else:
                y[i] = int(len(self.labels)-1)
                    
            # read class statistics if desired
            if self.normalize_class == "class":
                self.stats = pd.read_csv("Statistics/statistics_timedomain.csv")
            
            # apply voice activity detection and shift fragment to beginning
            if self.VAD:
                # find signal statistics and determine active voice segment
                mean = np.mean(abs(samples))
                first = np.argmax(abs(samples)>0.4*mean)
                last = len(samples) - int(np.argmax(abs(samples[::-1])>0.4*mean))
                # slice samples to retain voice
                samples = samples[first:last]

            # normalize data per class
            if self.normalize_class == "class":
                # subtract mean
                if self.subtract_mean:
                    samples = samples - self.stats[self.stats["class"]==self.labels[y[i]]]["mean"][y[i]]
                # normalize
                if self.normalize_type == None:
                    pass
                elif self.normalize_type == "std":
                    samples = samples/(self.normalize_eps + self.stats[self.stats["class"]==self.labels[y[i]]]["std"][y[i]])      
            # normalize per sample
            elif self.normalize_class == "sample":
                # subtract mean
                if self.subtract_mean:
                    samples = samples - np.mean(samples)
                    
                # normalize
                if self.normalize_type == None:
                    pass
                elif self.normalize_type == "std":
                    samples = samples/(self.normalize_eps+np.std(samples))
                elif self.normalize_type == "var": 
                    samples = samples/(self.normalize_eps+np.var(samples))
                elif self.normalize_type == "95":
                    samples = samples/(self.normalize_eps+np.quantile(samples, 0.95))
                elif self.normalize_type == "98":
                    samples = samples/(self.normalize_eps+np.quantile(samples, 0.98))
                else:
                    print("ERROR: normalization type invalid")
            else:
                print("ERROR: normalization class unknown")

            # zero-pad samples to 16000 samples
            samples = np.pad(samples, (0, self.sample_rate-len(samples)), 'constant')
            
            # downsample and save to data
            if self.downsample != 1:
                data[i,] = decimate(samples, self.downsample)
            else:
                data[i,] = samples
                
            # conversion to desired format
            if "speech" in self.data_type:
                if type(self.data_type)==tuple:
                    X[i,] = data[i]
                else:
                    X1[i,] = data[i]
            if "mfcc" in self.data_type:
                Mel = librosa.feature.melspectrogram(samples, sr=self.sample_rate, n_mels=128)
                log_S = librosa.power_to_db(Mel, ref=np.max)
                mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
                mfcc = librosa.feature.delta(mfcc, order=2)
                if type(self.data_type) == tuple:
                    X[i,] = mfcc
                else:
                    X3[i,] = mfcc
            if "spectrum" in self.data_type:
                window_size=20
                step_size=10
                eps=1e-10
                nperseg = int(round(window_size * self.sample_rate / 1e3))
                noverlap = int(round(step_size * self.sample_rate / 1e3))
                freqs, times, spec = signal.spectrogram(samples,
                                            fs=self.sample_rate,
                                            window='hann',
                                            nperseg=nperseg,
                                            noverlap=noverlap,
                                            detrend=False)
                if type(self.data_type) == tuple:
                    X[i,] = np.log(spec.T.astype(np.float32) + eps) 
                else:
                    X2[i,] = np.log(spec.T.astype(np.float32) + eps) 
            
        # add noise to data           
        if self.data_type ==tuple:
            if self.noise_var != 0:
                X = X + np.random.normal(scale=self.noise_var, size=np.shape(X))
        else:
            if self.noise_var != 0:
                X1 = X1 + np.random.normal(scale=self.noise_var, size=np.shape(X1))
                X2 = X2 + np.random.normal(scale=self.noise_var, size=np.shape(X2))
                X3 = X3 + np.random.normal(scale=self.noise_var, size=np.shape(X3))
        
        # return entries
        if self.data_type==tuple:
            if self.return_label:
                return np.expand_dims(X,-1), keras.utils.to_categorical(y, num_classes=len(self.labels))
            else:
                return np.expand_dims(X,-1)
        else:
            if self.return_label:
                return [np.expand_dims(X1,-1), np.expand_dims(X2,-1), np.expand_dims(X3,-1)], keras.utils.to_categorical(y, num_classes=len(self.labels))
            else:
                return [np.expand_dims(X1,-1), np.expand_dims(X2,-1), np.expand_dims(X3,-1)]
        
