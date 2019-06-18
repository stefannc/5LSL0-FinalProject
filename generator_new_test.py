"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : generator_new_test.py
                    Test script for the updated generator class
    Author        : Bart van Erp
    Date          : 17/06/2019

==============================================================================
"""

## Import arbitrary libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import openpyxl

## Import tensorflow and keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv1D, Conv2D, MaxPooling1D, LeakyReLU, AvgPool2D, UpSampling2D, ReLU, MaxPooling2D, Reshape, Flatten, Activation
from tensorflow.keras.losses import MSE, categorical_crossentropy
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

## Import selfmade moduels
from generator_new import DataGenerator
from model import deep_cnn


## Import options
from generator_options import train_speech_options, train_spectrum_options, train_mfcc_options, test_speech_options, test_spectrum_options, test_mfcc_options

## Create generators
traingen_speech = DataGenerator(train_speech_options)
traingen_spectrum = DataGenerator(train_spectrum_options)
traingen_mfcc = DataGenerator(train_mfcc_options)
testgen_speech = DataGenerator(test_speech_options)
testgen_spectrum = DataGenerator(test_spectrum_options)
testgen_mfcc = DataGenerator(test_mfcc_options)

traingen_speech[1]
traingen_spectrum[1]
traingen_mfcc[1]
testgen_speech[1]
testgen_spectrum[1]
testgen_mfcc[1]
