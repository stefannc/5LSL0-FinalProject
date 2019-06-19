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
from tensorflow.keras.layers import Input, Dense, Conv1D, Conv2D, Dropout, MaxPooling1D, LeakyReLU, AvgPool2D, UpSampling2D, ReLU, MaxPooling2D, Reshape, Flatten, Activation, Concatenate
from tensorflow.keras.losses import MSE, categorical_crossentropy
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

## Import selfmade moduels
from generator_new import DataGenerator
from model import deep_cnn2d_mfcc, deep_cnn1d_speech, deep_cnn2d_spectrum

## Import options
from generator_options import train_all_options, val_all_options, test_all_options

## Create generators
traingen_all = DataGenerator(train_all_options)
valgen_all = DataGenerator(val_all_options)
testgen_all = DataGenerator(test_all_options)

## Create models
model1 = deep_cnn1d_speech(shape=(16000,1), num_classes=12)
model2 = deep_cnn2d_spectrum(shape=(99,161,1), num_classes=12)
model3 = deep_cnn2d_mfcc(shape=(13,32,1), num_classes=12)

## Combine models
output_merged = Concatenate()([model1.output,model2.output,model3.output])
output_merged = Reshape((output_merged._shape_as_list()[1:]+[1]))(output_merged)
output_merged = Dropout(0.5)(output_merged)
output_merged = Conv1D(32, 3, padding="same", activation = 'relu', dilation_rate=64)(output_merged)
output_merged = Conv1D(32, 3, padding="same", activation = 'relu')(output_merged)
output_merged = MaxPooling1D(2, strides = 2, padding = 'same')(output_merged)
output_merged = Conv1D(32, 3, padding="same", activation = 'relu')(output_merged)
output_merged = Conv1D(32, 3, padding="same", activation = 'relu')(output_merged)
output_merged = MaxPooling1D(2, strides = 2, padding = 'same')(output_merged)
output_merged = Flatten()(output_merged)
finaloutput = Dense(12, activation='softmax')(output_merged)

Modeltotal = Model([model1.input,model2.input,model3.input], finaloutput)
Modeltotal.summary()

## Compile model
Modeltotal.compile(optimizer='Adam', 
                   loss = categorical_crossentropy,
                   metrics=['acc'])

## Train model
history = Modeltotal.fit_generator(generator=traingen_all,
                                   verbose = 1,
                                   epochs = 2,
                                   validation_data=valgen_all)

Modeltotal.save('../models/model3.h5')

# create figures (loss)
plt.figure()
plt.plot(history.history["loss"], label="training loss")
plt.plot(history.history["val_loss"], label="validation loss")
plt.grid()
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig("../figures/loss3.pdf")

plt.figure()
plt.plot(history.history["acc"], label="training accuracy")
plt.plot(history.history["val_acc"], label="validation accuracy")
plt.grid()
plt.legend()
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.savefig("../figures/acc3.pdf")
