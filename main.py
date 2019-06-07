"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : main.py
                    Main ML file for the 5LSL0 project
    Author        : Bart van Erp
    Date          : 06/06/2019

==============================================================================
"""

## Import arbitrary libraries
import numpy as np
import matplotlib.pyplot as plt

## Import tensorflow and keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv1D, Conv2D, MaxPooling1D, LeakyReLU, AvgPool2D, UpSampling2D, ReLU, MaxPooling2D, Reshape, Flatten, Activation
from tensorflow.keras.losses import MSE, categorical_crossentropy
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
    
## Import selfmade moduels
from Helpers.generator import DataGenerator, generator2

## Labels
labels=("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "silence", "unknown")

## Create generator
traingen1 = DataGenerator(data_path='Data/train/audio/',
                         data_listing='Data/train/lists/train_set.txt',
                         batch_size=64, 
                         dims=(16000,),
                         labels=labels)

traingen2 = generator2(data_path='Data/train/audio/',
                         data_listing='Data/train/lists/train_set.txt',
                         labels=labels,
                         batch_size=64)

testgen1 = DataGenerator(data_path='Data/train/audio/',
                         data_listing='Data/train/lists/test_set.txt',
                         batch_size=64, 
                         dims=(16000,),
                         labels=labels)

## Create test model
model = Sequential()
model.add(Conv1D(12, kernel_size=9, input_shape=(16000,1), padding="same"))
model.add(LeakyReLU())
model.add(MaxPooling1D(2))

model.add(Conv1D(12, kernel_size=9, padding="same"))
model.add(LeakyReLU())
model.add(MaxPooling1D(2))

model.add(Conv1D(12, kernel_size=9, padding="same"))
model.add(LeakyReLU())
model.add(MaxPooling1D(2))

model.add(Conv1D(12, kernel_size=7, padding="same"))
model.add(LeakyReLU())
model.add(MaxPooling1D(2))

model.add(Conv1D(12, kernel_size=7, padding="same"))
model.add(LeakyReLU())
model.add(MaxPooling1D(2))

model.add(Conv1D(12, kernel_size=7, padding="same"))
model.add(LeakyReLU())
model.add(MaxPooling1D(2))

model.add(Conv1D(12, kernel_size=7, padding="same"))
model.add(LeakyReLU())
model.add(MaxPooling1D(5))

model.add(Conv1D(12, kernel_size=7, padding="same"))
model.add(LeakyReLU())
model.add(MaxPooling1D(4))

model.add(Dense(1))
model.add(Flatten())
model.add(Activation('softmax'))
model.summary()
optimizer = Adam()

model.compile(optimizer=optimizer, loss=categorical_crossentropy)

# Train model on dataset
model.fit_generator(generator=traingen1,
                    validation_data=testgen1,
                    epochs=10)
