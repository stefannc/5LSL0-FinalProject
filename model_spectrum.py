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
from generator_spectrum import DataGenerator
from model import deep_cnn


## Labels
#labels=("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "silence", "unknown")
labels=("yes", "no")
num_classes = len(labels)
batch_size = 64


## normalization options
normalization = {"subtract_mean": True,
                 "epsilon": 1e-8,           # prevent division by 0
                 "normalize": "95",        # "none", "var", "std", "95", "97"
                 "type": "sample"}          # "sample", "class"

## evaluation metrics
metrics=['acc']

# Data paths
data_path = 'train/audio/'
data_lists = ''
final_test_path = '../test_files/audio/' # unzipped train and test data

## Create generator
traingen1 = iter(DataGenerator(data_path=data_path,
                         data_listing=data_lists+'train_set.txt',
                         batch_size=64, 
                         dims_in=(16000,1),
                         dims_out=(99,161),
                         labels=labels))

testgen1 = iter(DataGenerator(data_path=data_path,
                         data_listing=data_lists+'test_set.txt',
                         batch_size=64, 
                         dims_in=(16000,1),
                         dims_out=(99,161),
                         labels=labels,
                        ))

## Create test model
shape = (99,161,1)
model = deep_cnn(shape, num_classes)

#optimizer = Adam(lr = 0.0001)
model.compile(optimizer='Adam', loss=categorical_crossentropy, metrics = metrics)

# Train model on dataset
num_lines = sum(1 for line in open(data_lists+'train_set.txt'))
history = model.fit_generator(generator=traingen1,
                              steps_per_epoch= int(np.ceil(num_lines/batch_size)),
                              epochs=3,
                              verbose = 1,
                              validation_data=testgen1,
                              validation_steps=4)

y_files = os.listdir(final_test_path)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()