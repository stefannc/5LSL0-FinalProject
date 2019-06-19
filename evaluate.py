"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : evaluation.py
                    Evaluates model on test set
    Author        : Bart van Erp
    Date          : 19/06/2019

==============================================================================
"""

## Import modules
from tensorflow.keras.models import load_model
from generator_options import test_all_options
from generator_new import DataGenerator
import os



## Load model
model = load_model('../models/model1.h5')

## create classes
labels = ('yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown')

## create listing
f_test = open('../Data/test/lists/test_set.txt',"w+")
for k in os.listdir('../Data/test/audio'):
    f_test.write(k+'\n')
f_test.close()

## create evaluation generator
gen = DataGenerator(test_all_options)
y_pred = model.predict_generator(gen)
