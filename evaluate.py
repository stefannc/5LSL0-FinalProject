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
import numpy as np
import openpyxl
from tqdm import trange
import csv



## Load model
model = load_model('../models/model3.h5')

## create classes
labels = ('yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown')

## create listing
f_test = open('../Data/test/lists/test_set.txt',"w+")
for k in os.listdir('../Data/test/audio'):
    f_test.write(k+'\n')
f_test.close()

print("listing created")

## create evaluation generator
gen = DataGenerator(test_all_options)
y_pred = model.predict_generator(gen,
                                 verbose=1)

print("predictions made")

y_pred = np.argmax(y_pred, axis=1)

np.savetxt('probs.csv', y_pred, delimiter=',')

datafiles = os.listdir('../Data/test/audio')
with open('model3.csv', mode='w', newline='') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow(["fname", "label"])
    for i, file in enumerate(datafiles):
        employee_writer.writerow([file, labels[y_pred[i]]])
