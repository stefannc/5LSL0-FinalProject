# 5LSL0-FinalProject

Download data from https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data and put 'train' folder in 'Data'

# Aanpassingen in main om mfcc datagenerator te gebruiken:
dims_out=(13,32)
shape = (13,32,1)

#train model: 
from generator_mfcc import DataGenerator

#test model on our own made test set: 
from generator_mfcc_test import DataGenerator

#test model for final submission:
from generator_mfcc_finaltest import DataGenerator

# Aanpassingen in main om spectrum generator te gebruiken:
dims_out=(99,161)
shape = (99,161,1)

#train model:
from generator_spectrum import DataGenerator

#test model on our own made test set:
from generator_spectrum_test import DataGenerator

#test model for final submission:
from generator_spectrum_finaltest import DataGenerator
