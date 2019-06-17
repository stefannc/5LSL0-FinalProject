"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : datasplitter.py
                    Splits data in different sets and outputs txt files with 
                    paths from ./Data/train/audio
    Author        : Bart van Erp
    Date          : 06/06/2019

==============================================================================
"""

# import libraries
import os
import random
from tqdm import tqdm

# parameters
filepath_train = '.../Data/train'
validation_percentage = 10
testing_percentage = 10

# open .txt files
f_train = open(filepath_train+'/lists/train_set.txt',"w+")
f_test = open(filepath_train+'/lists/test_set.txt',"w+")
f_validation = open(filepath_train+'/lists/validation_set.txt',"w+")

# loop through all available folders
folders = os.listdir(filepath_train+'/audio')
for k1 in tqdm(folders, "processing folders"):
    
    # skip the background noise folder
    if k1 == "_background_noise_":
        continue
    
    # Loop through all available files
    files = os.listdir(filepath_train+'/audio/'+k1)
    for k2 in files:
        
        # calculate probability
        prob = random.random()*100
        
        # classify
        if prob > validation_percentage+testing_percentage:
            # train set
            f_train.write(k1+'/'+k2+'\n')
        elif prob < testing_percentage:
            # test set
            f_test.write(k1+'/'+k2+'\n')
        else:
            # validation set
            f_validation.write(k1+'/'+k2+'\n')

# close files
f_train.close()
f_test.close()
f_validation.close()