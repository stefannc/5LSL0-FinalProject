# 5LSL0 Final Assignment 
# datasplitter

# import libraries
import os
import random
from tqdm import tqdm

def datasplitter():
    # parameters
    filepath_train = 'Data/train'
    validation_percentage = 10
    testing_percentage = 10
    classes = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence']
    
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
        
        # loop through the unknown files
        elif k1 not in classes:
            files = os.listdir(filepath_train+'/audio/'+k1)
            
            for k2 in files:
                
                #calculate probability
                prob = random.random()*100
                
                #classify
                if prob > validation_percentage+testing_percentage:
                    #train set
                    f_train.write(k1+'/'+k2+'\n')
                elif prob < testing_percentage:
                    #test set
                    f_test.write(k1+'/'+k2+'\n')
                else:
                    #validation set
                    f_validation.write(k1+'/'+k2+'\n')
            
        
        # Loop through all available files
        else:
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
    return