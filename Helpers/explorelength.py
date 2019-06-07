"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : explorelength.py
                    visualise all possible lengths of the data segments
    Author        : Bart van Erp
    Date          : 06/06/2019

==============================================================================
"""

## import libraries
from scipy.io import wavfile
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
#fs, data = wavfile.read('./output/audio.wav')

# parameters
filepath_train = '../Data/train'

# create struct to save lengths and sampling frequencies
lengths =[]
fs = []

# loop through all available folders
folders = os.listdir(filepath_train+'/audio')
for k1 in tqdm(folders, "processing folders"):
    
    # skip the background noise folder
    if k1 == "_background_noise_":
        continue
    
    # Loop through all available files
    files = os.listdir(filepath_train+'/audio/'+k1)
    for k2 in files:
        
        # get fs and data
        fs_temp, lengths_temp = wavfile.read(filepath_train+'/audio/'+k1+'/'+k2)
        lengths.append(len(lengths_temp))
        fs.append(fs_temp)

# plot barplot with occurences
x1, y1 = np.unique(fs, return_counts=True)
plt.figure();
plt.scatter(x1,y1)
plt.xlabel('fs')
plt.ylabel('occurences')
plt.ylim((0, y1[0]*1.1))
plt.xlim((0, 20000))
plt.grid();
plt.savefig('../Figures/occurences_fs.pdf')

x2, y2 = np.unique(lengths, return_counts=True)
plt.figure()
plt.scatter(x2,y2)
plt.xlabel('data length')
plt.ylabel('occurences')
plt.grid();
plt.savefig('../Figures/occurences_datalength.pdf')

