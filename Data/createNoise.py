import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import os

#Global variables
TIME = 1
FS = 16000
N_SAMPLES = TIME * FS
MU_MU = -0.1644 #mean of mean
SIGMA_MU = 2022.657 #mean of std
MU_SIGMA = 2.951 #std of mean
SIGMA_SIGMA = 151.505 #std of std

def generate(mu, sigma):
    x = np.random.normal(mu, sigma, size = N_SAMPLES)
    return x

if 'noise' not in os.listdir('train/audio/'):
    print('No noise folder found in data, folder will be created')
    os.mkdir('Data/train/audio/noise') 
    print('Noise folder is created')

n_output = 1;
means = np.random.normal(MU_MU, MU_SIGMA, size = n_output)
stds = np.random.normal(SIGMA_MU, SIGMA_SIGMA, size = n_output)

for i in range(0, n_output):
    noise = generate(means[i], stds[i])
    noise.astype(np.int16)
    name = 'noise_' + str(i+1) + '.wav'
    wav.write('train/audio/noise/' + name, FS, noise)

print('Successfully created', i+1, 'noise .wav files')